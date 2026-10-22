// .glb / .gltf -> ModelData. Runs on loader threads, so no GPU or physics calls here.
#define CGLTF_IMPLEMENTATION
#include <cgltf.h>
#define STB_IMAGE_IMPLEMENTATION
#define STBI_ONLY_PNG
#define STBI_ONLY_JPEG
#include <stb_image.h>

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <map>
#include <mutex>

#include "world.h"

#include <meshoptimizer.h>

// Weld duplicate vertices, then reorder for the post-transform cache (within each material run, so the
// per-triangle material list stays valid) and for vertex fetch locality. Runs on the loader threads.
static void optimizeMesh(MeshData& md) {
    const size_t stride = 8 * sizeof(float), nv = md.verts.size() / 8, ni = md.idx.size();
    if (!nv || !ni) return;
    std::vector<uint32_t> remap(nv);
    size_t unique = meshopt_generateVertexRemap(remap.data(), md.idx.data(), ni, md.verts.data(), nv, stride);
    std::vector<float> verts(unique * 8);
    meshopt_remapVertexBuffer(verts.data(), md.verts.data(), nv, stride, remap.data());
    meshopt_remapIndexBuffer(md.idx.data(), md.idx.data(), ni, remap.data());
    for (size_t t0 = 0; t0 < md.triMat.size();) {
        size_t t1 = t0;
        while (t1 < md.triMat.size() && md.triMat[t1] == md.triMat[t0]) t1++;
        meshopt_optimizeVertexCache(&md.idx[t0 * 3], &md.idx[t0 * 3], (t1 - t0) * 3, unique);
        t0 = t1;
    }
    meshopt_optimizeVertexFetch(verts.data(), md.idx.data(), ni, verts.data(), unique, stride);
    md.verts.swap(verts);
}

static constexpr uint32_t MAX_TEX_SIZE = 2048;

// Minimal lookup in a node's extras JSON: {"asset":"door","flatten":1,"falloff":4}
bool extraStr(const char* json, const char* key, std::string& out) {
    if (!json) return false;
    std::string k = std::string("\"") + key + "\"";
    const char* p = strstr(json, k.c_str());
    if (!p) return false;
    p = strchr(p + k.size(), ':');
    if (!p) return false;
    p++;
    while (*p == ' ' || *p == '\t' || *p == '\n') p++;
    if (*p == '"') {
        const char* e = strchr(p + 1, '"');
        if (!e) return false;
        out.assign(p + 1, e);
    } else {
        const char* e = p;
        while (*e && *e != ',' && *e != '}' && *e != ' ' && *e != '\n') e++;
        out.assign(p, e);
    }
    return true;
}
float extraNum(const char* json, const char* key, float def) {
    std::string s;
    return extraStr(json, key, s) ? (s == "true" ? 1.f : s == "false" ? 0.f : strtof(s.c_str(), nullptr)) : def;
}

static Xf nodeXf(const cgltf_node* n) {
    float m[16];  // column-major
    cgltf_node_transform_world(n, m);
    Xf x;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 4; c++) x.m[r][c] = m[c * 4 + r];
    return x;
}

static bool decodeImage(const cgltf_image* img, const std::string& baseDir, TextureData& t) {
    int w, h, ch;
    stbi_uc* px = nullptr;
    if (img->buffer_view) {
        auto* bv = img->buffer_view;
        px = stbi_load_from_memory((const stbi_uc*)bv->buffer->data + bv->offset, (int)bv->size, &w, &h, &ch, 4);
    } else if (img->uri && strncmp(img->uri, "data:", 5) != 0) {
        std::string uri = img->uri;
        cgltf_decode_uri(uri.data());
        px = stbi_load((baseDir + uri.c_str()).c_str(), &w, &h, &ch, 4);
    }
    if (!px) return false;
    // box-downscale oversized textures (memory; no block compression yet)
    uint32_t W = (uint32_t)w, H = (uint32_t)h;
    std::vector<uint8_t> cur(px, px + (size_t)W * H * 4);
    stbi_image_free(px);
    while (W > MAX_TEX_SIZE || H > MAX_TEX_SIZE) {
        uint32_t nw = std::max(1u, W / 2), nh = std::max(1u, H / 2);
        std::vector<uint8_t> nx((size_t)nw * nh * 4);
        for (uint32_t y = 0; y < nh; y++)
            for (uint32_t x = 0; x < nw; x++)
                for (int c = 0; c < 4; c++) {
                    uint32_t x0 = std::min(2 * x, W - 1), x1 = std::min(2 * x + 1, W - 1), y0 = std::min(2 * y, H - 1), y1 = std::min(2 * y + 1, H - 1);
                    nx[((size_t)y * nw + x) * 4 + c] = (uint8_t)((cur[((size_t)y0 * W + x0) * 4 + c] + cur[((size_t)y0 * W + x1) * 4 + c] +
                                                                 cur[((size_t)y1 * W + x0) * 4 + c] + cur[((size_t)y1 * W + x1) * 4 + c] + 2) / 4);
                }
        cur.swap(nx);
        W = nw;
        H = nh;
    }
    t.w = W;
    t.h = H;
    t.rgba = std::move(cur);
    return true;
}

bool loadGltf(const std::string& path, ModelData& out, std::string& err) {
    cgltf_options opt{};
    cgltf_data* data = nullptr;
    if (cgltf_parse_file(&opt, path.c_str(), &data) != cgltf_result_success) { err = "parse failed"; return false; }
    if (cgltf_load_buffers(&opt, data, path.c_str()) != cgltf_result_success) { err = "buffers failed"; cgltf_free(data); return false; }
    std::string baseDir = path.substr(0, path.find_last_of('/') + 1);

    // materials (+ the textures they use; one TextureData per image and colour space)
    std::map<std::pair<const cgltf_image*, bool>, int> texOf;
    auto useTex = [&](const cgltf_texture_view& v, bool linear) {
        const cgltf_image* img = v.texture ? v.texture->image : nullptr;
        if (!img) return -1;
        auto it = texOf.find({img, linear});
        if (it == texOf.end()) {
            TextureData t;
            int id = -1;
            if (decodeImage(img, baseDir, t)) { t.linear = linear; id = (int)out.textures.size(); out.textures.push_back(std::move(t)); }
            it = texOf.emplace(std::make_pair(img, linear), id).first;
        }
        return it->second;
    };
    for (size_t i = 0; i < data->materials_count; i++) {
        const cgltf_material& gm = data->materials[i];
        MaterialData m;
        if (gm.has_pbr_metallic_roughness) {
            auto& pbr = gm.pbr_metallic_roughness;
            memcpy(m.color, pbr.base_color_factor, sizeof m.color);
            m.tex = useTex(pbr.base_color_texture, false);
            m.rough = pbr.roughness_factor;
            m.metal = pbr.metallic_factor;
            m.ormTex = useTex(pbr.metallic_roughness_texture, true);
        }
        m.normalTex = useTex(gm.normal_texture, true);
        if (m.normalTex >= 0) m.normalScale = gm.normal_texture.scale;
        m.occTex = useTex(gm.occlusion_texture, true);
        if (m.occTex >= 0) m.occStrength = gm.occlusion_texture.scale;
        m.emisTex = useTex(gm.emissive_texture, false);
        float es = gm.has_emissive_strength ? gm.emissive_strength.emissive_strength : 1.f;
        for (int k = 0; k < 3; k++) m.emissive[k] = gm.emissive_factor[k] * es;
        if (gm.has_clearcoat) { m.clearcoat = gm.clearcoat.clearcoat_factor; m.clearcoatRough = gm.clearcoat.clearcoat_roughness_factor; }
        if (gm.has_sheen) {
            memcpy(m.sheenColor, gm.sheen.sheen_color_factor, sizeof m.sheenColor);
            m.sheen = std::max({m.sheenColor[0], m.sheenColor[1], m.sheenColor[2]});
        }
        if (gm.has_anisotropy) { m.anisotropy = gm.anisotropy.anisotropy_strength; m.anisoRot = gm.anisotropy.anisotropy_rotation; }
        if (gm.has_transmission) m.transmission = gm.transmission.transmission_factor;
        if (gm.has_ior) m.ior = gm.ior.ior;
        if (gm.has_specular) m.specular = gm.specular.specular_factor;
        if (gm.has_diffuse_transmission) m.translucency = gm.diffuse_transmission.diffuse_transmission_factor;
        if (gm.extras.data) m.heightScale = extraNum(gm.extras.data, "heightScale", 0.f);
        out.materials.push_back(m);
    }
    uint32_t defaultMat = (uint32_t)out.materials.size();
    { MaterialData dm; dm.color[0] = dm.color[1] = dm.color[2] = 0.7f; out.materials.push_back(dm); }

    // meshes: all triangle primitives of a glTF mesh merged into one
    for (size_t mi = 0; mi < data->meshes_count; mi++) {
        const cgltf_mesh& gm = data->meshes[mi];
        MeshData md;
        for (size_t pi = 0; pi < gm.primitives_count; pi++) {
            const cgltf_primitive& p = gm.primitives[pi];
            if (p.type != cgltf_primitive_type_triangles) continue;
            const cgltf_accessor *pos = nullptr, *nrm = nullptr, *uv = nullptr;
            for (size_t a = 0; a < p.attributes_count; a++) {
                const auto& at = p.attributes[a];
                if (at.type == cgltf_attribute_type_position) pos = at.data;
                else if (at.type == cgltf_attribute_type_normal) nrm = at.data;
                else if (at.type == cgltf_attribute_type_texcoord && at.index == 0) uv = at.data;
            }
            if (!pos) continue;
            uint32_t base = (uint32_t)(md.verts.size() / 8);
            for (size_t v = 0; v < pos->count; v++) {
                float f[8] = {0, 0, 0, 0, 0, 0, 0, 0};
                cgltf_accessor_read_float(pos, v, f, 3);
                if (nrm) cgltf_accessor_read_float(nrm, v, f + 3, 3);
                if (uv) cgltf_accessor_read_float(uv, v, f + 6, 2);
                md.verts.insert(md.verts.end(), f, f + 8);
            }
            size_t first = md.idx.size();
            if (p.indices)
                for (size_t k = 0; k < p.indices->count; k++) md.idx.push_back(base + (uint32_t)cgltf_accessor_read_index(p.indices, k));
            else
                for (size_t k = 0; k < pos->count; k++) md.idx.push_back(base + (uint32_t)k);
            md.idx.resize(first + (md.idx.size() - first) / 3 * 3);
            uint32_t mat = p.material ? (uint32_t)(p.material - data->materials) : defaultMat;
            md.triMat.insert(md.triMat.end(), (md.idx.size() - first) / 3, mat);
            if (!nrm) {  // area-weighted vertex normals
                for (size_t k = first; k < md.idx.size(); k += 3) {
                    float* a = &md.verts[md.idx[k] * 8];
                    float* b = &md.verts[md.idx[k + 1] * 8];
                    float* c = &md.verts[md.idx[k + 2] * 8];
                    float e1[3] = {b[0] - a[0], b[1] - a[1], b[2] - a[2]}, e2[3] = {c[0] - a[0], c[1] - a[1], c[2] - a[2]};
                    float n[3] = {e1[1] * e2[2] - e1[2] * e2[1], e1[2] * e2[0] - e1[0] * e2[2], e1[0] * e2[1] - e1[1] * e2[0]};
                    for (float* v : {a, b, c})
                        for (int j = 0; j < 3; j++) v[3 + j] += n[j];
                }
                for (size_t v = base; v < md.verts.size() / 8; v++) {
                    float* n = &md.verts[v * 8 + 3];
                    float l = sqrtf(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
                    for (int j = 0; j < 3; j++) n[j] = l > 0 ? n[j] / l : (j == 1);
                }
            }
        }
        optimizeMesh(md);
        for (int j = 0; j < 3; j++) { md.bmin[j] = 1e30f; md.bmax[j] = -1e30f; }
        for (size_t v = 0; v < md.verts.size(); v += 8)
            for (int j = 0; j < 3; j++) {
                md.bmin[j] = std::min(md.bmin[j], md.verts[v + j]);
                md.bmax[j] = std::max(md.bmax[j], md.verts[v + j]);
            }
        out.meshes.push_back(std::move(md));
    }

    // nodes of the default scene
    for (size_t i = 0; i < data->nodes_count; i++) {
        const cgltf_node* n = &data->nodes[i];
        const char* ex = n->extras.data;
        Xf x = nodeXf(n);
        std::string asset;
        if (extraStr(ex, "asset", asset) && !asset.empty()) {
            InstanceData in;
            in.asset = asset;
            in.xf = x;
            in.collide = extraNum(ex, "collide", 1) != 0;
            if (n->name) in.name = n->name;
            if (ex) in.extras = ex;
            out.instances.push_back(in);
            continue;  // the node's mesh is just a Blender-side proxy
        }
        std::string pv;
        if (!n->mesh && (extraStr(ex, "tree", pv) || extraStr(ex, "rock", pv))) {
            PropData pd;
            pd.kind = extraStr(ex, "tree", pv) ? PROP_TREE : PROP_ROCK;
            pd.variant = pd.kind == PROP_TREE ? (int)extraNum(ex, "tree", 0) : 0;
            pd.xf = x;
            out.props.push_back(pd);
            continue;
        }
        if (extraNum(ex, "bake", 0) != 0) {
            // unit cube [-1, 1] -> world AABB of the transformed box
            ProbeVolume v;
            for (int k = 0; k < 3; k++) {
                float e = fabsf(x.m[k][0]) + fabsf(x.m[k][1]) + fabsf(x.m[k][2]);
                v.mn[k] = x.m[k][3] - e;
                v.mx[k] = x.m[k][3] + e;
            }
            v.name = n->name ? n->name : "volume";
            out.volumes.push_back(v);
            continue;
        }
        if (extraNum(ex, "flatten", 0) != 0) {
            // unit cube [-1, 1] -> footprint AABB of the transformed box
            float ex_ = fabsf(x.m[0][0]) + fabsf(x.m[0][2]), ez = fabsf(x.m[2][0]) + fabsf(x.m[2][2]);
            Zone z{x.m[0][3] - ex_, x.m[2][3] - ez, x.m[0][3] + ex_, x.m[2][3] + ez, x.m[1][3], extraNum(ex, "falloff", 8.f)};
            extraStr(ex, "mat", z.mat);
            z.tile = extraNum(ex, "tile", 2.f);
            out.zones.push_back(z);
            continue;
        }
        if (n->mesh && out.meshes[n->mesh - data->meshes].idx.size()) {
            InstanceData in;
            in.mesh = (int)(n->mesh - data->meshes);
            in.xf = x;
            in.collide = extraNum(ex, "collide", 1) != 0;
            if (n->name) in.name = n->name;
            if (ex) in.extras = ex;
            out.meshes[in.mesh].strength = extraNum(ex, "strength", 1.f);
            out.instances.push_back(in);
        }
        if (n->light && n->light->type != cgltf_light_type_directional) {
            LightData L;
            for (int k = 0; k < 3; k++) { L.pos[k] = x.m[k][3]; L.color[k] = n->light->color[k] * n->light->intensity; }
            out.lights.push_back(L);
        }
    }
    cgltf_free(data);
    return true;
}

// ---------------------------------------------------------------- asset aliases (renames / moves)
static std::mutex aliasMu;
static std::map<std::string, std::string> aliases;
static std::filesystem::file_time_type aliasTime{};
static std::string aliasDir;

static void loadAliases(const std::string& worldDir) {  // under aliasMu
    std::error_code ec;
    std::string path = worldDir + "assets/aliases.txt";
    auto t = std::filesystem::last_write_time(path, ec);
    if (ec) { if (aliasDir != worldDir) { aliases.clear(); aliasDir = worldDir; } return; }
    if (aliasDir == worldDir && t == aliasTime) return;
    aliasDir = worldDir;
    aliasTime = t;
    aliases.clear();
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        size_t tab = line.find('\t');
        if (tab != std::string::npos) aliases[line.substr(0, tab)] = line.substr(tab + 1);
    }
}
std::string resolveAsset(const std::string& worldDir, const std::string& name) {
    std::lock_guard lk(aliasMu);
    loadAliases(worldDir);
    std::string n = name;
    for (int i = 0; i < 32; i++) {  // follow chains (renamed twice), bounded against cycles
        auto it = aliases.find(n);
        if (it == aliases.end()) break;
        n = it->second;
    }
    return n;
}
void addAssetAlias(const std::string& worldDir, const std::string& from, const std::string& to) {
    std::lock_guard lk(aliasMu);
    if (from == to) return;
    loadAliases(worldDir);
    aliases.erase(to);  // `to` is a real asset now (a rename back must not loop)
    for (auto& [k, v] : aliases) if (v == from) v = to;  // older names follow it
    aliases[from] = to;
    std::ofstream f(worldDir + "assets/aliases.txt", std::ios::trunc);
    for (auto& [k, v] : aliases) f << k << '\t' << v << '\n';
    f.close();
    std::error_code ec;
    aliasTime = std::filesystem::last_write_time(worldDir + "assets/aliases.txt", ec);
}

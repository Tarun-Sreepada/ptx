// ModelData -> .glb (the editor's save). One glTF mesh per MeshData, split into a primitive per material run;
// textures re-encoded as PNG; lights as KHR_lights_punctual; volumes, zones and props as mesh-less nodes with the
// same extras gltf_load.cpp reads back. Positions are already in world space (cells) so nodes carry the instance
// transforms as matrices.
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <cstdarg>
#include <map>
#include <set>
#include <string>

#include "world/world.h"

static std::string fmt(const char* f, ...) {
    char b[1024];
    va_list a;
    va_start(a, f);
    vsnprintf(b, sizeof b, f, a);
    va_end(a);
    return b;
}

namespace {

struct Bin {
    std::vector<uint8_t> data;
    std::string views;  // JSON bufferViews
    int nViews = 0;
    int add(const void* p, size_t n, int target) {
        while (data.size() % 4) data.push_back(0);
        size_t off = data.size();
        data.insert(data.end(), (const uint8_t*)p, (const uint8_t*)p + n);
        char b[128];
        if (target) snprintf(b, sizeof b, "%s{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu,\"target\":%d}", nViews ? "," : "", off, n, target);
        else snprintf(b, sizeof b, "%s{\"buffer\":0,\"byteOffset\":%zu,\"byteLength\":%zu}", nViews ? "," : "", off, n);
        views += b;
        return nViews++;
    }
};

std::string esc(const std::string& s) {
    std::string o;
    for (char c : s) {
        if (c == '"' || c == '\\') o += '\\';
        if ((unsigned char)c >= 0x20) o += c;
    }
    return o;
}

std::string matrix(const Xf& x) {  // column-major 4x4
    char b[512];
    snprintf(b, sizeof b, "[%g,%g,%g,0,%g,%g,%g,0,%g,%g,%g,0,%g,%g,%g,1]", x.m[0][0], x.m[1][0], x.m[2][0], x.m[0][1], x.m[1][1], x.m[2][1],
             x.m[0][2], x.m[1][2], x.m[2][2], x.m[0][3], x.m[1][3], x.m[2][3]);
    return b;
}

// box node: unit cube [-1, 1] scaled to the AABB (what the loader turns back into a volume / zone)
Xf boxXf(const float mn[3], const float mx[3]) {
    Xf x = xfIdentity();
    for (int k = 0; k < 3; k++) { x.m[k][k] = std::max((mx[k] - mn[k]) * 0.5f, 1e-3f); x.m[k][3] = (mn[k] + mx[k]) * 0.5f; }
    return x;
}

}  // namespace

bool writeGlb(const std::string& path, const ModelData& m, std::string& err) {
    Bin bin;
    std::string accessors, meshes, images, textures, materials, nodes, lights;
    int nAcc = 0, nNodes = 0, nLights = 0;
    auto sep = [](std::string& s) { if (!s.empty()) s += ","; };
    auto accessor = [&](int view, int comp, size_t count, const char* type, const float* mn, const float* mx) {
        sep(accessors);
        char b[512];
        int n = snprintf(b, sizeof b, "{\"bufferView\":%d,\"componentType\":%d,\"count\":%zu,\"type\":\"%s\"", view, comp, count, type);
        if (mn) n += snprintf(b + n, sizeof b - n, ",\"min\":[%g,%g,%g],\"max\":[%g,%g,%g]", mn[0], mn[1], mn[2], mx[0], mx[1], mx[2]);
        snprintf(b + n, sizeof b - n, "}");
        accessors += b;
        return nAcc++;
    };

    for (size_t i = 0; i < m.textures.size(); i++) {
        const TextureData& t = m.textures[i];
        int len = 0;
        unsigned char* png = stbi_write_png_to_mem(t.rgba.data(), (int)t.w * 4, (int)t.w, (int)t.h, 4, &len);
        if (!png) { err = "png encode failed"; return false; }
        int v = bin.add(png, (size_t)len, 0);
        STBIW_FREE(png);
        sep(images);
        images += "{\"bufferView\":" + std::to_string(v) + ",\"mimeType\":\"image/png\"}";
        sep(textures);
        textures += "{\"sampler\":0,\"source\":" + std::to_string(i) + "}";
    }
    std::set<std::string> matExts;
    for (auto& mt : m.materials) {
        sep(materials);
        auto texRef = [&](const char* key, int t, const char* extra = "") {
            return t >= 0 && t < (int)m.textures.size() ? fmt(",\"%s\":{\"index\":%d%s}", key, t, extra) : std::string();
        };
        std::string j = fmt("{\"pbrMetallicRoughness\":{\"baseColorFactor\":[%g,%g,%g,%g],\"metallicFactor\":%g,\"roughnessFactor\":%g",
                            mt.color[0], mt.color[1], mt.color[2], mt.color[3], mt.metal, mt.rough);
        j += texRef("baseColorTexture", mt.tex) + texRef("metallicRoughnessTexture", mt.ormTex) + "}";
        j += texRef("normalTexture", mt.normalTex, fmt(",\"scale\":%g", mt.normalScale).c_str());
        j += texRef("occlusionTexture", mt.occTex, fmt(",\"strength\":%g", mt.occStrength).c_str());
        j += texRef("emissiveTexture", mt.emisTex);
        std::string ext;
        auto addExt = [&](const char* name, const std::string& body) {
            ext += (ext.empty() ? "" : ",") + fmt("\"%s\":{%s}", name, body.c_str());
            matExts.insert(name);
        };
        float e = std::max({mt.emissive[0], mt.emissive[1], mt.emissive[2]});
        if (e > 0.f) {
            float s = std::max(e, 1.f);
            j += fmt(",\"emissiveFactor\":[%g,%g,%g]", mt.emissive[0] / s, mt.emissive[1] / s, mt.emissive[2] / s);
            if (s > 1.f) addExt("KHR_materials_emissive_strength", fmt("\"emissiveStrength\":%g", s));
        }
        if (mt.clearcoat > 0.f) addExt("KHR_materials_clearcoat", fmt("\"clearcoatFactor\":%g,\"clearcoatRoughnessFactor\":%g", mt.clearcoat, mt.clearcoatRough));
        if (mt.sheen > 0.f) addExt("KHR_materials_sheen", fmt("\"sheenColorFactor\":[%g,%g,%g]", mt.sheenColor[0], mt.sheenColor[1], mt.sheenColor[2]));
        if (mt.anisotropy != 0.f) addExt("KHR_materials_anisotropy", fmt("\"anisotropyStrength\":%g,\"anisotropyRotation\":%g", mt.anisotropy, mt.anisoRot));
        if (mt.transmission > 0.f) addExt("KHR_materials_transmission", fmt("\"transmissionFactor\":%g", mt.transmission));
        if (mt.ior != 1.5f) addExt("KHR_materials_ior", fmt("\"ior\":%g", mt.ior));
        if (mt.specular != 1.f) addExt("KHR_materials_specular", fmt("\"specularFactor\":%g", mt.specular));
        if (mt.translucency > 0.f) addExt("KHR_materials_diffuse_transmission", fmt("\"diffuseTransmissionFactor\":%g", mt.translucency));
        if (!ext.empty()) j += ",\"extensions\":{" + ext + "}";
        if (mt.heightScale > 0.f || mt.sss > 0.f) {
            std::string ex;
            if (mt.heightScale > 0.f) ex += fmt("\"heightScale\":%g", mt.heightScale);
            if (mt.sss > 0.f) ex += fmt("%s\"sss\":%g", ex.empty() ? "" : ",", mt.sss);
            j += ",\"extras\":{" + ex + "}";
        }
        materials += j + "}";
    }

    // meshes: shared vertex accessors, one index accessor per material run
    std::vector<bool> used(m.meshes.size(), false);
    for (auto& in : m.instances) if (in.mesh >= 0 && in.mesh < (int)m.meshes.size()) used[in.mesh] = true;
    std::vector<int> meshIdx(m.meshes.size(), -1);
    int nMeshes = 0;
    for (size_t i = 0; i < m.meshes.size(); i++) {
        const MeshData& md = m.meshes[i];
        if (!used[i] || md.idx.empty()) continue;
        size_t nv = md.verts.size() / 8;
        std::vector<float> pos(nv * 3), nrm(nv * 3), uv(nv * 2);
        float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
        for (size_t v = 0; v < nv; v++) {
            for (int k = 0; k < 3; k++) {
                pos[v * 3 + k] = md.verts[v * 8 + k];
                nrm[v * 3 + k] = md.verts[v * 8 + 3 + k];
                mn[k] = std::min(mn[k], pos[v * 3 + k]); mx[k] = std::max(mx[k], pos[v * 3 + k]);
            }
            uv[v * 2] = md.verts[v * 8 + 6];
            uv[v * 2 + 1] = md.verts[v * 8 + 7];
        }
        int aP = accessor(bin.add(pos.data(), pos.size() * 4, 34962), 5126, nv, "VEC3", mn, mx);
        int aN = accessor(bin.add(nrm.data(), nrm.size() * 4, 34962), 5126, nv, "VEC3", nullptr, nullptr);
        int aT = accessor(bin.add(uv.data(), uv.size() * 4, 34962), 5126, nv, "VEC2", nullptr, nullptr);
        std::string prims;
        size_t nt = md.idx.size() / 3;
        for (size_t t0 = 0; t0 < nt;) {
            uint32_t mat = md.triMat.empty() ? 0 : md.triMat[t0];
            size_t t1 = t0;
            while (t1 < nt && (md.triMat.empty() ? 0 : md.triMat[t1]) == mat) t1++;
            int aI = accessor(bin.add(&md.idx[t0 * 3], (t1 - t0) * 12, 34963), 5125, (t1 - t0) * 3, "SCALAR", nullptr, nullptr);
            sep(prims);
            prims += "{\"attributes\":{\"POSITION\":" + std::to_string(aP) + ",\"NORMAL\":" + std::to_string(aN) + ",\"TEXCOORD_0\":" + std::to_string(aT) +
                     "},\"indices\":" + std::to_string(aI) + (mat < m.materials.size() ? ",\"material\":" + std::to_string(mat) : std::string()) + "}";
            t0 = t1;
        }
        sep(meshes);
        meshes += "{\"primitives\":[" + prims + "]}";
        meshIdx[i] = nMeshes++;
    }

    auto node = [&](const std::string& name, const Xf& x, const std::string& extra) {
        sep(nodes);
        nodes += "{\"name\":\"" + esc(name) + "\",\"matrix\":" + matrix(x) + extra + "}";
        nNodes++;
    };
    for (size_t i = 0; i < m.instances.size(); i++) {
        const InstanceData& in = m.instances[i];
        std::string name = in.name.empty() ? (in.asset.empty() ? "mesh" : in.asset) + std::to_string(i) : in.name;
        std::string ex;
        if (!in.asset.empty()) {
            ex = ",\"extras\":{\"asset\":\"" + esc(in.asset) + "\"" + (in.collide ? "" : ",\"collide\":0") + "}";
        } else {
            if (in.mesh < 0 || meshIdx[in.mesh] < 0) continue;
            ex = ",\"mesh\":" + std::to_string(meshIdx[in.mesh]);
            if (!in.extras.empty()) ex += ",\"extras\":" + in.extras;
            else if (!in.collide) ex += ",\"extras\":{\"collide\":0}";
        }
        node(name, in.xf, ex);
    }
    for (size_t i = 0; i < m.lights.size(); i++) {
        const LightData& L = m.lights[i];
        float I = std::max({L.color[0], L.color[1], L.color[2], 1e-6f});
        char b[256];
        snprintf(b, sizeof b, "%s{\"type\":\"point\",\"color\":[%g,%g,%g],\"intensity\":%g}", nLights ? "," : "", L.color[0] / I, L.color[1] / I, L.color[2] / I, I);
        lights += b;
        Xf x = xfIdentity();
        for (int k = 0; k < 3; k++) x.m[k][3] = L.pos[k];
        node("light" + std::to_string(i), x, ",\"extensions\":{\"KHR_lights_punctual\":{\"light\":" + std::to_string(nLights++) + "}}");
    }
    for (auto& v : m.volumes) node(v.name.empty() ? "bake" : v.name, boxXf(v.mn, v.mx), ",\"extras\":{\"bake\":1}");
    for (size_t i = 0; i < m.zones.size(); i++) {
        const Zone& z = m.zones[i];
        float mn[3] = {z.minx, z.height - 0.5f, z.minz}, mx[3] = {z.maxx, z.height + 0.5f, z.maxz};
        Xf x = boxXf(mn, mx);
        x.m[1][3] = z.height;
        char b[384];
        if (z.mat.empty()) snprintf(b, sizeof b, ",\"extras\":{\"flatten\":1,\"falloff\":%g}", z.falloff);
        else snprintf(b, sizeof b, ",\"extras\":{\"flatten\":1,\"falloff\":%g,\"mat\":\"%s\",\"tile\":%g}", z.falloff, z.mat.c_str(), z.tile);
        node("zone" + std::to_string(i), x, b);
    }
    for (size_t i = 0; i < m.props.size(); i++) {
        const PropData& p = m.props[i];
        node((p.kind == PROP_TREE ? "tree" : "rock") + std::to_string(i), p.xf,
             p.kind == PROP_TREE ? ",\"extras\":{\"tree\":" + std::to_string(p.variant) + "}" : std::string(",\"extras\":{\"rock\":1}"));
    }

    std::string sceneNodes;
    for (int i = 0; i < nNodes; i++) sceneNodes += (i ? "," : "") + std::to_string(i);
    while (bin.data.size() % 4) bin.data.push_back(0);
    std::string json = "{\"asset\":{\"version\":\"2.0\",\"generator\":\"ptx editor\"}";
    if (nLights) matExts.insert("KHR_lights_punctual");
    if (!matExts.empty()) {
        std::string used;
        for (auto& e : matExts) used += (used.empty() ? "\"" : ",\"") + e + "\"";
        json += ",\"extensionsUsed\":[" + used + "]";
    }
    if (nLights) json += ",\"extensions\":{\"KHR_lights_punctual\":{\"lights\":[" + lights + "]}}";
    json += ",\"scene\":0,\"scenes\":[{\"nodes\":[" + sceneNodes + "]}],\"nodes\":[" + nodes + "]";
    if (!meshes.empty()) json += ",\"meshes\":[" + meshes + "]";
    if (!materials.empty()) json += ",\"materials\":[" + materials + "]";
    if (!textures.empty()) json += ",\"textures\":[" + textures + "],\"images\":[" + images + "],\"samplers\":[{\"wrapS\":10497,\"wrapT\":10497}]";
    if (!accessors.empty()) json += ",\"accessors\":[" + accessors + "]";
    if (bin.nViews) json += ",\"bufferViews\":[" + bin.views + "],\"buffers\":[{\"byteLength\":" + std::to_string(bin.data.size()) + "}]";
    json += "}";
    while (json.size() % 4) json += ' ';

    std::string tmp = path + ".tmp";
    FILE* f = fopen(tmp.c_str(), "wb");
    if (!f) { err = "can't write " + tmp; return false; }
    uint32_t hdr[3] = {0x46546C67u, 2u, (uint32_t)(12 + 8 + json.size() + (bin.nViews ? 8 + bin.data.size() : 0))};
    fwrite(hdr, 4, 3, f);
    uint32_t jc[2] = {(uint32_t)json.size(), 0x4E4F534Au};
    fwrite(jc, 4, 2, f);
    fwrite(json.data(), 1, json.size(), f);
    if (bin.nViews) {
        uint32_t bc[2] = {(uint32_t)bin.data.size(), 0x004E4942u};
        fwrite(bc, 4, 2, f);
        fwrite(bin.data.data(), 1, bin.data.size(), f);
    }
    bool ok = fclose(f) == 0;
    if (!ok || rename(tmp.c_str(), path.c_str()) != 0) { err = "can't replace " + path; return false; }
    return true;
}

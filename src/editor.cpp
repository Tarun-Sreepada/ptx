#include "editor.h"

#include <imgui.h>

#include <stb_image.h>
#include <stb_image_write.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <cstdarg>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <map>

#include <meshoptimizer.h>

#include "matlib.h"
#include "worldmap.h"
#include "physics.h"
#include "trees.h"

namespace editor {
namespace {

constexpr float DEG = 3.14159265f / 180.f;

enum Kind { K_PRIM, K_MESH, K_ASSET, K_LIGHT, K_VOLUME, K_ZONE, K_TREE, K_ROCK, K_COUNT };
const char* KIND_NAME[K_COUNT] = {"Primitive", "Mesh", "Asset", "Light", "Bake volume", "Flatten zone", "Tree", "Rock"};
enum Shape { BOX, CYLINDER };
const char* TREE_NAME[trees::VARIANTS] = {"Species A, seed 1", "Species A, seed 2", "Species B, seed 1", "Species B, seed 2",
                                          "Species C, seed 1", "Species C, seed 2", "Species D, seed 1", "Species D, seed 2"};

// Everything in a cell as one editable list. Transforms are position + Euler angles (degrees, applied
// yaw * pitch * roll) + scale; primitives keep their size in the mesh (scale 1).
struct Item {
    Kind kind = K_PRIM;
    std::string name;
    float pos[3] = {}, rot[3] = {}, scale[3] = {1, 1, 1};  // rot: pitch (x), yaw (y), roll (z)
    // primitive
    int shape = BOX, sides = 24;
    float size[3] = {1, 1, 1};  // box x, y, z; cylinder diameter x, height y, diameter z
    std::string mat = "brick";
    float uvScale = 1.f, tint[3] = {1, 1, 1};
    float strength = 1.f;
    bool collide = true;
    // mesh / asset
    int mesh = -1;
    std::string asset, extras;
    // light
    float color[3] = {1.f, 0.85f, 0.65f}, intensity = 400.f;
    // zone
    float falloff = 8.f;
    std::string zoneMat;  // ground texture, empty = none
    float zoneTile = 2.f;
    // tree
    int variant = 0;
    // where it went in the live model (in-place moves)
    int slot = -1;
};

struct ECell {
    int cx = 0, cz = 0;
    std::vector<Item> items;
    std::vector<MeshData> meshes;  // the cell's own (imported) meshes, for K_MESH items
    std::vector<MaterialData> mats;
    std::vector<TextureData> texs;
    bool modified = false, dirty = false;  // differs from what was loaded / from disk
    bool rebuild = false, refresh = false;
    std::shared_ptr<Model> live;
    std::vector<std::vector<Item>> undo;
};

Host host;
bool on = false;
std::map<std::pair<int, int>, ECell> cells;
struct Sel { int cx = 0, cz = 0, idx = -1; } sel;
View view;
float grid = 0.25f, rotStep = 15.f;
std::vector<std::pair<int, int>> reverts;  // cells to reload from disk in apply()
bool snap = true, showVolumes = true, showAll = false;
bool groundSnap = true;  // placed / dragged assets and trees / rocks sit on the terrain...
float sinkDepth = 0.f;   // ...this far into it (m): trunks, foundations, rocks bedded in
std::string newMat = "brick";
std::string status;
std::vector<std::string> assetNames;
std::vector<std::string> assetDirs;  // folders under assets/, relative
char importPath[512] = "";
double lastRebuild = 0;
// dragging
bool dragging = false, dragVertical = false, dragMoved = false;
float dragPlaneY = 0, dragOff[3] = {}, dragN[3] = {}, dragP0[3] = {};

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---------------------------------------------------------------- math
Xf compose(const float p[3], const float r[3], const float s[3]) {
    float cx = cosf(r[0] * DEG), sx = sinf(r[0] * DEG), cy = cosf(r[1] * DEG), sy = sinf(r[1] * DEG), cz = cosf(r[2] * DEG), sz = sinf(r[2] * DEG);
    float R[3][3] = {{cy * cz + sy * sx * sz, -cy * sz + sy * sx * cz, sy * cx},
                     {cx * sz, cx * cz, -sx},
                     {-sy * cz + cy * sx * sz, sy * sz + cy * sx * cz, cy * cx}};
    Xf x;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) x.m[i][j] = R[i][j] * s[j];
        x.m[i][3] = p[i];
    }
    return x;
}
void decompose(const Xf& x, float p[3], float r[3], float s[3]) {
    float R[3][3];
    for (int j = 0; j < 3; j++) {
        s[j] = sqrtf(x.m[0][j] * x.m[0][j] + x.m[1][j] * x.m[1][j] + x.m[2][j] * x.m[2][j]);
        for (int i = 0; i < 3; i++) R[i][j] = s[j] > 1e-8f ? x.m[i][j] / s[j] : 0.f;
    }
    for (int i = 0; i < 3; i++) p[i] = x.m[i][3];
    r[0] = asinf(std::clamp(-R[1][2], -1.f, 1.f)) / DEG;
    r[1] = atan2f(R[0][2], R[2][2]) / DEG;
    r[2] = atan2f(R[1][0], R[1][1]) / DEG;
}
Xf inverse(const Xf& a) {
    const float(*m)[4] = a.m;
    float det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    float id = 1.f / (fabsf(det) > 1e-20f ? det : 1e-20f);
    Xf r;
    r.m[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * id; r.m[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id; r.m[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id;
    r.m[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * id; r.m[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id; r.m[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * id;
    r.m[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * id; r.m[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * id; r.m[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id;
    for (int i = 0; i < 3; i++) r.m[i][3] = -(r.m[i][0] * m[0][3] + r.m[i][1] * m[1][3] + r.m[i][2] * m[2][3]);
    return r;
}
Xf itemXf(const Item& it) {
    static const float one[3] = {1, 1, 1};
    return compose(it.pos, it.rot, it.kind == K_PRIM ? one : it.scale);
}
bool snapInvert = false;  // Shift held: snapping temporarily the other way
bool moveOnly = false;    // the pending refresh only moved meshes / assets / lights
float snapv(float v, float g) { return (snap != snapInvert) && g > 0 ? roundf(v / g) * g : v; }
int cellOf(float x) { return (int)floorf(x / stream::CELL); }
std::string cellPath(int cx, int cz) { return host.worldDir + "cells/" + std::to_string(cx) + "_" + std::to_string(cz) + ".glb"; }

// ---------------------------------------------------------------- primitives
// Box from y = 0 up, centred in x / z; per-face planar UVs in metres / uvScale. Cylinder: n-sided prism, smooth
// sides, flat caps. Both convex, so breakStatic shatters them as one piece each.
void genPrim(const Item& it, uint32_t mat, MeshData& m) {
    m = {};
    auto vtx = [&](float x, float y, float z, float nx, float ny, float nz, float u, float v) {
        m.verts.insert(m.verts.end(), {x, y, z, nx, ny, nz, u / it.uvScale, v / it.uvScale});
        return (uint32_t)(m.verts.size() / 8 - 1);
    };
    auto quad = [&](uint32_t a, uint32_t b, uint32_t c, uint32_t d) {
        m.idx.insert(m.idx.end(), {a, b, c, a, c, d});
        m.triMat.insert(m.triMat.end(), {mat, mat});
    };
    float hx = it.size[0] * 0.5f, hy = it.size[1], hz = it.size[2] * 0.5f;
    if (it.shape == BOX) {
        // each face: normal, then 4 corners CCW from outside, with their (u, v)
        struct F { float n[3]; float c[4][3]; int va; };
        const F faces[6] = {
            {{1, 0, 0}, {{hx, 0, hz}, {hx, 0, -hz}, {hx, hy, -hz}, {hx, hy, hz}}, 1},
            {{-1, 0, 0}, {{-hx, 0, -hz}, {-hx, 0, hz}, {-hx, hy, hz}, {-hx, hy, -hz}}, 1},
            {{0, 1, 0}, {{-hx, hy, hz}, {hx, hy, hz}, {hx, hy, -hz}, {-hx, hy, -hz}}, 2},
            {{0, -1, 0}, {{-hx, 0, -hz}, {hx, 0, -hz}, {hx, 0, hz}, {-hx, 0, hz}}, 2},
            {{0, 0, 1}, {{-hx, 0, hz}, {hx, 0, hz}, {hx, hy, hz}, {-hx, hy, hz}}, 1},
            {{0, 0, -1}, {{hx, 0, -hz}, {-hx, 0, -hz}, {-hx, hy, -hz}, {hx, hy, -hz}}, 1}};
        for (auto& f : faces) {
            uint32_t v[4];
            for (int k = 0; k < 4; k++) {
                const float* c = f.c[k];
                // u runs left to right seen from outside, v up the wall (texture upright, not mirrored)
                float u = f.n[0] > 0 ? -c[2] : f.n[0] < 0 ? c[2] : f.n[2] > 0 ? c[0] : f.n[2] < 0 ? -c[0] : c[0];
                v[k] = vtx(c[0], c[1], c[2], f.n[0], f.n[1], f.n[2], u, f.va == 1 ? c[1] : c[2]);
            }
            quad(v[0], v[1], v[2], v[3]);
        }
    } else {
        int n = std::clamp(it.sides, 3, 128);
        float rx = hx, rz = hz, circ = 3.14159f * (rx + rz);
        uint32_t base = (uint32_t)(m.verts.size() / 8);
        for (int i = 0; i <= n; i++) {
            float a = 6.2831853f * i / n, c = cosf(a), s = sinf(a);
            float nx = c / std::max(rx, 1e-4f), nz = s / std::max(rz, 1e-4f), l = sqrtf(nx * nx + nz * nz);
            float u = circ * i / n;
            vtx(c * rx, 0, s * rz, nx / l, 0, nz / l, u, 0);
            vtx(c * rx, hy, s * rz, nx / l, 0, nz / l, u, hy);
        }
        for (int i = 0; i < n; i++) {
            uint32_t a = base + 2 * i;
            quad(a, a + 1, a + 3, a + 2);
        }
        for (int cap = 0; cap < 2; cap++) {
            float y = cap ? hy : 0.f, ny = cap ? 1.f : -1.f;
            uint32_t c = vtx(0, y, 0, 0, ny, 0, 0, 0), first = (uint32_t)(m.verts.size() / 8);
            for (int i = 0; i < n; i++) {
                float a = 6.2831853f * i / n;
                vtx(cosf(a) * rx, y, sinf(a) * rz, 0, ny, 0, cosf(a) * rx, sinf(a) * rz);
            }
            for (int i = 0; i < n; i++) {
                uint32_t p = first + i, q = first + (i + 1) % n;
                if (cap) m.idx.insert(m.idx.end(), {c, q, p});
                else m.idx.insert(m.idx.end(), {c, p, q});
                m.triMat.push_back(mat);
            }
        }
    }
    for (int k = 0; k < 3; k++) { m.bmin[k] = 1e30f; m.bmax[k] = -1e30f; }
    for (size_t i = 0; i < m.verts.size(); i += 8)
        for (int k = 0; k < 3; k++) { m.bmin[k] = std::min(m.bmin[k], m.verts[i + k]); m.bmax[k] = std::max(m.bmax[k], m.verts[i + k]); }
    m.strength = it.strength;
    m.breakKind = breakKindOf(it.mat);
}

std::string fmt(const char* f, ...) {
    char b[1024];
    va_list ap;
    va_start(ap, f);
    vsnprintf(b, sizeof b, f, ap);
    va_end(ap);
    return b;
}
std::string primExtras(const Item& it) {
    return fmt("{\"prim\":\"%s\",\"sx\":%g,\"sy\":%g,\"sz\":%g,\"sides\":%d,\"mat\":\"%s\",\"uv\":%g,\"tr\":%g,\"tg\":%g,\"tb\":%g,\"strength\":%g%s}",
               it.shape == BOX ? "box" : "cylinder", it.size[0], it.size[1], it.size[2], it.sides, it.mat.c_str(), it.uvScale, it.tint[0],
               it.tint[1], it.tint[2], it.strength, it.collide ? "" : ",\"collide\":0");
}
// set / replace one numeric key in a flat extras object
std::string setKey(std::string json, const char* key, float v) {
    std::string k = std::string("\"") + key + "\"";
    std::string val = fmt("%g", v);
    size_t p = json.find(k);
    if (json.empty() || json[0] != '{') json = "{}";
    if (p == std::string::npos) {
        json.insert(1, k + ":" + val + (json.size() > 2 ? "," : ""));
        return json;
    }
    size_t c = json.find(':', p + k.size());
    size_t e = json.find_first_of(",}", c + 1);
    json.replace(c + 1, e - c - 1, val);
    return json;
}

// ---------------------------------------------------------------- cells
ECell importCell(int cx, int cz, const std::shared_ptr<Model>& live) {
    ECell c;
    c.cx = cx; c.cz = cz;
    c.live = live;
    if (!live) return c;
    const ModelData& d = live->data;
    c.meshes = d.meshes;
    for (auto& m : c.meshes) m.shape = nullptr;  // belong to the live model
    c.mats = d.materials;
    c.texs = d.textures;
    for (auto& in : d.instances) {
        Item it;
        it.name = in.name;
        it.collide = in.collide;
        decompose(in.xf, it.pos, it.rot, it.scale);
        const char* ex = in.extras.c_str();
        std::string prim;
        if (!in.asset.empty()) {
            it.kind = K_ASSET;
            it.asset = in.asset;
        } else if (extraStr(ex, "prim", prim)) {
            it.kind = K_PRIM;
            it.shape = prim == "cylinder" ? CYLINDER : BOX;
            it.size[0] = extraNum(ex, "sx", 1); it.size[1] = extraNum(ex, "sy", 1); it.size[2] = extraNum(ex, "sz", 1);
            it.sides = (int)extraNum(ex, "sides", 24);
            extraStr(ex, "mat", it.mat);
            it.uvScale = extraNum(ex, "uv", 1);
            it.tint[0] = extraNum(ex, "tr", 1); it.tint[1] = extraNum(ex, "tg", 1); it.tint[2] = extraNum(ex, "tb", 1);
            it.strength = extraNum(ex, "strength", 1);
            for (float& s : it.scale) s = 1.f;
        } else {
            it.kind = K_MESH;
            it.mesh = in.mesh;
            it.extras = in.extras;
            it.strength = in.mesh >= 0 ? d.meshes[in.mesh].strength : 1.f;
        }
        c.items.push_back(it);
    }
    for (auto& L : d.lights) {
        Item it;
        it.kind = K_LIGHT;
        memcpy(it.pos, L.pos, 12);
        it.intensity = std::max({L.color[0], L.color[1], L.color[2], 1e-6f});
        for (int k = 0; k < 3; k++) it.color[k] = L.color[k] / it.intensity;
        c.items.push_back(it);
    }
    for (auto& v : d.volumes) {
        Item it;
        it.kind = K_VOLUME;
        it.name = v.name;
        for (int k = 0; k < 3; k++) { it.pos[k] = (v.mn[k] + v.mx[k]) * 0.5f; it.scale[k] = (v.mx[k] - v.mn[k]) * 0.5f; }
        c.items.push_back(it);
    }
    for (auto& z : d.zones) {
        Item it;
        it.kind = K_ZONE;
        it.pos[0] = (z.minx + z.maxx) * 0.5f; it.pos[1] = z.height; it.pos[2] = (z.minz + z.maxz) * 0.5f;
        it.scale[0] = (z.maxx - z.minx) * 0.5f; it.scale[1] = 0.5f; it.scale[2] = (z.maxz - z.minz) * 0.5f;
        it.falloff = z.falloff;
        it.zoneMat = z.mat;
        it.zoneTile = z.tile;
        c.items.push_back(it);
    }
    for (auto& p : d.props) {
        Item it;
        it.kind = p.kind == PROP_TREE ? K_TREE : K_ROCK;
        it.variant = p.variant;
        decompose(p.xf, it.pos, it.rot, it.scale);
        c.items.push_back(it);
    }
    return c;
}

// cell (cx, cz) for editing: imported from the stream once loaded; a cell with no file starts empty
ECell* open(int cx, int cz) {
    auto it = cells.find({cx, cz});
    if (it != cells.end()) return &it->second;
    auto live = stream::cellModel(cx, cz);
    if (!live && std::filesystem::exists(cellPath(cx, cz))) return nullptr;  // still loading (or out of range)
    return &(cells[{cx, cz}] = importCell(cx, cz, live));
}

// The cell's items -> a model: the cell's own meshes still in use (materials and textures compacted), a mesh per
// primitive (materials per material + tint, the library texture once per name), instances, lights, volumes, zones,
// props. Each item's slot (instance / light / volume / zone / prop index) is recorded for in-place moves.
void build(ECell& c, ModelData& d, bool shapes) {
    d = {};
    std::map<int, int> meshMap, matMap, texMap;
    auto useTex = [&](int t) {
        if (t < 0 || t >= (int)c.texs.size()) return -1;
        auto [e, fresh] = texMap.emplace(t, (int)d.textures.size());
        if (fresh) d.textures.push_back(c.texs[t]);
        return e->second;
    };
    auto useMat = [&](uint32_t m) {
        auto [e, fresh] = matMap.emplace((int)m, (int)d.materials.size());
        if (fresh) {
            MaterialData md = m < c.mats.size() ? c.mats[m] : MaterialData{};
            forEachTex(md, useTex);
            d.materials.push_back(md);
        }
        return (uint32_t)e->second;
    };
    std::map<std::string, int> libTex, primMat;
    auto primMaterial = [&](const Item& it) {
        std::string key = it.mat + fmt("|%g|%g|%g", it.tint[0], it.tint[1], it.tint[2]);
        auto pm = primMat.find(key);
        if (pm != primMat.end()) return (uint32_t)pm->second;
        auto lt = libTex.find(it.mat);
        if (lt == libTex.end()) {  // albedo, normal + height, ORM: three consecutive textures
            lt = libTex.emplace(it.mat, (int)d.textures.size()).first;
            const matlib::Maps& mp = matlib::maps(it.mat);
            d.textures.push_back(matlib::texture(it.mat));
            d.textures.push_back(mp.normal);
            d.textures.push_back(mp.orm);
        }
        MaterialData md;
        memcpy(md.color, it.tint, 12);
        md.tex = lt->second;
        md.normalTex = lt->second + 1;
        md.ormTex = md.occTex = lt->second + 2;
        md.rough = md.metal = 1.f;  // factors x the ORM texture
        md.heightScale = matlib::maps(it.mat).heightScale;
        primMat[key] = (int)d.materials.size();
        d.materials.push_back(md);
        return (uint32_t)d.materials.size() - 1;
    };
    for (auto& it : c.items) {
        it.slot = -1;
        switch (it.kind) {
        case K_PRIM: {
            MeshData m;
            genPrim(it, primMaterial(it), m);
            InstanceData in;
            in.mesh = (int)d.meshes.size();
            d.meshes.push_back(std::move(m));
            in.xf = itemXf(it);
            in.collide = it.collide;
            in.name = it.name;
            in.extras = primExtras(it);
            it.slot = (int)d.instances.size();
            d.instances.push_back(in);
            break;
        }
        case K_MESH: {
            if (it.mesh < 0 || it.mesh >= (int)c.meshes.size()) break;
            auto [e, fresh] = meshMap.emplace(it.mesh, (int)d.meshes.size());
            if (fresh) {
                MeshData m = c.meshes[it.mesh];
                for (auto& t : m.triMat) t = useMat(t);
                m.shape = nullptr;
                d.meshes.push_back(std::move(m));
            }
            d.meshes[e->second].strength = it.strength;
            InstanceData in;
            in.mesh = e->second;
            in.xf = itemXf(it);
            in.collide = it.collide;
            in.name = it.name;
            in.extras = it.strength != 1.f || !it.extras.empty() ? setKey(it.extras, "strength", it.strength) : std::string();
            if (!it.collide) in.extras = setKey(in.extras, "collide", 0);
            it.slot = (int)d.instances.size();
            d.instances.push_back(in);
            break;
        }
        case K_ASSET: {
            InstanceData in;
            in.asset = it.asset;
            in.xf = itemXf(it);
            in.collide = it.collide;
            in.name = it.name;
            it.slot = (int)d.instances.size();
            d.instances.push_back(in);
            break;
        }
        case K_LIGHT: {
            LightData L;
            memcpy(L.pos, it.pos, 12);
            for (int k = 0; k < 3; k++) L.color[k] = it.color[k] * it.intensity;
            it.slot = (int)d.lights.size();
            d.lights.push_back(L);
            break;
        }
        case K_VOLUME: {
            ProbeVolume v;
            for (int k = 0; k < 3; k++) { v.mn[k] = it.pos[k] - fabsf(it.scale[k]); v.mx[k] = it.pos[k] + fabsf(it.scale[k]); }
            v.name = it.name.empty() ? fmt("vol_%d_%d_%zu", c.cx, c.cz, d.volumes.size()) : it.name;
            it.slot = (int)d.volumes.size();
            d.volumes.push_back(v);
            break;
        }
        case K_ZONE: {
            Zone z{it.pos[0] - fabsf(it.scale[0]), it.pos[2] - fabsf(it.scale[2]), it.pos[0] + fabsf(it.scale[0]), it.pos[2] + fabsf(it.scale[2]), it.pos[1], it.falloff, it.zoneMat, it.zoneTile};
            it.slot = (int)d.zones.size();
            d.zones.push_back(z);
            break;
        }
        case K_TREE: case K_ROCK: {
            PropData p;
            p.kind = it.kind == K_TREE ? PROP_TREE : PROP_ROCK;
            p.variant = it.variant;
            float r[3] = {0, it.rot[1], 0}, s[3] = {it.scale[0], it.scale[0], it.scale[0]};
            p.xf = compose(it.pos, r, s);
            it.slot = (int)d.props.size();
            d.props.push_back(p);
            break;
        }
        default: break;
        }
    }
    if (d.materials.empty()) d.materials.push_back(MaterialData{});
    if (!shapes) return;
    std::vector<bool> coll(d.meshes.size(), false);
    for (auto& in : d.instances) if (in.mesh >= 0 && in.collide) coll[in.mesh] = true;
    for (size_t i = 0; i < d.meshes.size(); i++) {
        auto& m = d.meshes[i];
        if (coll[i] && !m.idx.empty())
            m.shape = physics::meshShape(m.verts.data(), 8, (uint32_t)(m.verts.size() / 8), m.idx.data(), (uint32_t)(m.idx.size() / 3));
    }
}

void freeModel(std::shared_ptr<Model> m) { host.retire(std::move(m)); }

bool pushLive(ECell& c) {
    auto m = std::make_shared<Model>();
    build(c, m->data, true);
    m->state = 1;
    for (auto& md : m->data.meshes) m->bytes += md.verts.size() * 4 + md.idx.size() * 8;
    if (!host.upload(*m)) {
        for (auto& md : m->data.meshes) if (md.shape) physics::releaseShape(md.shape);
        status = "upload failed (out of mesh / material / texture slots)";
        return false;
    }
    std::vector<std::shared_ptr<Model>> freed;
    stream::setCellModel(c.cx, c.cz, m, freed);
    for (auto& f : freed) freeModel(f);
    c.live = m;
    lastRebuild = now();
    return true;
}

// move what a slot points at in the live model, then re-place the cell
bool moveInPlace(ECell& c, const Item& it) {
    if (!c.live || it.slot < 0 || stream::cellModel(c.cx, c.cz) != c.live) return false;
    ModelData& d = c.live->data;
    switch (it.kind) {
    case K_PRIM: case K_MESH: case K_ASSET:
        if (it.slot >= (int)d.instances.size()) return false;
        d.instances[it.slot].xf = itemXf(it);
        break;
    case K_LIGHT:
        if (it.slot >= (int)d.lights.size()) return false;
        memcpy(d.lights[it.slot].pos, it.pos, 12);
        break;
    case K_VOLUME:
        if (it.slot >= (int)d.volumes.size()) return false;
        for (int k = 0; k < 3; k++) { d.volumes[it.slot].mn[k] = it.pos[k] - fabsf(it.scale[k]); d.volumes[it.slot].mx[k] = it.pos[k] + fabsf(it.scale[k]); }
        break;
    case K_ZONE: {
        if (it.slot >= (int)d.zones.size()) return false;
        Zone& z = d.zones[it.slot];
        z = {it.pos[0] - fabsf(it.scale[0]), it.pos[2] - fabsf(it.scale[2]), it.pos[0] + fabsf(it.scale[0]), it.pos[2] + fabsf(it.scale[2]), it.pos[1], it.falloff, it.zoneMat, it.zoneTile};
        break;
    }
    case K_TREE: case K_ROCK: {
        if (it.slot >= (int)d.props.size()) return false;
        float r[3] = {0, it.rot[1], 0}, s[3] = {it.scale[0], it.scale[0], it.scale[0]};
        d.props[it.slot].xf = compose(it.pos, r, s);
        break;
    }
    default: return false;
    }
    return true;
}

Item* selected() {
    if (sel.idx < 0) return nullptr;
    auto it = cells.find({sel.cx, sel.cz});
    if (it == cells.end() || sel.idx >= (int)it->second.items.size()) { sel.idx = -1; return nullptr; }
    return &it->second.items[sel.idx];
}
ECell* selCell() {
    auto it = cells.find({sel.cx, sel.cz});
    return it == cells.end() ? nullptr : &it->second;
}
void pushUndo(ECell& c) {
    c.undo.push_back(c.items);
    if (c.undo.size() > 64) c.undo.erase(c.undo.begin());
}
void changed(ECell& c, bool geometry) {
    c.modified = c.dirty = true;
    if (geometry) c.rebuild = true;
    else c.refresh = true;
}

// an item that moved out of its cell's square goes to the cell it's in now
void rehome(ECell& c, int idx) {
    Item& it = c.items[idx];
    int nx = cellOf(it.pos[0]), nz = cellOf(it.pos[2]);
    if (it.kind == K_MESH || (nx == c.cx && nz == c.cz)) return;  // imported meshes stay with their cell's mesh data
    ECell* d = open(nx, nz);
    if (!d) return;
    pushUndo(*d);
    d->items.push_back(it);
    c.items.erase(c.items.begin() + idx);
    changed(c, true);
    changed(*d, true);
    if (sel.cx == c.cx && sel.cz == c.cz && sel.idx == idx) sel = {nx, nz, (int)d->items.size() - 1};
}

// ---------------------------------------------------------------- picking
void localBounds(const ECell& c, const Item& it, float mn[3], float mx[3]) {
    auto set = [&](float a, float b, float cc, float d, float e, float f) { mn[0] = a; mn[1] = b; mn[2] = cc; mx[0] = d; mx[1] = e; mx[2] = f; };
    switch (it.kind) {
    case K_PRIM: set(-it.size[0] * 0.5f, 0, -it.size[2] * 0.5f, it.size[0] * 0.5f, it.size[1], it.size[2] * 0.5f); return;
    case K_MESH:
        if (it.mesh >= 0 && it.mesh < (int)c.meshes.size()) { memcpy(mn, c.meshes[it.mesh].bmin, 12); memcpy(mx, c.meshes[it.mesh].bmax, 12); return; }
        break;
    case K_ASSET:
        if (Model* m = stream::findModel(assetFile(host.worldDir, it.asset))) {
            for (int k = 0; k < 3; k++) { mn[k] = 1e30f; mx[k] = -1e30f; }
            for (auto& in : m->data.instances) {
                if (in.mesh < 0) continue;
                float a[3], b[3];
                xfAabb(in.xf, m->data.meshes[in.mesh].bmin, m->data.meshes[in.mesh].bmax, a, b);
                for (int k = 0; k < 3; k++) { mn[k] = std::min(mn[k], a[k]); mx[k] = std::max(mx[k], b[k]); }
            }
            if (mn[0] <= mx[0]) return;
        }
        break;
    case K_LIGHT: set(-0.15f, -0.15f, -0.15f, 0.15f, 0.15f, 0.15f); return;
    case K_VOLUME: case K_ZONE: set(-1, -1, -1, 1, 1, 1); return;
    case K_TREE: set(-1, 0, -1, 1, 10, 1); return;
    case K_ROCK: set(-0.9f, -0.3f, -0.9f, 0.9f, 0.7f, 0.9f); return;
    default: break;
    }
    set(-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f);
}
Xf boundsXf(const Item& it) {
    if (it.kind == K_LIGHT) { float s[3] = {1, 1, 1}; return compose(it.pos, it.rot, s); }
    if (it.kind == K_VOLUME || it.kind == K_ZONE) { float r[3] = {0, 0, 0}; return compose(it.pos, r, it.scale); }
    if (it.kind == K_TREE || it.kind == K_ROCK) { float r[3] = {0, it.rot[1], 0}, s[3] = {it.scale[0], it.scale[0], it.scale[0]}; return compose(it.pos, r, s); }
    return itemXf(it);
}
// Put the item's base on the terrain under its centre, sunk by sinkDepth (its local bounds give where the base is,
// so a model with its origin mid-height still lands on its bottom).
void snapToGround(const ECell& c, Item& it) {
    float mn[3], mx[3], a[3], b[3];
    localBounds(c, it, mn, mx);
    xfAabb(boundsXf(it), mn, mx, a, b);
    it.pos[1] += host.height(it.pos[0], it.pos[2]) - sinkDepth - a[1];
}
bool groundable(const Item& it) { return it.kind == K_ASSET || it.kind == K_TREE || it.kind == K_ROCK; }  // not primitives: they stack
bool pickable(const Item& it) { return showVolumes || (it.kind != K_VOLUME && it.kind != K_ZONE); }

// ray vs the item's oriented box: entry distance (0 if inside), or < 0
float rayItem(const ECell& c, const Item& it, const float o[3], const float d[3]) {
    float mn[3], mx[3];
    localBounds(c, it, mn, mx);
    Xf inv = inverse(boundsXf(it));
    float lo[3], ld[3];
    for (int i = 0; i < 3; i++) {
        lo[i] = inv.m[i][0] * o[0] + inv.m[i][1] * o[1] + inv.m[i][2] * o[2] + inv.m[i][3];
        ld[i] = inv.m[i][0] * d[0] + inv.m[i][1] * d[1] + inv.m[i][2] * d[2];
    }
    float t0 = 0, t1 = 1e30f;
    for (int k = 0; k < 3; k++) {
        if (fabsf(ld[k]) < 1e-12f) { if (lo[k] < mn[k] || lo[k] > mx[k]) return -1; continue; }
        float a = (mn[k] - lo[k]) / ld[k], b = (mx[k] - lo[k]) / ld[k];
        if (a > b) std::swap(a, b);
        t0 = std::max(t0, a); t1 = std::min(t1, b);
        if (t0 > t1) return -1;
    }
    return t0;
}

struct Hit { int cx, cz, idx; float t; };
Hit pick(const float o[3], const float d[3]) {
    Hit best{0, 0, -1, 1e30f};
    float bestVol = 1e30f;
    for (auto& [k, c] : cells)
        for (size_t i = 0; i < c.items.size(); i++) {
            const Item& it = c.items[i];
            if (!pickable(it)) continue;
            float t = rayItem(c, it, o, d);
            if (t < 0 || t > 500.f) continue;
            float mn[3], mx[3];
            localBounds(c, it, mn, mx);
            float vol = (mx[0] - mn[0]) * (mx[1] - mn[1]) * (mx[2] - mn[2]);
            // nearest entry wins; when the ray starts inside several (a volume around a room), the smallest
            if (t < best.t - 1e-3f || (t < best.t + 1e-3f && vol < bestVol)) { best = {k.first, k.second, (int)i, t}; bestVol = vol; }
        }
    return best;
}
float rayTerrain(const float o[3], const float d[3]) {
    for (int k = 0; k < 3; k++) if (!std::isfinite(o[k]) || !std::isfinite(d[k])) return -1;
    float t = 0, prev = 0;
    for (int i = 0; i < 400 && t < 400.f; i++) {
        float y = o[1] + d[1] * t, h = host.height(o[0] + d[0] * t, o[2] + d[2] * t);
        if (y < h) {
            for (int j = 0; j < 12; j++) {  // bisect
                float m = 0.5f * (prev + t);
                if (o[1] + d[1] * m < host.height(o[0] + d[0] * m, o[2] + d[2] * m)) t = m; else prev = m;
            }
            return t;
        }
        prev = t;
        t += std::max(0.05f, 0.02f * t);
    }
    return -1;
}
void mouseRay(float d[3]) {
    ImGuiIO& io = ImGui::GetIO();
    float W = io.DisplaySize.x, H = io.DisplaySize.y;
    float x = (2.f * io.MousePos.x / W - 1.f) * view.tanHalfFov * W / H, y = (1.f - 2.f * io.MousePos.y / H) * view.tanHalfFov;
    float l = 0;
    for (int k = 0; k < 3; k++) { d[k] = view.fwd[k] + view.right[k] * x + view.up[k] * y; l += d[k] * d[k]; }
    l = sqrtf(l);
    for (int k = 0; k < 3; k++) d[k] /= l;
}
bool project(const float p[3], ImVec2& s) {
    ImGuiIO& io = ImGui::GetIO();
    float d[3] = {p[0] - view.pos[0], p[1] - view.pos[1], p[2] - view.pos[2]};
    float z = d[0] * view.fwd[0] + d[1] * view.fwd[1] + d[2] * view.fwd[2];
    if (z < 0.05f) return false;
    float W = io.DisplaySize.x, H = io.DisplaySize.y;
    float x = (d[0] * view.right[0] + d[1] * view.right[1] + d[2] * view.right[2]) / (z * view.tanHalfFov * W / H);
    float y = (d[0] * view.up[0] + d[1] * view.up[1] + d[2] * view.up[2]) / (z * view.tanHalfFov);
    s = ImVec2((x + 1.f) * 0.5f * W, (1.f - y) * 0.5f * H);
    return true;
}

// where new things go: what the screen centre looks at (items, then terrain), within 60 m
void placePoint(float out[3]) {
    float t = 1e30f;
    Hit h = pick(view.pos, view.fwd);
    if (h.idx >= 0) t = h.t;
    float tt = rayTerrain(view.pos, view.fwd);
    if (tt >= 0) t = std::min(t, tt);
    if (t > 60.f) t = 10.f;
    for (int k = 0; k < 3; k++) out[k] = view.pos[k] + view.fwd[k] * t;
    if (t == 10.f) out[1] = host.height(out[0], out[2]);
}
float camYawSnapped() { return roundf(atan2f(view.fwd[0], view.fwd[2]) / DEG / 90.f) * 90.f; }

void addItem(Item it, bool atCursor = true) {
    if (atCursor) {
        float p[3];
        placePoint(p);
        it.pos[0] = snapv(p[0], grid); it.pos[2] = snapv(p[2], grid);
        it.pos[1] = p[1] + (it.kind == K_LIGHT ? 2.f : it.kind == K_VOLUME ? it.scale[1] : 0.f);
        if (it.kind == K_PRIM || it.kind == K_ASSET) it.rot[1] = camYawSnapped();
    }
    ECell* c = open(cellOf(it.pos[0]), cellOf(it.pos[2]));
    if (!c) { status = "that cell is still loading"; return; }
    pushUndo(*c);
    if (groundSnap && groundable(it)) snapToGround(*c, it);
    c->items.push_back(it);
    changed(*c, true);
    sel = {c->cx, c->cz, (int)c->items.size() - 1};
}
Item prim(int shape, float sx, float sy, float sz, const std::string& mat, const char* name) {
    Item it;
    it.kind = K_PRIM;
    it.shape = shape;
    it.size[0] = sx; it.size[1] = sy; it.size[2] = sz;
    it.mat = mat;
    it.name = name;
    if (auto* e = matlib::find(mat)) it.strength = e->strength;
    return it;
}

// ---------------------------------------------------------------- asset library: import, thumbnails, browser
// Imports and thumbnails run on one background thread; finished thumbnails are uploaded in ui().
constexpr int THUMB = 128;
struct Job { bool import; std::string path; float keep = 1.f; std::string as; bool materials = false; };  // import a source file (keep = triangle fraction, as = asset name), or (re)make the thumbnail of asset <path>
std::mutex jobMu;
std::condition_variable jobCv;
std::deque<Job> jobQ;
bool workerUp = false;
std::atomic<int> jobsLeft{0};
struct Done { std::string name; std::vector<uint8_t> rgba; std::string msg; bool rescan = false; float foot = 0; };
std::vector<Done> jobDone;  // under jobMu
std::vector<std::string> pickedPaths;  // from the file picker, under jobMu
struct Thumb { uint64_t tex = 0; bool queued = false; float foot = 0; };  // foot: footprint radius (m) about the origin
std::map<std::string, Thumb> thumbs;
std::string dragAsset;  // asset dragged out of the browser, placed where it is released over the world
bool showBrowser = true;

float importKeep = 1.f;  // fraction of triangles kept on import (meshoptimizer simplify)

// Simplify every mesh to ~keep of its triangles, per material run (so triMat stays valid), then drop unused vertices.
// Fewer triangles = faster hulls / cutting when a tree breaks, smaller BLAS, quicker loads.
void simplify(ModelData& m, float keep) {
    for (MeshData& md : m.meshes) {
        size_t nv = md.verts.size() / 8;
        if (!nv || md.idx.size() < 300) continue;
        std::vector<uint32_t> idx, mats;
        for (size_t t0 = 0; t0 < md.triMat.size();) {
            size_t t1 = t0;
            while (t1 < md.triMat.size() && md.triMat[t1] == md.triMat[t0]) t1++;
            size_t n = (t1 - t0) * 3, target = std::max<size_t>(3, (size_t)(n * keep) / 3 * 3);
            std::vector<uint32_t> out(n);
            float err = 0;
            size_t got = meshopt_simplify(out.data(), &md.idx[t0 * 3], n, md.verts.data(), nv, 8 * sizeof(float), target, 0.08f, 0, &err);
            if (got > target * 1.3f)  // leaf cards and other disconnected bits barely simplify: collapse them spatially
                got = meshopt_simplifySloppy(out.data(), &md.idx[t0 * 3], n, md.verts.data(), nv, 8 * sizeof(float), target, 0.08f, &err);
            idx.insert(idx.end(), out.begin(), out.begin() + got);
            mats.insert(mats.end(), got / 3, md.triMat[t0]);
            t0 = t1;
        }
        std::vector<float> verts(md.verts.size());
        size_t used = meshopt_optimizeVertexFetch(verts.data(), idx.data(), idx.size(), md.verts.data(), nv, 8 * sizeof(float));
        verts.resize(used * 8);
        md.verts.swap(verts);
        md.idx.swap(idx);
        md.triMat.swap(mats);
    }
}
size_t triCount(const ModelData& m) { size_t n = 0; for (auto& md : m.meshes) n += md.idx.size() / 3; return n; }

std::string lower(std::string s) { for (auto& ch : s) ch = (char)tolower((unsigned char)ch); return s; }
std::string assetName(const std::filesystem::path& p) {
    std::string n = p.stem().string();
    for (auto& ch : n) if (!isalnum((unsigned char)ch) && ch != '_' && ch != '-') ch = '_';
    return n.empty() ? "asset" : n;
}

// Software-rasterized 3/4 view (lambert, base colour texture, alpha cutout), 2x supersampled, transparent background.
std::vector<uint8_t> renderThumb(const ModelData& m) {
    const int S = THUMB * 2;
    const float yaw = 35 * DEG, pitch = 22 * DEG;
    float fwd[3] = {-sinf(yaw) * cosf(pitch), -sinf(pitch), -cosf(yaw) * cosf(pitch)};
    float rt[3] = {cosf(yaw), 0, -sinf(yaw)};
    float up[3] = {rt[1] * fwd[2] - rt[2] * fwd[1], rt[2] * fwd[0] - rt[0] * fwd[2], rt[0] * fwd[1] - rt[1] * fwd[0]};
    auto dot = [](const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; };
    auto xfp = [](const Xf& x, const float* p, float* o) {
        for (int i = 0; i < 3; i++) o[i] = x.m[i][0] * p[0] + x.m[i][1] * p[1] + x.m[i][2] * p[2] + x.m[i][3];
    };
    // fit: the projected extent of every vertex
    float lo[2] = {1e30f, 1e30f}, hi[2] = {-1e30f, -1e30f};
    for (auto& in : m.instances) {
        if (in.mesh < 0 || in.mesh >= (int)m.meshes.size()) continue;
        const auto& v = m.meshes[in.mesh].verts;
        for (size_t i = 0; i + 8 <= v.size(); i += 8) {
            float w[3];
            xfp(in.xf, &v[i], w);
            float sx = dot(w, rt), sy = dot(w, up);
            lo[0] = std::min(lo[0], sx); hi[0] = std::max(hi[0], sx);
            lo[1] = std::min(lo[1], sy); hi[1] = std::max(hi[1], sy);
        }
    }
    if (lo[0] > hi[0]) return {};
    float ext = std::max(hi[0] - lo[0], hi[1] - lo[1]) * 1.08f + 1e-6f, cx = 0.5f * (lo[0] + hi[0]), cy = 0.5f * (lo[1] + hi[1]);
    std::vector<float> zb((size_t)S * S, 1e30f), col((size_t)S * S * 3, 0.f);
    std::vector<uint8_t> cov((size_t)S * S, 0);
    float L[3] = {0.45f, 0.8f, 0.35f};
    float ll = sqrtf(dot(L, L));
    for (float& c : L) c /= ll;
    for (auto& in : m.instances) {
        if (in.mesh < 0 || in.mesh >= (int)m.meshes.size()) continue;
        const MeshData& md = m.meshes[in.mesh];
        for (size_t t = 0; t * 3 + 2 < md.idx.size(); t++) {
            float w[3][3], P[3][3], uv[3][2];
            for (int k = 0; k < 3; k++) {
                const float* v = &md.verts[(size_t)md.idx[t * 3 + k] * 8];
                xfp(in.xf, v, w[k]);
                P[k][0] = ((dot(w[k], rt) - cx) / ext + 0.5f) * S;
                P[k][1] = (0.5f - (dot(w[k], up) - cy) / ext) * S;
                P[k][2] = dot(w[k], fwd);
                uv[k][0] = v[6]; uv[k][1] = v[7];
            }
            float area = (P[1][0] - P[0][0]) * (P[2][1] - P[0][1]) - (P[2][0] - P[0][0]) * (P[1][1] - P[0][1]);
            if (fabsf(area) < 1e-9f) continue;
            float e1[3], e2[3], n[3];
            for (int k = 0; k < 3; k++) { e1[k] = w[1][k] - w[0][k]; e2[k] = w[2][k] - w[0][k]; }
            n[0] = e1[1] * e2[2] - e1[2] * e2[1]; n[1] = e1[2] * e2[0] - e1[0] * e2[2]; n[2] = e1[0] * e2[1] - e1[1] * e2[0];
            float nl = sqrtf(dot(n, n));
            if (nl < 1e-20f) continue;
            for (float& c : n) c /= nl;
            if (dot(n, fwd) > 0) for (float& c : n) c = -c;  // two-sided (leaves)
            float shade = 0.35f + 0.65f * std::max(0.f, dot(n, L));
            int mi = t < md.triMat.size() ? (int)md.triMat[t] : -1;
            const MaterialData* mt = mi >= 0 && mi < (int)m.materials.size() ? &m.materials[mi] : nullptr;
            const TextureData* tx = mt && mt->tex >= 0 && mt->tex < (int)m.textures.size() && !m.textures[mt->tex].rgba.empty() ? &m.textures[mt->tex] : nullptr;
            int x0 = std::max(0, (int)floorf(std::min({P[0][0], P[1][0], P[2][0]}))), x1 = std::min(S - 1, (int)ceilf(std::max({P[0][0], P[1][0], P[2][0]})));
            int y0 = std::max(0, (int)floorf(std::min({P[0][1], P[1][1], P[2][1]}))), y1 = std::min(S - 1, (int)ceilf(std::max({P[0][1], P[1][1], P[2][1]})));
            for (int y = y0; y <= y1; y++)
                for (int x = x0; x <= x1; x++) {
                    float px = x + 0.5f, py = y + 0.5f;
                    float b0 = ((P[1][0] - px) * (P[2][1] - py) - (P[2][0] - px) * (P[1][1] - py)) / area;
                    float b1 = ((P[2][0] - px) * (P[0][1] - py) - (P[0][0] - px) * (P[2][1] - py)) / area;
                    float b2 = 1.f - b0 - b1;
                    if (b0 < 0 || b1 < 0 || b2 < 0) continue;
                    float z = b0 * P[0][2] + b1 * P[1][2] + b2 * P[2][2];
                    size_t o = (size_t)y * S + x;
                    if (z >= zb[o]) continue;
                    float c[4] = {1, 1, 1, 1};
                    if (mt) for (int k = 0; k < 4; k++) c[k] = mt->color[k];
                    if (tx) {
                        float u = b0 * uv[0][0] + b1 * uv[1][0] + b2 * uv[2][0], v = b0 * uv[0][1] + b1 * uv[1][1] + b2 * uv[2][1];
                        u -= floorf(u); v -= floorf(v);
                        const uint8_t* s = &tx->rgba[((size_t)std::min(tx->h - 1, (uint32_t)(v * tx->h)) * tx->w + std::min(tx->w - 1, (uint32_t)(u * tx->w))) * 4];
                        for (int k = 0; k < 4; k++) c[k] *= s[k] / 255.f;
                    }
                    if (c[3] < 0.5f) continue;  // cutout
                    zb[o] = z;
                    cov[o] = 1;
                    for (int k = 0; k < 3; k++) col[o * 3 + k] = std::min(1.f, c[k] * shade);
                }
        }
    }
    std::vector<uint8_t> out((size_t)THUMB * THUMB * 4);
    for (int y = 0; y < THUMB; y++)
        for (int x = 0; x < THUMB; x++) {
            float c[3] = {}, a = 0;
            for (int j = 0; j < 2; j++)
                for (int i = 0; i < 2; i++) {
                    size_t o = (size_t)(y * 2 + j) * S + x * 2 + i;
                    if (!cov[o]) continue;
                    a += 0.25f;
                    for (int k = 0; k < 3; k++) c[k] += col[o * 3 + k];
                }
            uint8_t* d = &out[((size_t)y * THUMB + x) * 4];
            for (int k = 0; k < 3; k++) d[k] = (uint8_t)(a > 0 ? std::min(255.f, c[k] / (a * 4) * 255.f + 0.5f) : 0);
            d[3] = (uint8_t)(a * 255.f + 0.5f);
        }
    return out;
}

std::string thumbPath(const std::string& name) { return host.worldDir + "assets/.thumbs/" + name + ".png"; }

// worker side: make (or load the cached) thumbnail of assets/<name>.glb
// footprint radius in xz about the model origin (scatter spacing)
float footprint(const ModelData& m) {
    float r = 0;
    for (auto& in : m.instances) {
        if (in.mesh < 0 || in.mesh >= (int)m.meshes.size()) continue;
        float a[3], b[3];
        xfAabb(in.xf, m.meshes[in.mesh].bmin, m.meshes[in.mesh].bmax, a, b);
        r = std::max({r, -a[0], b[0], -a[2], b[2]});
    }
    return r;
}

std::vector<uint8_t> makeThumb(const std::string& name, const ModelData* loaded, float& foot) {
    std::error_code ec;
    std::string glb = assetFile(host.worldDir, name), tp = thumbPath(name);
    if (!loaded && std::filesystem::exists(tp, ec) && std::filesystem::last_write_time(tp, ec) >= std::filesystem::last_write_time(glb, ec)) {
        if (FILE* f = fopen((tp + ".txt").c_str(), "r")) { if (fscanf(f, "%f", &foot) != 1) foot = 0; fclose(f); }
        int w, h, ch;
        if (stbi_uc* px = stbi_load(tp.c_str(), &w, &h, &ch, 4)) {
            std::vector<uint8_t> r;
            if (w == THUMB && h == THUMB) r.assign(px, px + (size_t)w * h * 4);
            stbi_image_free(px);
            if (!r.empty() && foot > 0) return r;
        }
    }
    ModelData m;
    std::string err;
    if (!loaded) { if (!loadGltf(glb, m, err)) return {}; loaded = &m; }
    std::vector<uint8_t> px = renderThumb(*loaded);
    foot = footprint(*loaded);
    if (!px.empty()) {
        std::filesystem::create_directories(std::filesystem::path(tp).parent_path(), ec);
        stbi_write_png(tp.c_str(), THUMB, THUMB, 4, px.data(), THUMB * 4);
        if (FILE* f = fopen((tp + ".txt").c_str(), "w")) { fprintf(f, "%g\n", foot); fclose(f); }
    }
    return px;
}

// glTF materials -> PBR sets in <world>/textures/<name>[_i]/: albedo.png, normal.png, orm.png (occlusion r,
// roughness g, metalness b), usable on primitives and flatten zones. Returns how many were written.
int extractMaterials(const ModelData& m, const std::string& name, std::string& names) {
    int n = 0;
    std::error_code ec;
    for (size_t i = 0; i < m.materials.size(); i++) {
        const MaterialData& mt = m.materials[i];
        auto tex = [&](int k) -> const TextureData* { return k >= 0 && k < (int)m.textures.size() && !m.textures[k].rgba.empty() ? &m.textures[k] : nullptr; };
        const TextureData* alb = tex(mt.tex);
        if (!alb) continue;
        std::string set = m.materials.size() == 1 ? name : fmt("%s_%zu", name.c_str(), i);
        std::string dir = host.worldDir + "textures/" + set + "/";
        std::filesystem::create_directories(dir, ec);
        stbi_write_png((dir + "albedo.png").c_str(), alb->w, alb->h, 4, alb->rgba.data(), alb->w * 4);
        if (const TextureData* nt = tex(mt.normalTex)) stbi_write_png((dir + "normal.png").c_str(), nt->w, nt->h, 4, nt->rgba.data(), nt->w * 4);
        else std::filesystem::remove(dir + "normal.png", ec);
        const TextureData *mr = tex(mt.ormTex), *oc = tex(mt.occTex);
        if (mr || oc) {
            const TextureData* b = mr ? mr : oc;
            std::vector<uint8_t> px((size_t)b->w * b->h * 4);
            for (size_t p = 0; p < (size_t)b->w * b->h; p++) {
                px[p * 4 + 0] = oc && oc->w == b->w && oc->h == b->h ? oc->rgba[p * 4] : 255;
                px[p * 4 + 1] = mr ? mr->rgba[p * 4 + 1] : (uint8_t)(mt.rough * 255.f);
                px[p * 4 + 2] = mr ? mr->rgba[p * 4 + 2] : (uint8_t)(mt.metal * 255.f);
                px[p * 4 + 3] = 255;
            }
            stbi_write_png((dir + "orm.png").c_str(), b->w, b->h, 4, px.data(), b->w * 4);
        } else std::filesystem::remove(dir + "orm.png", ec);
        names += (n ? ", set:" : "set:") + set;
        n++;
    }
    return n;
}

Done runJob(const Job& j) {
    Done d;
    if (!j.import) { d.name = j.path; d.rgba = makeThumb(j.path, nullptr, d.foot); return d; }
    std::filesystem::path src(j.path);
    std::string ext = lower(src.extension().string());
    std::error_code ec;
    if (ext == ".png" || ext == ".jpg" || ext == ".jpeg") {
        std::filesystem::create_directories(host.worldDir + "textures", ec);
        std::filesystem::copy_file(src, host.worldDir + "textures/" + src.filename().string(), std::filesystem::copy_options::overwrite_existing, ec);
        d.msg = ec ? "texture import failed: " + ec.message() : "texture " + src.filename().string() + " added";
        d.rescan = !ec;
        return d;
    }
    // .gltf (with its .bin + textures) or .glb -> one self-contained assets/<name>.glb
    d.name = j.as.empty() ? assetName(src) : j.as;
    ModelData m;
    std::string err, dst = host.worldDir + "assets/" + d.name + ".glb";
    std::filesystem::create_directories(host.worldDir + "assets", ec);
    if (!loadGltf(j.path, m, err)) { d.msg = "import " + src.filename().string() + " failed: " + err; d.name.clear(); return d; }
    if (j.materials) {  // only its materials, into the texture library
        std::string names;
        int n = extractMaterials(m, d.name, names);
        d.msg = n ? "materials added: " + names : "no textured materials in " + src.filename().string();
        d.rescan = n > 0;
        d.name.clear();
        return d;
    }
    // a texture pack (a plane or swatch carrying a material): the material is what's wanted
    if (triCount(m) <= 2048 && m.materials.size() <= 2) {
        std::string names;
        if (extractMaterials(m, d.name, names)) {
            d.msg = "texture pack: materials added (" + names + "); pick them under Material / Ground texture";
            d.rescan = true;
            d.name.clear();
            return d;
        }
    }
    size_t tris = triCount(m);
    if (j.keep < 0.999f) simplify(m, j.keep);
    bool ok = ext == ".glb" && j.keep >= 0.999f ? (std::filesystem::copy_file(src, dst, std::filesystem::copy_options::overwrite_existing, ec), !ec)
                                                : writeGlb(dst, m, err);
    if (!ok) { d.msg = "import " + src.filename().string() + " failed: " + (ec ? ec.message() : err); d.name.clear(); return d; }
    d.rgba = makeThumb(d.name, &m, d.foot);
    d.msg = j.keep < 0.999f ? fmt("imported %s: %zu -> %zu triangles", d.name.c_str(), tris, triCount(m)) : fmt("imported %s (%zu triangles)", d.name.c_str(), tris);
    return d;
}

void queueJob(Job j) {
    std::lock_guard lk(jobMu);
    jobQ.push_back(std::move(j));
    jobsLeft++;
    if (!workerUp) {
        workerUp = true;
        std::thread([] {
            for (;;) {
                Job j;
                {
                    std::unique_lock lk(jobMu);
                    jobCv.wait(lk, [] { return !jobQ.empty(); });
                    j = jobQ.front();
                    jobQ.pop_front();
                }
                Done d = runJob(j);
                std::lock_guard lk(jobMu);
                jobDone.push_back(std::move(d));
                jobsLeft--;
            }
        }).detach();
    }
    jobCv.notify_one();
}

void scanAssets() {
    assetNames.clear();
    assetDirs.clear();
    std::error_code ec;
    // every .glb / .gltf under assets/, nested folders too (a .gltf is used in place with its .bin and textures).
    // Names are paths relative to assets/; top-level .glb files keep their bare name (what older cells store).
    const std::filesystem::path root = host.worldDir + "assets/";
    for (auto it = std::filesystem::recursive_directory_iterator(root, std::filesystem::directory_options::follow_directory_symlink, ec);
         it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
        const auto& p = it->path();
        if (it->is_directory(ec)) {
            if (p.filename().string().rfind('.', 0) == 0) { it.disable_recursion_pending(); continue; }  // .thumbs
            assetDirs.push_back(std::filesystem::relative(p, root, ec).generic_string());
            continue;
        }
        std::string e = lower(p.extension().string());
        if (e != ".glb" && e != ".gltf") continue;
        std::string rel = std::filesystem::relative(p, root, ec).generic_string();
        if (e == ".glb" && p.parent_path() == root.parent_path()) rel = p.stem().string();
        assetNames.push_back(rel);
    }
    std::sort(assetNames.begin(), assetNames.end());
    std::sort(assetDirs.begin(), assetDirs.end());
    if (!host.thumbnail) return;
    for (auto& n : assetNames)
        if (Thumb& t = thumbs[n]; !t.tex && !t.queued) { t.queued = true; queueJob({false, n}); }
}

// uploads finished thumbnails, picks up new assets
void importPaths(const std::vector<std::string>& paths);
void pollJobs() {
    std::vector<Done> done;
    std::vector<std::string> picked;
    { std::lock_guard lk(jobMu); done.swap(jobDone); picked.swap(pickedPaths); }
    if (!picked.empty()) importPaths(picked);
    bool rescan = false;
    for (auto& d : done) {
        if (!d.msg.empty()) status = d.msg;
        if (d.rescan) matlib::rescan();
        if (d.name.empty()) continue;
        rescan |= !std::count(assetNames.begin(), assetNames.end(), d.name);
        Thumb& t = thumbs[d.name];
        t.queued = true;  // made (or failed): don't retry every scan
        t.foot = d.foot;
        if (!d.rgba.empty() && host.thumbnail) t.tex = host.thumbnail(d.rgba.data(), THUMB, THUMB);
    }
    if (rescan) scanAssets();
}

void importPaths(const std::vector<std::string>& paths) {
    int n = 0;
    std::error_code ec;
    auto take = [&](const std::filesystem::path& p) {
        std::string e = lower(p.extension().string());
        if (e == ".gltf" || e == ".glb" || e == ".png" || e == ".jpg" || e == ".jpeg") { queueJob({true, p.string(), importKeep}); n++; }
    };
    for (auto& s : paths) {
        std::filesystem::path p(s);
        if (std::filesystem::is_directory(p, ec)) {
            // a folder: its models (a model's own texture folder is not imported as world textures)
            for (auto it = std::filesystem::recursive_directory_iterator(p, ec); it != std::filesystem::recursive_directory_iterator(); it.increment(ec)) {
                std::string e = lower(it->path().extension().string());
                if (e == ".gltf" || e == ".glb") take(it->path());
            }
        } else take(p);
    }
    status = n ? fmt("importing %d file%s...", n, n == 1 ? "" : "s") : "nothing to import (drop .gltf / .glb files, folders or PNG / JPG)";
}

// the point under the mouse (items, then terrain), false when it looks at the sky
bool mousePoint(float out[3]) {
    float d[3];
    mouseRay(d);
    float t = 1e30f;
    Hit h = pick(view.pos, d);
    if (h.idx >= 0) t = h.t;
    float tt = rayTerrain(view.pos, d);
    if (tt >= 0) t = std::min(t, tt);
    if (t > 400.f) return false;
    for (int k = 0; k < 3; k++) out[k] = view.pos[k] + d[k] * t;
    return true;
}

void addAsset(const std::string& name, const float* at) {
    Item it;
    it.kind = K_ASSET;
    it.asset = it.name = name;
    if (!at) { addItem(it); return; }
    it.pos[0] = snapv(at[0], grid); it.pos[1] = at[1]; it.pos[2] = snapv(at[2], grid);
    it.rot[1] = camYawSnapped();
    addItem(it, false);
}

// Native file picker (KDE kdialog, else zenity) on its own thread; the chosen paths are imported.
std::atomic<bool> pickerOpen{false};
void openPicker(bool folders) {
    if (pickerOpen.exchange(true)) return;
    std::thread([folders] {
        std::string cmd = folders
            ? "kdialog --title 'Import asset folder' --getexistingdirectory ~ 2>/dev/null || zenity --file-selection --directory --title='Import asset folder' 2>/dev/null"
            : "kdialog --title 'Import assets' --multiple --separate-output --getopenfilename ~ 'Models and textures (*.gltf *.glb *.png *.jpg *.jpeg)' 2>/dev/null"
              " || zenity --file-selection --multiple --separator='\n' --title='Import assets' --file-filter='Models and textures | *.gltf *.glb *.png *.jpg *.jpeg' 2>/dev/null";
        std::vector<std::string> paths;
        if (FILE* f = popen(cmd.c_str(), "r")) {
            char line[4096];
            while (fgets(line, sizeof line, f)) {
                std::string l(line);
                while (!l.empty() && (l.back() == '\n' || l.back() == '\r')) l.pop_back();
                if (!l.empty()) paths.push_back(l);
            }
            pclose(f);
        }
        if (!paths.empty()) { std::lock_guard lk(jobMu); pickedPaths.insert(pickedPaths.end(), paths.begin(), paths.end()); }
        pickerOpen = false;
    }).detach();
}
void openFolder(const std::string& dir) {
    std::string cmd = "xdg-open '" + dir + "' >/dev/null 2>&1 &";
    if (system(cmd.c_str()) != 0) status = "couldn't open " + dir;
}

// ---- scatter: a weighted mix of assets dropped over a disc, dart throwing with footprint spacing
struct MixEntry { std::string name; float weight = 1.f; };
std::vector<MixEntry> scatterMix;
float scatterRadius = 20.f, scatterDensity = 2.f, scatterSpacing = 0.8f, scatterScale[2] = {0.8f, 1.2f};
bool showScatter = false;

float footOf(const std::string& name) {
    auto it = thumbs.find(name);
    return it != thumbs.end() && it->second.foot > 0 ? it->second.foot : 2.f;  // unknown yet: assume a small tree
}

// Fill the disc around `at` up to `density` instances per 100 m2. Each candidate's footprint (bounding radius x
// scale x spacing) must clear the ones already there (assets in nearby cells included), so nothing interpenetrates.
void scatter(const float at[3]) {
    float wsum = 0;
    for (auto& e : scatterMix) wsum += std::max(e.weight, 0.f);
    if (wsum <= 0) { status = "add assets to the scatter mix first (right-click a thumbnail)"; return; }
    struct Disc { float x, z, r; };
    std::vector<Disc> taken;
    float R = scatterRadius;
    for (auto& [k, c] : cells)
        for (auto& it : c.items)
            if (it.kind == K_ASSET || it.kind == K_TREE) {
                float dx = it.pos[0] - at[0], dz = it.pos[2] - at[2];
                if (dx * dx + dz * dz < (R + 30) * (R + 30))
                    taken.push_back({it.pos[0], it.pos[2], (it.kind == K_ASSET ? footOf(it.asset) : 1.f) * it.scale[0] * scatterSpacing});
            }
    int want = (int)(scatterDensity * 3.14159f * R * R / 100.f), placed = 0, skippedCells = 0;
    std::map<std::pair<int, int>, bool> undone;
    for (int tries = 0; placed < want && tries < want * 30; tries++) {
        float a = (float)rand() / RAND_MAX * 6.2831853f, d = R * sqrtf((float)rand() / RAND_MAX);
        float x = at[0] + cosf(a) * d, z = at[2] + sinf(a) * d;
        float pick = (float)rand() / RAND_MAX * wsum;
        const MixEntry* e = &scatterMix.back();
        for (auto& m : scatterMix) { if ((pick -= std::max(m.weight, 0.f)) <= 0) { e = &m; break; } }
        float sc = scatterScale[0] + (scatterScale[1] - scatterScale[0]) * (float)rand() / RAND_MAX;
        float r = footOf(e->name) * sc * scatterSpacing;
        bool clear = true;
        for (auto& t : taken) if ((t.x - x) * (t.x - x) + (t.z - z) * (t.z - z) < (t.r + r) * (t.r + r)) { clear = false; break; }
        if (!clear) continue;
        ECell* c = open(cellOf(x), cellOf(z));
        if (!c) { skippedCells++; continue; }
        if (!undone[{c->cx, c->cz}]) { pushUndo(*c); undone[{c->cx, c->cz}] = true; }  // one undo step per cell
        Item it;
        it.kind = K_ASSET;
        it.asset = it.name = e->name;
        it.pos[0] = x; it.pos[2] = z;
        it.rot[1] = (float)rand() / RAND_MAX * 360.f;
        it.scale[0] = it.scale[1] = it.scale[2] = sc;
        snapToGround(*c, it);
        c->items.push_back(it);
        changed(*c, true);
        taken.push_back({x, z, r});
        placed++;
    }
    status = fmt("scattered %d of %d (%s)", placed, want, placed < want ? "the rest had no room at this spacing" : "full");
    if (skippedCells) status += fmt(", %d in cells still loading", skippedCells);
}
// remove scattered mix assets inside the disc
void unscatter(const float at[3]) {
    int n = 0;
    for (auto& [k, c] : cells) {
        bool any = false;
        for (size_t i = c.items.size(); i-- > 0;) {
            const Item& it = c.items[i];
            float dx = it.pos[0] - at[0], dz = it.pos[2] - at[2];
            if (it.kind != K_ASSET || dx * dx + dz * dz > scatterRadius * scatterRadius) continue;
            if (!std::any_of(scatterMix.begin(), scatterMix.end(), [&](auto& m) { return m.name == it.asset; })) continue;
            if (!any) { pushUndo(c); any = true; }
            c.items.erase(c.items.begin() + i);
            n++;
        }
        if (any) { changed(c, true); if (sel.cx == c.cx && sel.cz == c.cz) sel.idx = -1; }
    }
    status = fmt("removed %d", n);
}

void scatterPanel() {
    ImGui::SetNextWindowSize(ImVec2(340, 360), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Scatter (forests, rocks)", &showScatter)) { ImGui::End(); return; }
    ImGui::TextWrapped("Mix: right-click thumbnails > Add to scatter mix, or drag them here.");
    ImGui::BeginChild("mix", ImVec2(0, 130), ImGuiChildFlags_Borders);
    for (size_t i = 0; i < scatterMix.size(); i++) {
        ImGui::PushID((int)i);
        uint64_t tex = thumbs.count(scatterMix[i].name) ? thumbs[scatterMix[i].name].tex : 0;
        if (tex) { ImGui::Image((ImTextureID)tex, ImVec2(28, 28)); ImGui::SameLine(); }
        ImGui::SetNextItemWidth(90);
        ImGui::SliderFloat("##w", &scatterMix[i].weight, 0, 10, "weight %.1f");
        ImGui::SameLine();
        ImGui::TextUnformatted(scatterMix[i].name.c_str());
        ImGui::SameLine();
        bool del = ImGui::SmallButton("x");
        ImGui::PopID();
        if (del) { scatterMix.erase(scatterMix.begin() + i); break; }
    }
    ImGui::EndChild();
    if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload* p = ImGui::AcceptDragDropPayload("PTX_ASSET")) {
            std::string n = (const char*)p->Data;
            if (!std::any_of(scatterMix.begin(), scatterMix.end(), [&](auto& m) { return m.name == n; })) scatterMix.push_back({n});
            dragAsset.clear();
        }
        ImGui::EndDragDropTarget();
    }
    ImGui::DragFloat("Radius (m)", &scatterRadius, 0.2f, 1.f, 300.f, "%.0f");
    ImGui::DragFloat("Density (per 100 m2)", &scatterDensity, 0.02f, 0.01f, 50.f, "%.2f");
    ImGui::DragFloat("Spacing", &scatterSpacing, 0.01f, 0.1f, 2.f, "%.2f x bounds");
    ImGui::SetItemTooltip("1 = bounding footprints never overlap; lower lets canopies interleave, higher thins it out");
    ImGui::DragFloat2("Scale range", scatterScale, 0.01f, 0.1f, 5.f, "%.2f");
    ImGui::TextDisabled("Sits on the ground, sunk by 'Sink into ground' (Assets window). Random yaw.");
    float p[3];
    placePoint(p);
    if (ImGui::Button("Scatter at crosshair")) scatter(p);
    ImGui::SameLine();
    if (ImGui::Button("Clear mix assets in radius")) unscatter(p);
    ImGui::TextDisabled("Ctrl+Z undoes per cell");
    ImGui::SeparatorText("World forest");
    static float builtinW = 0.f;
    ImGui::SetNextItemWidth(120);
    ImGui::DragFloat("Built-in trees weight", &builtinW, 0.05f, 0.f, 20.f, "%.2f");
    ImGui::SetItemTooltip("0 = the mix replaces the procedural trees everywhere; > 0 keeps that share of them");
    if (ImGui::Button("Use this mix for the world's trees")) {
        std::string path = host.worldDir + "forest.txt";
        if (FILE* f = fopen(path.c_str(), "w")) {
            fprintf(f, "# world forest: <weight> <asset>, read at startup (ptx editor)\nbuiltin %g\nspacing %g\n", builtinW, scatterSpacing);
            for (auto& e : scatterMix) if (e.weight > 0) fprintf(f, "%g %s\n", e.weight, e.name.c_str());
            fclose(f);
            status = "wrote forest.txt: restart ptx to grow the new forest (variants + impostors are built at startup)";
        } else status = "couldn't write " + path;
    }
    ImGui::SameLine();
    if (ImGui::Button("Restore built-in")) {
        std::error_code ec;
        std::filesystem::remove(host.worldDir + "forest.txt", ec);
        status = "forest.txt removed: built-in trees after a restart";
    }
    ImGui::TextDisabled("Uses Spacing above; the world density slider still applies. Up to 12 assets.");
    // preview ring on the ground
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    ImVec2 prev;
    bool have = false;
    for (int i = 0; i <= 48; i++) {
        float a = i / 48.f * 6.2831853f, q[3] = {p[0] + cosf(a) * scatterRadius, 0, p[2] + sinf(a) * scatterRadius};
        q[1] = host.height(q[0], q[2]) + 0.2f;
        ImVec2 sc;
        bool ok = project(q, sc);
        if (ok && have) dl->AddLine(prev, sc, IM_COL32(120, 255, 140, 200), 2.f);
        prev = sc; have = ok;
    }
    ImGui::End();
}

// ---- library management: folders, rename / move (leaves an alias so placed copies keep working), delete
enum { POP_NONE, POP_NEWFOLDER, POP_RENAME, POP_RENAMEDIR, POP_DELETE };
int popup = POP_NONE, popupOpen = POP_NONE;
char popupName[128] = "";
std::string popupTarget, curDir;

std::string dirOf(const std::string& n) { size_t p = n.rfind('/'); return p == std::string::npos ? "" : n.substr(0, p); }
std::string baseOf(const std::string& n) {
    std::string b = n.substr(n.rfind('/') == std::string::npos ? 0 : n.rfind('/') + 1);
    std::string l = lower(b);
    for (const char* e : {".glb", ".gltf"}) if (l.size() > strlen(e) && l.compare(l.size() - strlen(e), strlen(e), e) == 0) return b.substr(0, b.size() - strlen(e));
    return b;
}
bool isLoose(const std::string& n) { std::string l = lower(n); return l.size() > 5 && l.compare(l.size() - 5, 5, ".gltf") == 0; }
// the name scanAssets gives a .glb at assets/<dir>/<base>.glb
std::string glbName(const std::string& dir, const std::string& base) { return dir.empty() ? base : dir + "/" + base + ".glb"; }

void renameThumb(const std::string& from, const std::string& to) {
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(thumbPath(to)).parent_path(), ec);
    std::filesystem::rename(thumbPath(from), thumbPath(to), ec);
    std::filesystem::rename(thumbPath(from) + ".txt", thumbPath(to) + ".txt", ec);
    if (auto it = thumbs.find(from); it != thumbs.end()) { thumbs[to] = it->second; thumbs.erase(it); }
    for (auto& m : scatterMix) if (m.name == from) m.name = to;
}

bool moveAsset(const std::string& name, const std::string& dir, const std::string& base) {
    if (isLoose(name)) { status = "pack " + name + " into a .glb first (right-click > Pack): its parts would be left behind"; return false; }
    std::string to = glbName(dir, base);
    if (to == name || base.empty()) return false;
    std::error_code ec;
    std::string src = assetFile(host.worldDir, name), dst = host.worldDir + "assets/" + (dir.empty() ? "" : dir + "/") + base + ".glb";
    if (std::filesystem::exists(dst, ec)) { status = to + " already exists"; return false; }
    std::filesystem::create_directories(std::filesystem::path(dst).parent_path(), ec);
    std::filesystem::rename(src, dst, ec);
    if (ec) { status = "move failed: " + ec.message(); return false; }
    addAssetAlias(host.worldDir, name, to);
    renameThumb(name, to);
    status = name + " -> " + to;
    scanAssets();
    return true;
}

void deleteAsset(const std::string& name) {
    std::error_code ec;
    std::filesystem::remove(assetFile(host.worldDir, name), ec);
    std::filesystem::remove(thumbPath(name), ec);
    std::filesystem::remove(thumbPath(name) + ".txt", ec);
    thumbs.erase(name);
    scatterMix.erase(std::remove_if(scatterMix.begin(), scatterMix.end(), [&](auto& m) { return m.name == name; }), scatterMix.end());
    status = ec ? "delete failed: " + ec.message() : "deleted " + name;
    scanAssets();
}

void renameFolder(const std::string& dir, const std::string& newBase) {
    std::string to = dirOf(dir).empty() ? newBase : dirOf(dir) + "/" + newBase;
    if (to == dir || newBase.empty()) return;
    std::error_code ec;
    if (std::filesystem::exists(host.worldDir + "assets/" + to, ec)) { status = to + " already exists"; return; }
    std::vector<std::string> inside;
    for (auto& n : assetNames) if (n.rfind(dir + "/", 0) == 0) inside.push_back(n);
    std::filesystem::rename(host.worldDir + "assets/" + dir, host.worldDir + "assets/" + to, ec);
    if (ec) { status = "rename failed: " + ec.message(); return; }
    std::filesystem::rename(host.worldDir + "assets/.thumbs/" + dir, host.worldDir + "assets/.thumbs/" + to, ec);
    for (auto& n : inside) {
        std::string nn = to + n.substr(dir.size());
        addAssetAlias(host.worldDir, n, nn);
        if (auto it = thumbs.find(n); it != thumbs.end()) { thumbs[nn] = it->second; thumbs.erase(it); }
        for (auto& m : scatterMix) if (m.name == n) m.name = nn;
    }
    if (curDir == dir || curDir.rfind(dir + "/", 0) == 0) curDir = to + curDir.substr(dir.size());
    status = "folder " + dir + " -> " + to;
    scanAssets();
}

void deleteFolder(const std::string& dir) {
    std::error_code ec;
    if (!std::filesystem::is_empty(host.worldDir + "assets/" + dir, ec)) { status = dir + " isn't empty: move or delete its assets first"; return; }
    std::filesystem::remove(host.worldDir + "assets/" + dir, ec);
    std::filesystem::remove_all(host.worldDir + "assets/.thumbs/" + dir, ec);
    if (curDir == dir) curDir = dirOf(dir);
    status = ec ? "delete failed: " + ec.message() : "folder " + dir + " deleted";
    scanAssets();
}

void assetBrowser() {
    ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(20, io.DisplaySize.y - 320), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(std::min(900.f, io.DisplaySize.x - 440), 300), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Assets", &showBrowser)) { ImGui::End(); return; }
    static char filter[64] = "";
    static float tile = 96;
    ImGui::SetNextItemWidth(160);
    ImGui::InputTextWithHint("##filter", "search", filter, sizeof filter);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(110);
    ImGui::SliderFloat("##size", &tile, 48, 192, "size %.0f");
    ImGui::SameLine();
    if (ImGui::Button("Rescan")) { scanAssets(); matlib::rescan(); }
    ImGui::SetItemTooltip("Re-read world/assets/ (nested folders too): .glb and .gltf files placed there show up without importing");
    ImGui::SameLine();
    ImGui::BeginDisabled(pickerOpen);
    if (ImGui::Button("Import files...")) openPicker(false);
    ImGui::SameLine();
    if (ImGui::Button("Import folder...")) openPicker(true);
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Open assets folder")) {
        std::error_code ec;
        std::filesystem::create_directories(host.worldDir + "assets", ec);
        openFolder(std::filesystem::absolute(host.worldDir + "assets").string());
    }
    ImGui::Checkbox("Scatter tool", &showScatter);
    ImGui::SameLine();
    {
        float pct = importKeep * 100.f;
        ImGui::SetNextItemWidth(150);
        if (ImGui::SliderFloat("##detail", &pct, 1, 100, pct >= 99.95f ? "import: full detail" : "import: %.0f%% triangles")) importKeep = pct / 100.f;
        ImGui::SetItemTooltip("Simplify meshes on import (meshoptimizer). Trees with millions of triangles break / cut down\n"
                              "much faster and load quicker at 10-25%%. Existing assets: right-click > Make simplified copy.");
        ImGui::SameLine();
    }
    ImGui::Checkbox("Snap to ground", &groundSnap);
    ImGui::SetItemTooltip("Placed and dragged assets, trees and rocks sit on the terrain (their lowest point)");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(110);
    ImGui::DragFloat("Sink into ground", &sinkDepth, 0.01f, -2.f, 5.f, "%.2f m");
    ImGui::SetItemTooltip("How far the base goes below the surface (roots, foundations, bedded rocks). Negative floats it.");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(std::max(80.f, ImGui::GetContentRegionAvail().x - 70));
    bool go = ImGui::InputTextWithHint("##imp", "/path/to/model.gltf, .glb or folder (or drop onto the window)", importPath, sizeof importPath,
                                       ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine();
    if ((ImGui::Button("Import") || go) && importPath[0]) importPaths({importPath});
    if (int n = jobsLeft.load()) { ImGui::SameLine(); ImGui::TextDisabled("working (%d)...", n); }

    // breadcrumb: assets / a / b
    if (ImGui::SmallButton("assets")) curDir.clear();
    {
        std::string acc;
        size_t p0 = 0;
        while (p0 < curDir.size()) {
            size_t p1 = curDir.find('/', p0);
            if (p1 == std::string::npos) p1 = curDir.size();
            acc = curDir.substr(0, p1);
            ImGui::SameLine(0, 2); ImGui::TextDisabled("/"); ImGui::SameLine(0, 2);
            if (ImGui::SmallButton((curDir.substr(p0, p1 - p0) + "##bc" + acc).c_str())) { curDir = acc; break; }
            p0 = p1 + 1;
        }
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("+ New folder")) { popup = POP_NEWFOLDER; popupName[0] = 0; }

    ImGui::BeginChild("grid", ImVec2(0, 0), ImGuiChildFlags_Borders);
    const float pad = ImGui::GetStyle().FramePadding.x, sp = ImGui::GetStyle().ItemSpacing.x, cell = tile + 2 * pad;
    int cols = std::max(1, (int)((ImGui::GetContentRegionAvail().x + sp) / (cell + sp))), col = 0;
    std::string f = lower(filter);
    auto label = [&](const std::string& s) {  // clipped to the tile
        std::string l = s;
        while (l.size() > 3 && ImGui::CalcTextSize(l.c_str()).x > cell) l = l.substr(0, l.size() - 4) + "..";
        ImGui::TextUnformatted(l.c_str());
    };
    auto next = [&] { if (++col < cols) ImGui::SameLine(); else col = 0; };
    // a folder tile: double-click opens, assets dropped on it move there
    auto folderTile = [&](const std::string& dir, const std::string& shown, bool up) {
        ImGui::PushID(("dir:" + dir + (up ? ".." : "")).c_str());
        ImGui::BeginGroup();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.32f, 0.27f, 0.12f, 1));
        ImGui::Button(up ? "<-" : "[ ]", ImVec2(cell, cell));
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) curDir = dir;
        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload* p = ImGui::AcceptDragDropPayload("PTX_ASSET")) {
                std::string n = (const char*)p->Data;
                moveAsset(n, dir, baseOf(n));
                dragAsset.clear();
            }
            ImGui::EndDragDropTarget();
        }
        ImGui::SetItemTooltip(up ? "up (drop assets here to move them up)" : "%s\ndouble-click to open, drop assets on it to move them in", dir.c_str());
        if (!up && ImGui::BeginPopupContextItem("fctx")) {
            if (ImGui::MenuItem("Open")) curDir = dir;
            if (ImGui::MenuItem("Rename...")) { popup = POP_RENAMEDIR; popupTarget = dir; snprintf(popupName, sizeof popupName, "%s", shown.c_str()); }
            if (ImGui::MenuItem("Delete (when empty)")) deleteFolder(dir);
            if (ImGui::MenuItem("Show in folder")) openFolder(std::filesystem::absolute(host.worldDir + "assets/" + dir).string());
            ImGui::EndPopup();
        }
        label(shown);
        ImGui::EndGroup();
        ImGui::PopID();
        next();
    };
    if (f.empty()) {
        if (!curDir.empty()) folderTile(dirOf(curDir), "..", true);
        for (auto& d : assetDirs) if (dirOf(d) == curDir) folderTile(d, baseOf(d), false);
    }
    int shownCount = 0;
    for (auto& name : assetNames) {
        if (!f.empty() ? lower(name).find(f) == std::string::npos : dirOf(name) != curDir) continue;
        shownCount++;
        ImGui::PushID(name.c_str());
        ImGui::BeginGroup();
        uint64_t tex = thumbs.count(name) ? thumbs[name].tex : 0;
        if (tex) ImGui::ImageButton("##t", (ImTextureID)tex, ImVec2(tile, tile), ImVec2(0, 0), ImVec2(1, 1), ImVec4(0.12f, 0.13f, 0.15f, 1));
        else ImGui::Button("...", ImVec2(cell, cell));
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) addAsset(name, nullptr);
        if (ImGui::BeginDragDropSource()) {
            ImGui::SetDragDropPayload("PTX_ASSET", name.c_str(), name.size() + 1);
            if (tex) ImGui::Image((ImTextureID)tex, ImVec2(64, 64));
            ImGui::TextUnformatted(name.c_str());
            ImGui::EndDragDropSource();
        }
        ImGui::SetItemTooltip("%s\ndrag into the world or onto a folder; double-click adds it at the crosshair (right-click: more)", name.c_str());
        bool loose = isLoose(name);
        if (ImGui::BeginPopupContextItem("ctx")) {
            if (ImGui::MenuItem("Add at crosshair")) addAsset(name, nullptr);
            if (ImGui::MenuItem("Add to scatter mix")) {
                if (!std::any_of(scatterMix.begin(), scatterMix.end(), [&](auto& m) { return m.name == name; })) scatterMix.push_back({name});
                showScatter = true;
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Rename...", nullptr, false, !loose)) { popup = POP_RENAME; popupTarget = name; snprintf(popupName, sizeof popupName, "%s", baseOf(name).c_str()); }
            if (ImGui::BeginMenu("Move to", !loose)) {
                if (ImGui::MenuItem("assets (top)", nullptr, false, dirOf(name) != "")) moveAsset(name, "", baseOf(name));
                for (auto& d : assetDirs)
                    if (ImGui::MenuItem(d.c_str(), nullptr, false, d != dirOf(name))) moveAsset(name, d, baseOf(name));
                ImGui::EndMenu();
            }
            if (loose) ImGui::SetItemTooltip("Pack it into one .glb first: a loose .gltf's .bin and textures would be left behind");
            if (ImGui::MenuItem("Delete...")) { popup = POP_DELETE; popupTarget = name; }
            ImGui::Separator();
            if (ImGui::MenuItem("Pack into one .glb", nullptr, false, loose)) queueJob({true, assetFile(host.worldDir, name)});
            ImGui::SetItemTooltip("A .gltf keeps its mesh (.bin) and textures as separate files: packing bakes them into one\n"
                                  "self-contained .glb so they can't get separated (the .gltf is left as it was)");
            if (ImGui::MenuItem("Extract materials to texture library")) queueJob({true, assetFile(host.worldDir, name), 1.f, assetName(name), true});
            static float keepPct = 25.f;
            ImGui::SetNextItemWidth(120);
            ImGui::SliderFloat("##keep", &keepPct, 1, 90, "%.0f%% triangles");
            ImGui::SameLine();
            if (ImGui::MenuItem("Make simplified copy")) queueJob({true, assetFile(host.worldDir, name), keepPct / 100.f, assetName(name) + fmt("_%d", (int)keepPct)});
            if (ImGui::MenuItem("Show in folder")) openFolder(std::filesystem::absolute(std::filesystem::path(assetFile(host.worldDir, name)).parent_path()).string());
            ImGui::EndPopup();
        }
        label(baseOf(name));
        ImGui::EndGroup();
        ImGui::PopID();
        next();
    }
    if (assetNames.empty())
        ImGui::TextWrapped("No assets yet. Drag .gltf / .glb files or whole folders from your file manager onto this window to import them.");
    else if (!shownCount && f.empty() && curDir.size())
        ImGui::TextDisabled("(empty folder: drag assets onto its tile, or drop files here to import)");
    ImGui::EndChild();

    // ---- dialogs
    const char* titles[] = {"", "New folder", "Rename asset", "Rename folder", "Delete asset"};
    if (popup) { ImGui::OpenPopup(titles[popup]); popupOpen = popup; popup = POP_NONE; }
    for (int k = 1; k <= 4; k++) {
        if (!ImGui::BeginPopupModal(titles[k], nullptr, ImGuiWindowFlags_AlwaysAutoResize)) continue;
        bool ok = false, close = false;
        if (k == POP_DELETE) {
            int refs = 0;
            for (auto& [kk, c] : cells) for (auto& it : c.items) refs += it.kind == K_ASSET && resolveAsset(host.worldDir, it.asset) == resolveAsset(host.worldDir, popupTarget);
            ImGui::Text("Delete %s from disk?", popupTarget.c_str());
            if (refs) ImGui::TextColored(ImVec4(1, 0.6f, 0.3f, 1), "%d placed cop%s in open cells will vanish (and any in other cells).", refs, refs == 1 ? "y" : "ies");
            if (isLoose(popupTarget)) ImGui::TextDisabled("Only the .gltf is removed; its .bin / textures stay in the folder.");
            ok = ImGui::Button("Delete");
        } else {
            if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
            ok = ImGui::InputText("Name", popupName, sizeof popupName, ImGuiInputTextFlags_EnterReturnsTrue);
            ok |= ImGui::Button("OK");
        }
        ImGui::SameLine();
        close = ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape);
        if (ok) {
            std::string nm = assetName(std::filesystem::path(std::string(popupName) + ".x"));  // sanitised, no extension
            if (k == POP_NEWFOLDER) {
                std::error_code ec;
                std::string d = curDir.empty() ? nm : curDir + "/" + nm;
                std::filesystem::create_directories(host.worldDir + "assets/" + d, ec);
                status = ec ? "couldn't create folder: " + ec.message() : "folder " + d + " created";
                scanAssets();
            } else if (k == POP_RENAME) moveAsset(popupTarget, dirOf(popupTarget), nm);
            else if (k == POP_RENAMEDIR) renameFolder(popupTarget, nm);
            else if (k == POP_DELETE) deleteAsset(popupTarget);
            close = true;
        }
        if (close) { ImGui::CloseCurrentPopup(); popupOpen = POP_NONE; }
        ImGui::EndPopup();
    }
    ImGui::End();
}

// an asset dragged from the browser: ghost marker over the world, placed on release
void assetDrop() {
    if (const ImGuiPayload* p = ImGui::GetDragDropPayload(); p && p->IsDataType("PTX_ASSET")) dragAsset = (const char*)p->Data;
    if (dragAsset.empty()) return;
    bool overUi = ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow);
    float at[3];
    bool hit = !overUi && mousePoint(at);
    if (hit) {
        ImVec2 s;
        if (project(at, s)) {
            ImDrawList* dl = ImGui::GetForegroundDrawList();
            dl->AddCircle(s, 10.f, IM_COL32(255, 210, 40, 230), 0, 2.f);
            dl->AddText(ImVec2(s.x + 14, s.y - 7), IM_COL32(255, 255, 255, 230), dragAsset.c_str());
        }
    }
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        if (hit) addAsset(dragAsset, at);
        dragAsset.clear();
    }
}

void deleteSelected() {
    ECell* c = selCell();
    if (!c || !selected()) return;
    pushUndo(*c);
    c->items.erase(c->items.begin() + sel.idx);
    sel.idx = -1;
    changed(*c, true);
}
void duplicateSelected() {
    ECell* c = selCell();
    Item* it = selected();
    if (!c || !it) return;
    Item cp = *it;
    float yaw = cp.rot[1] * DEG, step = std::max(grid, 0.25f);
    float w = cp.kind == K_PRIM ? cp.size[0] : 1.f;  // next to it along its own x
    cp.pos[0] += cosf(yaw) * std::max(w, step);
    cp.pos[2] -= sinf(yaw) * std::max(w, step);
    cp.slot = -1;
    pushUndo(*c);
    c->items.push_back(cp);
    sel.idx = (int)c->items.size() - 1;
    changed(*c, true);
    rehome(*c, sel.idx);
}

void drawBox(ImDrawList* dl, const ECell& c, const Item& it, ImU32 col, float thick) {
    float mn[3], mx[3];
    localBounds(c, it, mn, mx);
    Xf x = boundsXf(it);
    ImVec2 s[8];
    bool ok[8];
    for (int i = 0; i < 8; i++) {
        float l[3] = {i & 1 ? mx[0] : mn[0], i & 2 ? mx[1] : mn[1], i & 4 ? mx[2] : mn[2]}, w[3];
        xfPoint(x, l, w);
        ok[i] = project(w, s[i]);
    }
    const int e[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3}, {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
    for (auto& k : e)
        if (ok[k[0]] && ok[k[1]]) dl->AddLine(s[k[0]], s[k[1]], col, thick);
}

bool matCombo(const char* label, std::string& mat, bool allowNone = false) {
    bool ch = false;
    const matlib::Entry* cur = matlib::find(mat);
    if (cur) {
        ImGui::ColorButton("##sw", ImVec4(cur->swatch[0], cur->swatch[1], cur->swatch[2], 1), ImGuiColorEditFlags_NoTooltip, ImVec2(18, 18));
        ImGui::SameLine();
    }
    if (ImGui::BeginCombo(label, mat.empty() ? "(none)" : mat.c_str())) {
        if (allowNone && ImGui::Selectable("(none)", mat.empty())) { mat.clear(); ch = true; }
        for (auto& e : matlib::list()) {
            ImGui::ColorButton(("##" + e.name).c_str(), ImVec4(e.swatch[0], e.swatch[1], e.swatch[2], 1), ImGuiColorEditFlags_NoTooltip, ImVec2(16, 16));
            ImGui::SameLine();
            if (ImGui::Selectable(e.name.c_str(), e.name == mat)) { mat = e.name; ch = true; }
        }
        ImGui::EndCombo();
    }
    return ch;
}

void inspector() {
    ECell* c = selCell();
    Item* it = selected();
    if (!c || !it) { ImGui::TextDisabled("nothing selected (click something)"); return; }
    ImGui::Text("%s in cell %d, %d", KIND_NAME[it->kind], c->cx, c->cz);
    char nm[128];
    snprintf(nm, sizeof nm, "%s", it->name.c_str());
    if (ImGui::InputText("Name", nm, sizeof nm)) { it->name = nm; changed(*c, true); }
    bool begin = false, move = false, geom = false;
    auto track = [&](bool edited, bool isGeom) {
        begin |= ImGui::IsItemActivated();
        if (edited) (isGeom ? geom : move) = true;
    };
    track(ImGui::DragFloat3("Position", it->pos, 0.05f), false);
    if (it->kind != K_LIGHT && it->kind != K_VOLUME && it->kind != K_ZONE) {
        if (it->kind == K_TREE || it->kind == K_ROCK) track(ImGui::DragFloat("Yaw", &it->rot[1], 1.f), false);
        else track(ImGui::DragFloat3("Rotation (x y z)", it->rot, 1.f), false);
    }
    if (it->kind == K_MESH || it->kind == K_ASSET || it->kind == K_VOLUME || it->kind == K_ZONE)
        track(ImGui::DragFloat3(it->kind == K_VOLUME || it->kind == K_ZONE ? "Half size" : "Scale", it->scale, 0.01f, 0.01f, 1000.f), false);
    if (it->kind == K_TREE || it->kind == K_ROCK) track(ImGui::DragFloat("Scale", &it->scale[0], 0.01f, 0.1f, 10.f), false);
    if (it->kind == K_PRIM) {
        int sh = it->shape;
        if (ImGui::Combo("Shape", &sh, "Box\0Cylinder\0")) { begin = true; it->shape = sh; geom = true; }
        track(ImGui::DragFloat3(it->shape == BOX ? "Size (w h d)" : "Size (dx h dz)", it->size, 0.01f, 0.02f, 200.f), true);
        if (it->shape == CYLINDER) track(ImGui::SliderInt("Sides", &it->sides, 3, 64), true);
        if (matCombo("Material", it->mat)) {
            begin = true; geom = true;
            if (auto* e = matlib::find(it->mat)) it->strength = e->strength;
        }
        track(ImGui::DragFloat("Texture size (m)", &it->uvScale, 0.01f, 0.05f, 20.f), true);
        track(ImGui::ColorEdit3("Tint", it->tint), true);
    }
    if (it->kind == K_PRIM || it->kind == K_MESH) {
        track(ImGui::DragFloat("Strength", &it->strength, 0.02f, 0.02f, 100.f, "%.2f x brick"), true);
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("How hard it is to breach: energy needed scales with this (brick 1, concrete 2, metal 5, planks 0.4)");
    }
    if (it->kind == K_PRIM || it->kind == K_MESH || it->kind == K_ASSET) {
        bool col = it->collide;
        if (ImGui::Checkbox("Collide / breakable", &col)) { begin = true; it->collide = col; geom = true; }
    }
    if (it->kind == K_ASSET) ImGui::Text("Asset: %s", it->asset.c_str());
    if (it->kind == K_LIGHT) {
        track(ImGui::ColorEdit3("Colour", it->color), true);
        track(ImGui::DragFloat("Intensity (cd)", &it->intensity, 2.f, 0.f, 100000.f, "%.0f", ImGuiSliderFlags_Logarithmic), true);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Physical units: a 60 W bulb is ~70 cd, a bright LED lamp ~150, a street light ~2000.\n"
                              "Next to daylight (~100000 lux) any lamp is invisible, as in reality: test at night or indoors.");
    }
    if (it->kind == K_ZONE) {
        track(ImGui::DragFloat("Falloff (m)", &it->falloff, 0.1f, 0.f, 100.f), true);
        if (matCombo("Ground texture", it->zoneMat, true)) { begin = true; geom = true; }
        if (!it->zoneMat.empty()) track(ImGui::DragFloat("Texture size (m)##zt", &it->zoneTile, 0.02f, 0.1f, 50.f), true);
    }
    if (it->kind == K_TREE) {
        int v = it->variant;
        if (ImGui::Combo("Variant", &v, TREE_NAME, trees::VARIANTS)) { begin = true; it->variant = v; geom = true; }
    }
    if (begin) {  // snapshot taken before this edit started
        std::vector<Item> snap = c->items;
        c->undo.push_back(snap);
    }
    if (geom) changed(*c, true);
    else if (move) changed(*c, false);
    if (ImGui::Button("Duplicate (Ctrl+D)")) duplicateSelected();
    ImGui::SameLine();
    if (ImGui::Button("Drop to ground")) {
        pushUndo(*c);
        snapToGround(*c, *it);
        changed(*c, false);
    }
    ImGui::SameLine();
    if (ImGui::Button("Delete (Del)")) deleteSelected();
}

// ---------------------------------------------------------------- world map editor (M)
// Terrain sculpting and painting on the world map grid (worldmap.h), drawn rivers, and the snow brush. While it's
// on, the left mouse paints with the current tool instead of picking items.
bool mapMode = false, mapOpen = true;
int mapTool = worldmap::RAISE, mapTexSlot = 1;
float mapRadius = 12.f, mapStrength = 2.f, mapFlatten = 0.f;
bool stroking = false;
enum { MT_RIVER = 100, MT_SNOW_ADD, MT_SNOW_REMOVE };
std::vector<worldmap::RiverPt> riverPts;
float riverW = 4.f, riverD = 1.2f;  // the pen: each point takes the width / depth it was drawn with (wheel / Shift+wheel)
float snowRate = 2.f;

bool terrainUnderMouse(float out[3]) {
    float d[3];
    mouseRay(d);
    float t = rayTerrain(view.pos, d);
    if (t < 0) return false;
    for (int k = 0; k < 3; k++) out[k] = view.pos[k] + d[k] * t;
    return true;
}
void groundCircle(const float c[3], float r, ImU32 col) {
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    ImVec2 prev;
    bool have = false;
    for (int i = 0; i <= 64; i++) {
        float a = i / 64.f * 6.2831853f, q[3] = {c[0] + cosf(a) * r, 0, c[2] + sinf(a) * r};
        q[1] = host.height(q[0], q[2]) + 0.15f;
        ImVec2 sc;
        bool ok = project(q, sc);
        if (ok && have) dl->AddLine(prev, sc, col, 2.f);
        prev = sc; have = ok;
    }
}
void commitRiver() {
    if (riverPts.size() < 2) { status = "a river needs at least two points"; return; }
    float total = 0;
    for (size_t i = 1; i < riverPts.size(); i++) total += hypotf(riverPts[i].x - riverPts[i - 1].x, riverPts[i].z - riverPts[i - 1].z);
    worldmap::beginStroke();
    worldmap::addRiver(riverPts);
    status = fmt("river carved: %zu points, %.0f m (water runs from the first point downhill)", riverPts.size(), total);
    riverPts.clear();
}

void mapTools() {
    ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureKeyboard && ImGui::IsKeyPressed(ImGuiKey_M, false)) { mapMode = !mapMode; mapOpen = true; }
    if (!mapMode) { if (host.snowBrush) host.snowBrush(0, 0, 0, 0); return; }
    // ---- panel
    ImGui::SetNextWindowPos(ImVec2(20, 60), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(330, 560), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(250, 150), ImVec2(FLT_MAX, std::max(200.f, io.DisplaySize.y - 20.f)));
    if (ImGui::Begin("World map (M)", &mapOpen)) {
        static const struct { int t; const char* n; const char* tip; } tools[] = {
            {worldmap::RAISE, "Raise", "Hold to build up ground (strength = m/s at the centre)"},
            {worldmap::LOWER, "Lower", "Hold to dig down"},
            {worldmap::SMOOTH, "Smooth", "Average out bumps"},
            {worldmap::FLATTEN, "Flatten", "Pull toward the flatten height (Ctrl+click picks it)"},
            {worldmap::NOISE, "Roughen", "Add natural unevenness"},
            {worldmap::PAINT_GRASS, "Grass +", "Paint grass (up to 2x, also on bare ground)"},
            {worldmap::ERASE_GRASS, "Grass -", "Remove grass"},
            {worldmap::PAINT_TREES, "Trees +", "Paint forest density (up to 2x)"},
            {worldmap::ERASE_TREES, "Trees -", "Clear trees"},
            {worldmap::PAINT_TEXTURE, "Texture", "Paint the selected ground texture"},
            {worldmap::ERASE_TEXTURE, "Texture -", "Back to the natural ground"},
            {MT_RIVER, "River", "Click points (or drag) along the course, Enter to carve it. Water runs from the first point."},
            {worldmap::ERASE_WATER, "Water -", "Remove drawn river water (the carved bed stays)"},
            {MT_SNOW_ADD, "Snow +", "Pile snow (it slides when steeper than the angle of repose)"},
            {MT_SNOW_REMOVE, "Snow -", "Shovel snow away"},
        };
        int col = 0;
        for (auto& t : tools) {
            if (ImGui::RadioButton(t.n, mapTool == t.t)) { mapTool = t.t; if (mapTool != MT_RIVER) riverPts.clear(); }
            ImGui::SetItemTooltip("%s", t.tip);
            if (++col % 3) ImGui::SameLine(); else col = 0;
        }
        ImGui::NewLine();
        ImGui::SeparatorText("Brush");
        ImGui::SliderFloat("Radius", &mapRadius, 1.f, 500.f, "%.1f m", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Strength", &mapStrength, 0.05f, 100.f, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::SetItemTooltip("Raise / lower: metres per second at the centre. Paint / smooth: how fast it takes.");
        if (mapTool == worldmap::FLATTEN) ImGui::DragFloat("Flatten height", &mapFlatten, 0.1f, -500.f, 5000.f, "%.1f m");
        if (mapTool == worldmap::PAINT_TEXTURE || mapTool == worldmap::ERASE_TEXTURE) {
            ImGui::SeparatorText("Ground textures");
            for (int i = 1; i < worldmap::PALETTE; i++) {
                ImGui::PushID(i);
                if (ImGui::RadioButton("##use", mapTexSlot == i)) mapTexSlot = i;
                ImGui::SameLine();
                std::string nm = worldmap::paletteName(i);
                ImGui::SetNextItemWidth(200);
                if (matCombo(fmt("slot %d", i).c_str(), nm, true)) { worldmap::setPalette(i, nm); mapTexSlot = i; }
                ImGui::PopID();
            }
        }
        if (mapTool == MT_RIVER) {
            ImGui::SeparatorText("River pen");
            ImGui::DragFloat("Half-width (wheel)", &riverW, 0.05f, 0.5f, 60.f, "%.1f m");
            ImGui::DragFloat("Depth (Shift+wheel)", &riverD, 0.05f, 0.2f, 20.f, "%.1f m");
            ImGui::TextWrapped("Click or hold and drag to draw; scroll while drawing to widen / narrow from that point on.");
            ImGui::Text("%zu points", riverPts.size());
            if (ImGui::Button("Carve river (Enter)")) commitRiver();
            ImGui::SameLine();
            if (ImGui::Button("Undo point (Bksp)") && !riverPts.empty()) riverPts.pop_back();
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) riverPts.clear();
        }
        if (mapTool == MT_SNOW_ADD || mapTool == MT_SNOW_REMOVE) {
            ImGui::SliderFloat("Snow rate", &snowRate, 0.1f, 50.f, "%.1f m/s", ImGuiSliderFlags_Logarithmic);
            ImGui::TextWrapped("Snow lives in the simulation (Path Tracer > Weather): it slides, melts below the snow line and the melt water runs toward rivers.");
        }
        ImGui::Separator();
        if (ImGui::Button("Undo (Ctrl+Z)")) worldmap::undo();
        ImGui::SameLine();
        if (ImGui::Button(worldmap::dirty() ? "Save (Ctrl+S) *" : "Save (Ctrl+S)")) { std::string err; if (!saveAll(err)) status = "save failed: " + err; }
        ImGui::TextWrapped("4 km x 4 km around the origin, 2 m cells. Saved to world/terrain/map.bin. M: back to object editing.");
        if (!status.empty()) ImGui::TextWrapped("%s", status.c_str());
    }
    ImGui::End();
    if (!mapOpen) { mapMode = false; mapOpen = true; }

    // ---- keys
    if (!io.WantCaptureKeyboard) {
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Z, false)) worldmap::undo();
        if (mapTool == MT_RIVER) {
            if (ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false)) commitRiver();
            if (ImGui::IsKeyPressed(ImGuiKey_Backspace, false) && !riverPts.empty()) riverPts.pop_back();
            if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) riverPts.clear();
        }
    }

    // ---- brush
    float at[3];
    bool hit = !io.WantCaptureMouse && terrainUnderMouse(at);
    bool snowT = mapTool == MT_SNOW_ADD || mapTool == MT_SNOW_REMOVE;
    if (mapTool == MT_RIVER) {
        // the pen: wheel = width, Shift+wheel = depth (over the world, not the panels)
        if (!io.WantCaptureMouse && io.MouseWheel != 0.f) {
            float k = powf(1.12f, io.MouseWheel);
            if (io.KeyShift) riverD = std::clamp(riverD * k, 0.2f, 20.f);
            else riverW = std::clamp(riverW * k, 0.5f, 60.f);
        }
        // the course so far, and the next segment to the mouse
        ImDrawList* dl = ImGui::GetBackgroundDrawList();
        for (auto& rp : riverPts) {  // each point's own width
            float c[3] = {rp.x, 0, rp.z};
            groundCircle(c, rp.width, IM_COL32(80, 170, 255, 70));
        }
        for (size_t i = 0; i < riverPts.size(); i++) {
            float q[3] = {riverPts[i].x, host.height(riverPts[i].x, riverPts[i].z) + 0.3f, riverPts[i].z};
            ImVec2 a;
            if (!project(q, a)) continue;
            dl->AddCircleFilled(a, 4, IM_COL32(80, 170, 255, 230));
            if (i + 1 < riverPts.size() || hit) {
                float q2[3];
                if (i + 1 < riverPts.size()) { q2[0] = riverPts[i + 1].x; q2[2] = riverPts[i + 1].z; }
                else { q2[0] = at[0]; q2[2] = at[2]; }
                q2[1] = host.height(q2[0], q2[2]) + 0.3f;
                ImVec2 b;
                if (project(q2, b)) dl->AddLine(a, b, IM_COL32(80, 170, 255, i + 1 < riverPts.size() ? 230 : 120), 3.f);
            }
        }
        if (hit) {
            groundCircle(at, riverW, IM_COL32(80, 170, 255, 200));
            ImVec2 sp;
            float q[3] = {at[0], at[1] + 0.5f, at[2]};
            if (project(q, sp))
                dl->AddText(ImVec2(sp.x + 14, sp.y - 8), IM_COL32(200, 230, 255, 255), fmt("width %.1f m  depth %.1f m", 2.f * riverW, riverD).c_str());
        }
        if (hit && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) riverPts.push_back({at[0], at[2], riverW, riverD});
        else if (hit && ImGui::IsMouseDown(ImGuiMouseButton_Left) && !riverPts.empty() &&
                 hypotf(at[0] - riverPts.back().x, at[2] - riverPts.back().z) > std::clamp(riverW * 0.75f, 2.f, 8.f))
            riverPts.push_back({at[0], at[2], riverW, riverD});  // dragging draws the course with the pen's current size
        if (host.snowBrush) host.snowBrush(0, 0, 0, 0);
        return;
    }
    if (hit) groundCircle(at, mapRadius, snowT ? IM_COL32(230, 240, 255, 200) : IM_COL32(255, 200, 60, 200));
    if (hit && io.KeyCtrl && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && mapTool == worldmap::FLATTEN) {
        mapFlatten = at[1];
        status = fmt("flatten height %.1f m", mapFlatten);
        return;
    }
    bool down = hit && ImGui::IsMouseDown(ImGuiMouseButton_Left);
    if (snowT) {
        if (host.snowBrush) host.snowBrush(at[0], at[2], down ? mapRadius : 0.f, mapTool == MT_SNOW_ADD ? snowRate : -snowRate);
        return;
    }
    if (host.snowBrush) host.snowBrush(0, 0, 0, 0);
    if (down && !stroking) { worldmap::beginStroke(); stroking = true; }
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) stroking = false;
    if (down) worldmap::brush((worldmap::Tool)mapTool, at[0], at[2], mapRadius, mapStrength, std::min(io.DeltaTime, 0.05f), mapFlatten, mapTexSlot);
}

}  // namespace

void init(const Host& h) {
    host = h;
    matlib::init(h.worldDir);
    scanAssets();
}
bool active() { return on; }
void setActive(bool v) { on = v; dragging = false; }
bool takeMoveOnly() { bool r = moveOnly; moveOnly = false; return r; }
bool wantsMouse() { return on; }

void demoHouse() {
    float p[3];
    placePoint(p);
    float yaw = camYawSnapped() * DEG;
    p[0] = snapv(p[0], 0.5f); p[2] = snapv(p[2], 0.5f);
    // a house in its own frame: x across (8 m), z deep (6 m), walls 0.25 thick, 3 m storey; front at -z
    const float W = 8, D = 6, H = 3, T = 0.25f;
    std::vector<Item> parts;
    auto put = [&](Item it, float x, float y, float z) {
        it.pos[0] = p[0] + cosf(yaw) * x + sinf(yaw) * z;
        it.pos[2] = p[2] - sinf(yaw) * x + cosf(yaw) * z;
        it.pos[1] = p[1] + y;
        it.rot[1] += yaw / DEG;
        parts.push_back(it);
    };
    put(prim(BOX, W + 0.6f, 0.3f, D + 0.6f, "concrete", "slab"), 0, -0.1f, 0);
    put(prim(BOX, W, H, T, "brick", "back wall"), 0, 0.2f, D * 0.5f - T * 0.5f);
    // side walls with a window each: below, above, either side of a 1.2 x 1.2 m opening
    for (int s = -1; s <= 1; s += 2) {
        float x = s * (W * 0.5f - T * 0.5f), d = D - 2 * T;
        Item lo = prim(BOX, d, 1.0f, T, "brick", "side wall"); lo.rot[1] = 90;
        put(lo, x, 0.2f, 0);
        Item hi = prim(BOX, d, H - 2.2f, T, "brick", "side wall"); hi.rot[1] = 90;
        put(hi, x, 2.4f, 0);
        Item a = prim(BOX, (d - 1.2f) * 0.5f, 1.2f, T, "brick", "side wall"); a.rot[1] = 90;
        put(a, x, 1.2f, -(1.2f + (d - 1.2f) * 0.5f) * 0.5f);
        put(a, x, 1.2f, (1.2f + (d - 1.2f) * 0.5f) * 0.5f);
    }
    // front: door 1.1 x 2.2 in the middle, lintel over it
    float side = (W - 1.1f) * 0.5f;
    put(prim(BOX, side, H, T, "brick", "front wall"), -(1.1f + side) * 0.5f, 0.2f, -D * 0.5f + T * 0.5f);
    put(prim(BOX, side, H, T, "brick", "front wall"), (1.1f + side) * 0.5f, 0.2f, -D * 0.5f + T * 0.5f);
    put(prim(BOX, 1.1f, H - 2.2f, T, "concrete", "lintel"), 0, 2.4f, -D * 0.5f + T * 0.5f);
    for (int sx = -1; sx <= 1; sx += 2)  // porch posts
        put(prim(CYLINDER, 0.3f, H, 0.3f, "planks", "post"), sx * 1.2f, 0.2f, -D * 0.5f - 1.4f);
    put(prim(BOX, W + 0.8f, 0.25f, D + 2.4f, "planks", "roof"), 0, 0.2f + H, -0.8f);
    Item L;
    L.kind = K_LIGHT;
    L.intensity = 400;
    put(L, 0, 2.5f, 0.5f);
    Item V;
    V.kind = K_VOLUME;
    V.name = fmt("house_%d", (int)now() % 100000);
    V.scale[0] = W * 0.5f; V.scale[1] = H * 0.5f + 0.2f; V.scale[2] = D * 0.5f;
    put(V, 0, H * 0.5f + 0.2f, 0);
    parts.back().scale[0] = fabsf(cosf(yaw)) * W * 0.5f + fabsf(sinf(yaw)) * D * 0.5f;  // axis-aligned box around it
    parts.back().scale[2] = fabsf(sinf(yaw)) * W * 0.5f + fabsf(cosf(yaw)) * D * 0.5f;
    parts.back().rot[1] = 0;
    ECell* c = open(cellOf(p[0]), cellOf(p[2]));
    if (!c) { status = "that cell is still loading"; return; }
    pushUndo(*c);
    for (auto& it : parts) c->items.push_back(it);
    changed(*c, true);
    sel = {c->cx, c->cz, (int)c->items.size() - (int)parts.size() + 1};
    status = "example house added";
}

bool saveAll(std::string& err) {
    int n = 0;
    for (auto& [k, c] : cells) {
        if (!c.dirty) continue;
        std::string path = cellPath(c.cx, c.cz);
        std::error_code ec;
        std::filesystem::create_directories(host.worldDir + "cells", ec);
        if (std::filesystem::exists(path) && !std::filesystem::exists(path + ".bak")) std::filesystem::copy_file(path, path + ".bak", ec);
        ModelData d;
        build(c, d, false);
        if (!writeGlb(path, d, err)) return false;
        c.dirty = false;
        n++;
    }
    status = fmt("saved %d cell%s", n, n == 1 ? "" : "s");
    if (worldmap::dirty()) {
        if (!worldmap::save(err)) return false;
        status += ", world map";
    }
    return true;
}

void apply() {
    for (auto [cx, cz] : reverts) {  // the saved file (or nothing) back into the stream, edits dropped
        auto m = std::make_shared<Model>();
        std::string err, path = cellPath(cx, cz);
        if (std::filesystem::exists(path) && !loadGltf(path, m->data, err)) { status = "revert failed: " + err; continue; }
        std::vector<bool> coll(m->data.meshes.size(), false);
        for (auto& in : m->data.instances) if (in.mesh >= 0 && in.collide) coll[in.mesh] = true;
        for (size_t i = 0; i < m->data.meshes.size(); i++) {
            auto& md = m->data.meshes[i];
            if (coll[i] && !md.idx.empty()) md.shape = physics::meshShape(md.verts.data(), 8, (uint32_t)(md.verts.size() / 8), md.idx.data(), (uint32_t)(md.idx.size() / 3));
        }
        m->state = 1;
        if (!host.upload(*m)) { status = "revert: upload failed"; continue; }
        std::vector<std::shared_ptr<Model>> freed;
        stream::setCellModel(cx, cz, m, freed);
        for (auto& f : freed) freeModel(f);
        cells.erase({cx, cz});
        if (sel.cx == cx && sel.cz == cz) sel.idx = -1;
        status = "cell reverted to the saved file";
    }
    reverts.clear();
    if (cells.empty()) return;
    bool editing = ImGui::GetCurrentContext() && ImGui::IsAnyItemActive();
    for (auto& [k, c] : cells) {
        auto live = stream::cellModel(c.cx, c.cz);
        // the stream reloaded the cell from disk under unsaved edits: put them back
        if (c.modified && live && live != c.live) c.rebuild = true;
        if (c.rebuild) {
            if (editing && now() - lastRebuild < 0.1) continue;  // slider drags: ~10 rebuilds a second
            c.rebuild = c.refresh = false;
            pushLive(c);
        } else if (c.refresh) {
            c.refresh = false;
            Item* d = dragging ? selected() : nullptr;
            moveOnly = d && (d->kind == K_PRIM || d->kind == K_MESH || d->kind == K_ASSET || d->kind == K_LIGHT);
            bool ok = true;
            for (auto& it : c.items) ok &= it.slot >= 0 ? moveInPlace(c, it) : true;
            if (ok) stream::refreshCell(c.cx, c.cz);
            else pushLive(c);
        }
    }
}

void ui(const View& v) {
    view = v;
    if (!on) return;
    pollJobs();
    ImGuiIO& io = ImGui::GetIO();
    // keep the cells around the camera open for picking
    int ccx = cellOf(v.pos[0]), ccz = cellOf(v.pos[2]);
    for (int dz = -1; dz <= 1; dz++)
        for (int dx = -1; dx <= 1; dx++) open(ccx + dx, ccz + dz);

    mapTools();
    // ---- mouse: pick, drag
    float o[3] = {v.pos[0], v.pos[1], v.pos[2]}, d[3];
    mouseRay(d);
    if (!mapMode && !io.WantCaptureMouse && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        Hit h = pick(o, d);
        sel = {h.cx, h.cz, h.idx};
        if (Item* it = selected()) {
            dragging = true;
            dragMoved = false;
            dragVertical = io.KeyAlt;
            float hp[3] = {o[0] + d[0] * h.t, o[1] + d[1] * h.t, o[2] + d[2] * h.t};
            memcpy(dragP0, hp, 12);
            for (int k = 0; k < 3; k++) dragOff[k] = it->pos[k] - hp[k];
            dragPlaneY = hp[1];
            float n[3] = {-v.fwd[0], 0, -v.fwd[2]}, l = sqrtf(n[0] * n[0] + n[2] * n[2]);
            dragN[0] = n[0] / std::max(l, 1e-6f); dragN[1] = 0; dragN[2] = n[2] / std::max(l, 1e-6f);
        }
    }
    snapInvert = io.KeyShift;
    if (!io.WantCaptureKeyboard && ImGui::IsKeyPressed(ImGuiKey_N, false)) snap = !snap;
    if (dragging) {
        Item* it = selected();
        ECell* c = selCell();
        if (!it || !c || !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            dragging = false;
            if (it && c && dragMoved) { changed(*c, false); rehome(*c, sel.idx); }  // full refresh once the drag ends
        } else if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 1.f)) {
            float t;
            float p[3];
            if (dragVertical) {  // plane through the grab point facing the camera, vertical motion only
                float den = d[0] * dragN[0] + d[2] * dragN[2];
                t = fabsf(den) > 1e-6f ? ((dragP0[0] - o[0]) * dragN[0] + (dragP0[2] - o[2]) * dragN[2]) / den : -1;
                if (t > 0) {
                    float y = o[1] + d[1] * t + dragOff[1];
                    if (!dragMoved) pushUndo(*c);
                    it->pos[1] = snapv(y, grid);
                    dragMoved = true;
                    changed(*c, false);
                }
            } else if (fabsf(d[1]) > 1e-4f && (t = (dragPlaneY - o[1]) / d[1]) > 0 && t < 2000.f) {
                for (int k = 0; k < 3; k++) p[k] = o[k] + d[k] * t;
                if (!dragMoved) pushUndo(*c);
                it->pos[0] = snapv(p[0] + dragOff[0], grid);
                it->pos[2] = snapv(p[2] + dragOff[2], grid);
                if (groundSnap && groundable(*it)) snapToGround(*c, *it);
                dragMoved = true;
                changed(*c, false);
            }
        }
    }

    // ---- keys
    if (!io.WantCaptureKeyboard) {
        bool ctrl = io.KeyCtrl;
        if (ImGui::IsKeyPressed(ImGuiKey_Delete)) deleteSelected();
        if (ctrl && ImGui::IsKeyPressed(ImGuiKey_D, false)) duplicateSelected();
        if (ctrl && ImGui::IsKeyPressed(ImGuiKey_S, false)) { std::string err; if (!saveAll(err)) status = "save failed: " + err; }
        if (!mapMode && ctrl && ImGui::IsKeyPressed(ImGuiKey_Z, false))
            if (ECell* c = selCell(); c && !c->undo.empty()) {
                c->items = c->undo.back();
                c->undo.pop_back();
                if (sel.idx >= (int)c->items.size()) sel.idx = -1;
                changed(*c, true);
            }
        Item* it = selected();
        ECell* c = selCell();
        if (it && c) {
            float mv[3] = {0, 0, 0};
            float g = std::max(grid, 0.05f);
            if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) mv[0] -= g;
            if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) mv[0] += g;
            if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) mv[2] += g;
            if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) mv[2] -= g;
            if (ImGui::IsKeyPressed(ImGuiKey_PageUp)) mv[1] += g;
            if (ImGui::IsKeyPressed(ImGuiKey_PageDown)) mv[1] -= g;
            bool rot = !ctrl && ImGui::IsKeyPressed(ImGuiKey_R, false);
            if (mv[0] || mv[1] || mv[2] || rot) {
                pushUndo(*c);
                for (int k = 0; k < 3; k++) it->pos[k] += mv[k];
                if (rot) it->rot[1] = fmodf(it->rot[1] + (io.KeyShift ? -rotStep : rotStep) + 360.f, 360.f);
                changed(*c, false);
                if (mv[0] || mv[2]) rehome(*c, sel.idx);
            }
        }
    }

    // ---- overlay
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    for (auto& [k, c] : cells)
        for (size_t i = 0; i < c.items.size(); i++) {
            const Item& it = c.items[i];
            bool isSel = k.first == sel.cx && k.second == sel.cz && (int)i == sel.idx;
            if (isSel) continue;
            if (it.kind == K_LIGHT) {
                ImVec2 s;
                if (project(it.pos, s)) dl->AddCircle(s, 6.f, IM_COL32(255, 220, 120, 200), 0, 2.f);
            } else if ((it.kind == K_VOLUME && showVolumes) ) drawBox(dl, c, it, IM_COL32(80, 200, 255, 110), 1.f);
            else if (it.kind == K_ZONE && showVolumes) drawBox(dl, c, it, IM_COL32(120, 255, 120, 110), 1.f);
            else if (showAll) drawBox(dl, c, it, IM_COL32(255, 255, 255, 50), 1.f);
        }
    if (Item* it = selected()) drawBox(dl, *selCell(), *it, IM_COL32(255, 210, 40, 255), 2.f);
    {   // placement point
        ImVec2 c(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f);
        dl->AddLine(ImVec2(c.x - 6, c.y), ImVec2(c.x + 6, c.y), IM_COL32(255, 255, 255, 160));
        dl->AddLine(ImVec2(c.x, c.y - 6), ImVec2(c.x, c.y + 6), IM_COL32(255, 255, 255, 160));
    }

    // ---- panel
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 400, 20), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(380, 720), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(250, 150), ImVec2(FLT_MAX, std::max(200.f, io.DisplaySize.y - 20.f)));  // scrolls, never off-screen
    ImGui::Begin("World editor (Tab)");
    int dirty = 0;
    for (auto& [k, c] : cells) dirty += c.dirty;
    if (ImGui::Button(dirty ? "Save (Ctrl+S) *" : "Save (Ctrl+S)")) { std::string err; if (!saveAll(err)) status = "save failed: " + err; }
    ImGui::SameLine();
    ImGui::TextDisabled("%d unsaved cell%s", dirty, dirty == 1 ? "" : "s");
    if (!status.empty()) ImGui::TextWrapped("%s", status.c_str());
    ImGui::Checkbox("Snap (N)", &snap);
    ImGui::SetItemTooltip("Hold Shift while dragging to invert: free movement with snap on, grid steps with it off");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    ImGui::DragFloat("Grid (m)", &grid, 0.01f, 0.01f, 10.f, "%.2f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    ImGui::DragFloat("R step", &rotStep, 1.f, 1.f, 90.f, "%.0f");
    ImGui::Checkbox("Show volumes / zones", &showVolumes);
    ImGui::SameLine();
    ImGui::Checkbox("Show all bounds", &showAll);

    if (ImGui::CollapsingHeader("Add (at the crosshair)", ImGuiTreeNodeFlags_DefaultOpen)) {
        matCombo("New material", newMat);
        if (ImGui::Button("Wall")) addItem(prim(BOX, 4, 3, 0.25f, newMat, "wall"));
        ImGui::SameLine();
        if (ImGui::Button("Floor")) addItem(prim(BOX, 4, 0.2f, 4, newMat, "floor"));
        ImGui::SameLine();
        if (ImGui::Button("Box")) addItem(prim(BOX, 1, 1, 1, newMat, "box"));
        ImGui::SameLine();
        if (ImGui::Button("Beam")) addItem(prim(BOX, 4, 0.3f, 0.3f, newMat, "beam"));
        if (ImGui::Button("Pillar")) addItem(prim(CYLINDER, 0.4f, 3, 0.4f, newMat, "pillar"));
        ImGui::SameLine();
        if (ImGui::Button("Cylinder")) addItem(prim(CYLINDER, 1, 1, 1, newMat, "cylinder"));
        ImGui::SameLine();
        if (ImGui::Button("Example house")) demoHouse();
        if (ImGui::Button("Light")) { Item it; it.kind = K_LIGHT; it.name = "light"; addItem(it); }
        ImGui::SameLine();
        if (ImGui::Button("Bake volume")) {
            Item it; it.kind = K_VOLUME; it.scale[0] = 4; it.scale[1] = 1.6f; it.scale[2] = 3;
            it.name = fmt("vol_%d", (int)now() % 100000);
            addItem(it);
        }
        ImGui::SameLine();
        if (ImGui::Button("Flatten zone")) { Item it; it.kind = K_ZONE; it.scale[0] = it.scale[2] = 10; it.scale[1] = 0.5f; addItem(it); }
        static int treeVar = 0;
        ImGui::SetNextItemWidth(130);
        ImGui::Combo("##tv", &treeVar, TREE_NAME, trees::VARIANTS);
        ImGui::SameLine();
        if (ImGui::Button("Tree")) { Item it; it.kind = K_TREE; it.variant = treeVar; it.rot[1] = (float)(rand() % 360); addItem(it); }
        ImGui::SameLine();
        if (ImGui::Button("Rock")) { Item it; it.kind = K_ROCK; it.rot[1] = (float)(rand() % 360); addItem(it); }
        ImGui::Checkbox("Asset browser", &showBrowser);
        ImGui::SameLine();
        ImGui::TextDisabled("(drop files / folders on the window to import)");
        if (ImGui::Button("Rescan assets + textures")) { scanAssets(); matlib::rescan(); }
    }
    if (ImGui::CollapsingHeader("Selection", ImGuiTreeNodeFlags_DefaultOpen)) inspector();
    if (ImGui::CollapsingHeader("Cell contents")) {
        ECell* c = open(ccx, ccz);
        if (!c) ImGui::TextDisabled("loading...");
        else {
            ImGui::Text("cell %d, %d: %zu items", c->cx, c->cz, c->items.size());
            ImGui::BeginChild("list", ImVec2(0, 200), true);
            for (size_t i = 0; i < c->items.size(); i++) {
                const Item& it = c->items[i];
                std::string label = fmt("%s  %s##%zu", KIND_NAME[it.kind], it.name.empty() ? (it.kind == K_ASSET ? it.asset.c_str() : "") : it.name.c_str(), i);
                if (ImGui::Selectable(label.c_str(), sel.cx == c->cx && sel.cz == c->cz && sel.idx == (int)i)) sel = {c->cx, c->cz, (int)i};
            }
            ImGui::EndChild();
            if (ImGui::Button("Revert cell to saved")) reverts.push_back({c->cx, c->cz});
        }
    }
    if (ImGui::CollapsingHeader("Help")) {
        ImGui::TextWrapped(
            "Click: select.  Drag: move on the ground plane (Alt+drag: up / down).\n"
            "R / Shift+R: rotate by the step.  Arrows, PgUp/PgDn: nudge by the grid.\n"
            "Ctrl+D duplicate, Del delete, Ctrl+Z undo, Ctrl+S save.\n"
            "Right mouse + WASD / Q E: fly.  G: throw a rock to test.\n"
            "Primitives are breakable: strength scales the energy a breach takes. Textures: built-in set, plus any "
            "PNG / JPG in world/textures/ (or dropped on the window). Drag assets from the Assets window into the world. Saved into world/cells/<x>_<z>.glb (old file kept as .bak).");
    }
    ImGui::End();
    if (showBrowser) assetBrowser();
    if (showScatter) scatterPanel();
    assetDrop();
}

void dropFiles(const std::vector<std::string>& paths) {
    if (host.worldDir.empty()) return;
    setActive(true);
    showBrowser = true;
    importPaths(paths);
}

}  // namespace editor

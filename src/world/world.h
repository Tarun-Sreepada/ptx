// CPU-side world data: what a .glb exported from Blender turns into, and the streaming API.
//
// World layout (WORLD_DIR):
//   cells/<x>_<z>.glb  one per 64 m x 64 m cell, authored in WORLD coordinates (cell x covers [x*64, x*64+64))
//   assets/<name>.glb  reusable pieces (doors, furniture, building kits) in their own local space
// Node custom properties (Blender "Custom Properties", exported with "Include > Custom Properties"):
//   asset   = "<name>"  place assets/<name>.glb at this node's transform (the node's own mesh, if any, is a proxy)
//   flatten = 1         node is a box (e.g. an Empty, display "Cube"): terrain under its x/z footprint is pulled
//                       to the node's height (city ground, plots, roads). falloff = <m> blend band (default 8)
//   collide = 0         no physics for this node's mesh (foliage, decals, light fixtures)
//   bake    = 1         node is a box around a building interior: the renderer bakes an irradiance probe grid in it
//                       (lights on + lights off), and the lights inside it switch together (E key)
// Point/spot lights export via KHR_lights_punctual (Blender exporter: "Punctual Lights").
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <cstring>
#include <string>
#include <vector>

#include "world/terrain.h"

struct Xf { float m[3][4]; };  // row-major 3x4, same layout as VkTransformMatrixKHR
inline Xf xfIdentity() { return {{{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}}}; }
inline Xf xfMul(const Xf& a, const Xf& b) {
    Xf r;
    for (int i = 0; i < 3; i++)
        for (int j = 0; j < 4; j++)
            r.m[i][j] = a.m[i][0] * b.m[0][j] + a.m[i][1] * b.m[1][j] + a.m[i][2] * b.m[2][j] + (j == 3 ? a.m[i][3] : 0);
    return r;
}
inline void xfPoint(const Xf& a, const float p[3], float o[3]) {
    for (int i = 0; i < 3; i++) o[i] = a.m[i][0] * p[0] + a.m[i][1] * p[1] + a.m[i][2] * p[2] + a.m[i][3];
}
// world AABB of a transformed object AABB
inline void xfAabb(const Xf& a, const float mn[3], const float mx[3], float omn[3], float omx[3]) {
    for (int i = 0; i < 3; i++) {
        float c = a.m[i][3], e = 0;
        for (int j = 0; j < 3; j++) {
            c += a.m[i][j] * (mn[j] + mx[j]) * 0.5f;
            e += fabsf(a.m[i][j]) * (mx[j] - mn[j]) * 0.5f;
        }
        omn[i] = c - e;
        omx[i] = c + e;
    }
}

// How a static mesh fails when hit (main.cpp breakStatic): Voronoi shatter (concrete, stone, plaster, ...),
// masonry that comes apart along its mortar into whole bricks, or metal that dents and only tears when hit very hard.
enum { BREAK_SHATTER, BREAK_BRICK, BREAK_METAL };
inline int breakKindOf(const std::string& mat) { return mat == "brick" ? BREAK_BRICK : mat == "metal" ? BREAK_METAL : BREAK_SHATTER; }
struct MeshData {
    std::vector<float> verts;      // pos3 nrm3 uv2
    std::vector<uint32_t> idx;
    std::vector<uint32_t> triMat;  // model-local material index per triangle
    float bmin[3] = {}, bmax[3] = {};
    void* shape = nullptr;         // physics mesh shape (refcounted, see physics.h), built by the loader thread
    float strength = 1.f;          // breakability (main.cpp breakStatic): x the breach energy of a brick wall
    int breakKind = 0;             // BREAK_*: how it fails (from the building material, "mat" extra)
    int skin = -1;                 // ModelData::skins index when skinned (bind pose verts, model space; not welded)
    std::vector<float> skinW;      // skinned: per vertex 4 weights + 4 joint indices (as floats), skel.h
};
// glTF metal-rough PBR plus the KHR_materials_* lobes the renderer shades (clearcoat, sheen, anisotropy,
// transmission, ior, specular). Texture indices are into ModelData::textures, -1 = none.
struct MaterialData {
    float color[4] = {1, 1, 1, 1}; float emissive[3] = {}; int tex = -1;
    float rough = 0.8f, metal = 0.f;
    int normalTex = -1;              // tangent-space normal (rgb); alpha = height when heightScale > 0
    int ormTex = -1;                 // g = roughness, b = metalness (glTF metallicRoughnessTexture)
    int occTex = -1, emisTex = -1;   // occlusion (r), emissive colour
    float normalScale = 1.f, occStrength = 1.f;
    float heightScale = 0.f;         // parallax occlusion depth (m), ptx extension (material extras "heightScale")
    float clearcoat = 0.f, clearcoatRough = 0.03f;
    float sheen = 0.f, sheenColor[3] = {1, 1, 1};
    float anisotropy = 0.f, anisoRot = 0.f;
    float transmission = 0.f, ior = 1.5f, specular = 1.f;
    float translucency = 0.f;        // diffuse transmission (leaves, thin fabric): KHR_materials_diffuse_transmission
    float sss = 0.f;                 // subsurface scattering profile (sss.glsl): 1 skin, 2 wax, 3 snow, 4 marble; extras "sss"
};
// Subsurface profile from a name or number in extras ("skin", "wax", "snow", "marble"), 0 = none.
float sssProfileOf(const std::string& s);
// Every texture index of a material, e.g. to remap them when copying materials between models.
template <class F> void forEachTex(MaterialData& m, F f) {
    for (int* t : {&m.tex, &m.normalTex, &m.ormTex, &m.occTex, &m.emisTex}) *t = f(*t);
}
struct TextureData {
    uint32_t w = 0, h = 0; std::vector<uint8_t> rgba; bool linear = false;  // sRGB unless linear (normal / ORM data), <= 2048
    // BC7 blocks of every mip level, level 0 first (texcomp.h); empty = upload rgba and build mips on the GPU
    std::vector<uint8_t> bc7; uint32_t bcMips = 0; std::vector<uint64_t> bcLevels;  // byte offset of each level in bc7
};
struct LightData { float pos[3]; float color[3]; };                         // color = rgb * intensity, glTF units (candela)
struct ProbeVolume { float mn[3], mx[3]; std::string name; };  // world-space box (bake = 1 node)
struct InstanceData {
    int mesh = -1; std::string asset; Xf xf; bool collide = true;
    std::string name, extras;  // node name and raw extras JSON (the editor keeps its primitive parameters there)
};
// hand-placed procedural things: extras tree = <variant> or rock = 1 on a mesh-less node
enum PropKind { PROP_TREE, PROP_ROCK };
struct PropData { int kind = PROP_TREE, variant = 0; Xf xf; };

// Skeletal animation (glTF skins + animations, evaluated by skel.h). Joints are a skin's own list, so vertex joint
// indices index them directly; parents may come after children in it (order[] is parents first).
struct SkinData {
    std::vector<std::string> names;
    std::vector<int> parent;        // joint index, -1: a root, whose parent node's world transform is rootXf
    std::vector<int> order;         // evaluation order, parents first
    std::vector<float> rest;        // per joint local t3 r4 (quaternion xyzw) s3
    std::vector<Xf> invBind;        // model space -> joint space at bind
    std::vector<Xf> rootXf;         // per joint (identity unless a root)
};
struct ClipChannel { int joint = 0, path = 0; bool step = false; std::vector<float> t, v; };  // path 0 T, 1 R, 2 S
struct ClipData { std::string name; float duration = 0.f; int skin = 0; std::vector<ClipChannel> ch; };

struct ModelData {
    std::vector<MeshData> meshes;
    std::vector<SkinData> skins;
    std::vector<ClipData> clips;
    std::vector<MaterialData> materials;
    std::vector<TextureData> textures;
    std::vector<InstanceData> instances;  // scene nodes: own meshes or asset references
    std::vector<LightData> lights;
    std::vector<Zone> zones;
    std::vector<ProbeVolume> volumes;
    std::vector<PropData> props;
};
bool loadGltf(const std::string& path, ModelData& out, std::string& err);
// An asset reference ("door", "trees/oak/oak.gltf") -> its file: names may sit in subfolders of <world>/assets/;
// a name without .gltf / .glb means <name>.glb (older cells store it that way).
// Renamed / moved assets leave "old<TAB>new" in <world>/assets/aliases.txt so cells and forest.txt saved
// before still find them (gltf_load.cpp; re-read when the file changes, thread safe).
std::string resolveAsset(const std::string& worldDir, const std::string& name);
void addAssetAlias(const std::string& worldDir, const std::string& from, const std::string& to);
inline std::string assetFile(const std::string& worldDir, const std::string& name0) {
    const std::string name = resolveAsset(worldDir, name0);
    auto ends = [&](const char* e) { size_t n = strlen(e); return name.size() > n && name.compare(name.size() - n, n, e) == 0; };
    return worldDir + "assets/" + name + (ends(".gltf") || ends(".glb") ? "" : ".glb");
}
// Write a cell / asset .glb (gltf_write.cpp): meshes split per material into primitives, textures as PNG, lights as
// KHR_lights_punctual, volumes (bake = 1), zones (flatten = 1) and props as mesh-less nodes with extras.
bool writeGlb(const std::string& path, const ModelData& m, std::string& err);
// Minimal lookups in a node's extras JSON: {"asset":"door","flatten":1,"falloff":4} (flat keys, no arrays)
bool extraStr(const char* json, const char* key, std::string& out);
float extraNum(const char* json, const char* key, float def);

// ---------------------------------------------------------------- streaming
struct Model {  // one .glb, shared by every cell that references it
    std::string path;
    ModelData data;
    int state = 0;            // 0 loading, 1 ready, 2 failed (written under the stream lock)
    int refs = 0;
    void* gpu = nullptr;      // renderer-owned upload; null until uploaded
    size_t bytes = 0;         // approximate GPU size, for the upload budget
};
struct Placed { Model* model; int mesh; Xf xf; };
struct WorldLight { float pos[3]; float color[3]; };

namespace stream {
inline constexpr float CELL = 64.f;
struct Stats { int cellsKnown, cellsLoaded, cellsActive, models, loading, terrainTiles; };
void init(const std::string& worldDir, int threads);
void shutdown();
// Call once per frame (after the GPU fence). upload() returns false when this frame's budget is spent;
// release() frees the GPU side. Returns true when the set of placed meshes / lights / zones changed.
bool update(float x, float y, float z, int loadRadius, const std::function<bool(Model&)>& upload, const std::function<void(Model&)>& release);
void forEachPlaced(const std::function<void(const Placed&)>& f);
void terrainChanged();  // terrain function changed (river settings): rebuild collision tiles
// Destruction: a placed mesh can be swapped for a damaged copy (mesh 0 of `with`, uploaded, MeshData::shape set),
// found by placedKey (stable across cell reloads, and the same for the copy). The copy stays in place when the cell
// reloads. Returns the copy it replaced, if any, for the caller to free once nothing uses it.
std::string placedKey(const Placed& p);
std::shared_ptr<Model> overridePlaced(const std::string& key, std::shared_ptr<Model> with);
void clearOverrides(std::vector<std::shared_ptr<Model>>& freed);  // undo all damage
// Editor: the loaded own model of a cell (null if not loaded), and replacing it with an edited one (uploaded, shapes
// built). Creates the cell if needed. Old model and any damaged copies of its meshes go to `freed` for the caller to
// free after the frame. refreshCell re-places an active cell after its model's instances / lights were moved in place.
std::shared_ptr<Model> cellModel(int cx, int cz);
void setCellModel(int cx, int cz, std::shared_ptr<Model> m, std::vector<std::shared_ptr<Model>>& freed);
void refreshCell(int cx, int cz);
Model* findModel(const std::string& path);  // a loaded shared model (assets), or null
struct WorldProp { int kind, variant; Xf xf; };
const std::vector<WorldProp>& props();
const std::vector<WorldLight>& lights();
const std::vector<Zone>& zones();
const std::vector<ProbeVolume>& volumes();
Stats stats();
}  // namespace stream

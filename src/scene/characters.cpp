#include "scene/characters.h"
#include <unordered_map>
#include "sim/water.h"
#include "sim/water_splash.h"
#include "sim/trample.h"
#include <vector>
#include <string>
#include <map>
#include <memory>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <vulkan/vulkan.h>
#include "characters/character.h"
#include "world/world.h"
#include "core/limits.h"
#include "destruction/thrown.h"
#include "destruction/tree_damage.h"
#include "gpu/accel.h"
#include "gpu/context.h"
#include "gpu/memory.h"
#include "gpu/pipelines.h"
#include "gpu/vk.h"
#include "render/passes.h"
#include "scene/model_instances.h"
#include "scene/models.h"
#include "scene/scene_gpu.h"
#include "scene/vegetation.h"
#include "sim/debris.h"
#include "world/terrain_query.h"

uint32_t numCharInst = 0;

static constexpr VkBuildAccelerationStructureFlagsKHR charBlasFlags =
    VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;

// ---------------------------------------------------------------- characters (character.h): GPU skinning + BLAS refit
// A character asset is uploaded once (materials, textures, indices, bind-pose vertices). Each character then owns,
// per skinned mesh, a vertex buffer that skin.comp writes every frame, two mesh slots (the skinned mesh, and the
// skinning source: bind verts | weights | matrices, see skin.comp) and a BLAS that is refit in place most frames
// and rebuilt every CHAR_REBUILD frames (refits degrade as limbs move far from where the BLAS was built).
static constexpr uint32_t CHAR_REBUILD = 30;

static std::map<std::string, CharAsset> charAssets;

std::vector<CharSpawn> charSpawns;  // processed in the upload window

std::string charWorldDir;

static CharAsset* charAsset(const std::string& name) {
    auto it = charAssets.find(name);
    if (it != charAssets.end()) return it->second.model ? &it->second : nullptr;
    CharAsset& a = charAssets[name];
    auto m = std::make_shared<Model>();
    m->path = assetFile(charWorldDir, name);
    std::string err;
    if (!loadGltf(m->path, m->data, err)) { fprintf(stderr, "character %s: %s\n", m->path.c_str(), err.c_str()); return nullptr; }
    if (m->data.skins.empty()) { fprintf(stderr, "character %s: no skin\n", m->path.c_str()); return nullptr; }
    // characters are skin (and fur over skin) unless the asset says otherwise: subsurface skin profile (sss.glsl)
    for (MaterialData& md : m->data.materials)
        if (md.sss == 0.f && md.metal < 0.5f && md.transmission < 0.5f) md.sss = 1.f;
    uploadModel(*m);
    if (!m->gpu) return nullptr;
    for (auto& md : m->data.meshes) {
        Buffer b;
        if (md.skin >= 0) {
            b = mkBuf(md.skinW.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            memcpy(b.map, md.skinW.data(), md.skinW.size() * 4);
        }
        a.skinW.push_back(b);
    }
    a.model = m;
    printf("character %s: %zu joints, %zu clips", name.c_str(), m->data.skins[0].parent.size(), m->data.clips.size());
    for (auto& c : m->data.clips) printf(" %s (%.1f s)", c.name.c_str(), c.duration);
    printf("\n");
    return &a;
}

void spawnCharacters() {
    for (const CharSpawn& sp : charSpawns) {
        CharAsset* a = charAsset(sp.asset);
        if (!a) continue;
        const ModelData& d = a->model->data;
        auto* g = (GpuModel*)a->model->gpu;
        float p[3] = {sp.pos[0], heightFn(sp.pos[0], sp.pos[2]), sp.pos[2]};
        character::Character* c = character::spawn(d, p, sp.yaw, sp.height);
        if (!c) continue;
        c->walkSpeed = sp.walk;
        c->wild = sp.wild;
        c->forward = sp.forward;
        if (!sp.clip.empty()) character::play(*c, character::findClip(d, sp.clip.c_str()), 0.f);
        auto* gc = new GpuChar;
        gc->asset = a;
        gc->mats = mkBuf(d.skins[c->skin].parent.size() * 48, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                         VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        for (uint32_t i = 0; i < d.meshes.size(); i++) {
            const MeshData& md = d.meshes[i];
            if (md.skin != c->skin || !g->meshes[i].blas || meshSlots.cap - meshSlots.used() < 2) continue;
            GpuCharMesh cm;
            cm.mesh = i;
            cm.nv = (uint32_t)(md.verts.size() / 8);
            cm.ntri = (uint32_t)(md.idx.size() / 3);
            cm.verts = mkBuf((VkDeviceSize)cm.nv * 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            VkDeviceAddress src[4];  // the model's own entry: bind verts, indices, triangle materials
            memcpy(src, (char*)meshInfoBuf.map + g->meshes[i].slot * 32, 32);
            cm.prevVerts = mkBuf((VkDeviceSize)cm.nv * 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                                 VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
            cm.slot = meshSlots.alloc();
            cm.srcSlot = meshSlots.alloc();
            cm.prevSlot = meshSlots.alloc();
            VkDeviceAddress dst[4] = {cm.verts.addr, src[1], src[2], 0}, skin[4] = {src[0], a->skinW[i].addr, gc->mats.addr, 0};
            VkDeviceAddress prv[4] = {cm.prevVerts.addr, src[1], src[2], 0};
            memcpy((char*)meshInfoBuf.map + cm.slot * 32, dst, 32);
            memcpy((char*)meshInfoBuf.map + cm.srcSlot * 32, skin, 32);
            memcpy((char*)meshInfoBuf.map + cm.prevSlot * 32, prv, 32);
            cm.geom = VkAccelerationStructureGeometryKHR{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
            cm.geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
            cm.geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
            auto& t = cm.geom.geometry.triangles;
            t.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
            t.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
            t.vertexData.deviceAddress = cm.verts.addr;
            t.vertexStride = 32;
            t.maxVertex = cm.nv - 1;
            t.indexType = VK_INDEX_TYPE_UINT32;
            t.indexData.deviceAddress = src[1];
            cm.blas = mkAccel(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, charBlasFlags, &cm.geom, cm.ntri);
            gc->meshes.push_back(cm);
        }
        c->user = gc;
    }
    charSpawns.clear();
}

void releaseCharacter(character::Character* c) {
    if (auto* gc = (GpuChar*)c->user) {
        for (auto& cm : gc->meshes) {
            meshSlots.release(cm.slot);
            meshSlots.release(cm.srcSlot);
            meshSlots.release(cm.prevSlot);
            Accel a = cm.blas;
            Buffer v = cm.verts, pv = cm.prevVerts;
            retireNext.push_back([a, v, pv]() mutable { pDestroyAS(dev, a.as, nullptr); freeBuf(a.buf); freeBuf(a.scratch); freeBuf(v); freeBuf(pv); });
        }
        Buffer m = gc->mats;
        retireNext.push_back([m]() mutable { freeBuf(m); });
        delete gc;
    }
    character::remove(c);
}

// skinning matrices -> GPU, skinned meshes -> TLAS (after the stumps); every frame
void writeCharacters() {
    numCharInst = 0;
    auto* inst = (VkAccelerationStructureInstanceKHR*)instBuf.map;
    uint32_t base = INST_MODELS + numModelInst + numRocks + numTrees + numPlants + numThrown + numSprayInst + numFallen;
    for (character::Character* c : character::all()) {
        auto* gc = (GpuChar*)c->user;
        if (!gc || c->skinMats.empty()) continue;
        memcpy(gc->mats.map, c->skinMats.data(), c->skinMats.size() * 4);
        Xf w = character::worldXf(*c);
        for (auto& cm : gc->meshes)
            if (numCharInst < (uint32_t)MAX_CHAR_INST) setInstance(inst[base + numCharInst++], 3 + cm.slot, cm.blas.addr, w.m);
    }
}

// render graph: skin every character mesh, then build / refit its BLAS (before the scene's TLAS build)
void recordSkinning(VkCommandBuffer cb) {
    bool any = false;
    for (character::Character* c : character::all())  // keep last frame's skinned vertices (motion vectors)
        if (auto* gc = (GpuChar*)c->user)
            for (auto& cm : gc->meshes) {
                VkBufferCopy r{0, 0, (VkDeviceSize)cm.nv * 32};
                vkCmdCopyBuffer(cb, cm.verts.b, cm.prevVerts.b, 1, &r);
                any = true;
            }
    if (any) memBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT | VK_ACCESS_TRANSFER_WRITE_BIT,
                        VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT);
    for (character::Character* c : character::all())
        if (auto* gc = (GpuChar*)c->user)
            for (auto& cm : gc->meshes) dispatch(cb, P_SKIN, (cm.nv + 63) / 64, 1, {cm.slot, cm.srcSlot, cm.nv, 0});
    memBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_READ_BIT);
    for (character::Character* c : character::all()) {
        auto* gc = (GpuChar*)c->user;
        if (!gc) continue;
        bool refit = gc->frames++ % CHAR_REBUILD != 0;
        for (auto& cm : gc->meshes) cmdBuild(cb, cm.blas, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, charBlasFlags, &cm.geom, cm.ntri, refit);
    }
    memBarrier(cb, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
               VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
}

namespace {
constexpr float STRIDE = 0.7f;   // m per step of a 1.75 m character
}  // namespace

void characterSteps(float dt) {
    struct Gait { float walked = 0.f; bool left = false; };
    static std::unordered_map<const character::Character*, Gait> gait;
    for (character::Character* c : character::all()) {
        if (c->mode != character::ANIMATED || c->walkSpeed <= 0.f) continue;
        Gait& g = gait[c];
        g.walked += c->walkSpeed * dt;
        if (g.walked < STRIDE * c->scale) continue;
        g.walked = 0.f;
        g.left = !g.left;
        float wl = waterLevelAt(c->pos[0], c->pos[2]), wade = wl - c->pos[1];
        if (wade > 0.02f) {  // wading: a splash with every stride
            float v[3] = {sinf(c->yaw) * c->walkSpeed, -1.5f, cosf(c->yaw) * c->walkSpeed};
            waterImpact(V3{c->pos[0], wl, c->pos[2]}, std::min(0.15f + wade, 0.8f), v, std::min(0.4f + wade, 1.f));
        } else footstep(V3{c->pos[0], c->pos[1], c->pos[2]}, c->yaw, g.left);  // a footprint, trodden grass
    }
}

#include "render/veg_gpu.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <vector>
#include <cstdio>
#include <unordered_map>
#include <meshoptimizer.h>
#include "app/settings.h"
#include "gpu/memory.h"
#include "gpu/context.h"
#include "gpu/descriptors.h"
#include "world/world.h"
#include "gpu/pipelines.h"
#include "render/culling_vegetation.h"
#include "render/occlusion.h"
#include "render/passes.h"
#include "scene/models.h"
#include "scene/rocks.h"
#include "scene/scene_gpu.h"
#include "destruction/thrown.h"
#include "scene/vegetation.h"

Buffer vegInstBuf, vegCmdBuf, rockVisBuf, meshletBuf, meshletDataBuf, vegBucketBuf;

namespace {
// per instance, 6 vec4 (vegcull.comp): 3 transform rows | slot, sway height, sway phase, bucket (+ tree flag) |
// bounding sphere | trees: local centre + yaw
constexpr uint32_t REC_FLOATS = 24;
constexpr uint32_t MAX_BUCKETS = MAX_TREE_VARIANTS + trees::PLANT_VARIANTS;
constexpr uint32_t CMD_UINTS = 5;  // VkDrawIndexedIndirectCommand
// vegCmdBuf: MAX_BUCKETS indexed draws | impostor draw (4) | rock draw (5) | MAX_BUCKETS mesh-task draws (3) | flags (4)
// | the same again as the reset template
constexpr VkDeviceSize IMP_CMD_OFF = MAX_BUCKETS * CMD_UINTS * 4, ROCK_CMD_OFF = IMP_CMD_OFF + 16, MESH_CMD_OFF = ROCK_CMD_OFF + 20,
                       FLAGS_OFF = MESH_CMD_OFF + MAX_BUCKETS * 12, CMD_BYTES = FLAGS_OFF + 16;
// meshlets (raster_model.task / .mesh): at most this many vertices and triangles each, triangles in mesh order
constexpr uint32_t ML_VERTS = 64, ML_TRIS = 64, ML_PER_TASK = 32;
uint32_t numInst = 0, nTreeBuckets = 0, nPlantBuckets = 0;
bool ready = false;
const void* meshletsFor[2] = {};  // the GpuModels the meshlets were built from (rebuilt when vegetation regenerates)

// Meshlets of every tree and plant variant (meshoptimizer: spatially tight clusters, so their bounding spheres
// cull well), each with its triangles' indices in the mesh (gl_PrimitiveID: the per-triangle materials stay as
// they are) and a bounding sphere. Bucket table: first meshlet, count, task groups per instance, flags offset.
void buildMeshlets() {
    std::vector<uint32_t> ml, data;  // ml: per meshlet 2 uvec4 (vertOffset, vertCount | triCount << 8, primOffset, triOffset), (centre, radius)
    std::vector<uint32_t> buckets;
    auto addMesh = [&](const MeshData& m) {
        uint32_t first = (uint32_t)ml.size() / 8;
        size_t ni = m.idx.size(), nv = m.verts.size() / 8;
        if (ni) {
            size_t cap = meshopt_buildMeshletsBound(ni, ML_VERTS, ML_TRIS);
            std::vector<meshopt_Meshlet> mls(cap);
            std::vector<unsigned int> mv(cap * ML_VERTS);
            std::vector<unsigned char> mt(cap * ML_TRIS * 3);
            size_t n = meshopt_buildMeshlets(mls.data(), mv.data(), mt.data(), m.idx.data(), ni, m.verts.data(), nv, 8 * sizeof(float), ML_VERTS, ML_TRIS, 0.f);
            // each meshlet triangle back to its index in the mesh: by its corners in order (rotations alike; the two
            // windings of a double-sided leaf stay apart), duplicates taken in turn
            std::unordered_multimap<uint64_t, uint32_t> tri;
            auto key = [](uint32_t a, uint32_t b, uint32_t c) {
                while (a > b || a > c) { uint32_t t = a; a = b; b = c; c = t; }  // smallest first, cyclic order kept
                return (uint64_t)a | ((uint64_t)b << 21) | ((uint64_t)c << 42);  // exact: meshes have < 2^21 vertices
            };
            for (uint32_t t = 0; t < ni / 3; t++) tri.emplace(key(m.idx[3 * t], m.idx[3 * t + 1], m.idx[3 * t + 2]), t);
            for (size_t i = 0; i < n; i++) {
                const meshopt_Meshlet& k = mls[i];
                meshopt_Bounds bd = meshopt_computeMeshletBounds(&mv[k.vertex_offset], &mt[k.triangle_offset], k.triangle_count, m.verts.data(), nv, 8 * sizeof(float));
                uint32_t vo = (uint32_t)data.size();
                data.insert(data.end(), mv.begin() + k.vertex_offset, mv.begin() + k.vertex_offset + k.vertex_count);
                uint32_t to = (uint32_t)data.size();
                for (uint32_t t = 0; t < k.triangle_count; t++) {
                    const unsigned char* c = &mt[k.triangle_offset + 3 * t];
                    data.push_back(c[0] | (c[1] << 8) | (c[2] << 16));
                }
                uint32_t po = (uint32_t)data.size();
                for (uint32_t t = 0; t < k.triangle_count; t++) {
                    const unsigned char* c = &mt[k.triangle_offset + 3 * t];
                    uint32_t a = mv[k.vertex_offset + c[0]], b = mv[k.vertex_offset + c[1]], cc = mv[k.vertex_offset + c[2]];
                    auto it = tri.find(key(a, b, cc));
                    uint32_t prim = 0;
                    if (it != tri.end()) { prim = it->second; tri.erase(it); }
                    data.push_back(prim);
                }
                float b4[4] = {bd.center[0], bd.center[1], bd.center[2], bd.radius};
                uint32_t bu[4];
                memcpy(bu, b4, 16);
                uint32_t rec[8] = {vo, k.vertex_count | (k.triangle_count << 8), po, to, bu[0], bu[1], bu[2], bu[3]};
                ml.insert(ml.end(), rec, rec + 8);
            }
        }
        uint32_t n = (uint32_t)ml.size() / 8 - first;
        buckets.insert(buckets.end(), {first, n, (n + ML_PER_TASK - 1) / ML_PER_TASK, (uint32_t)(FLAGS_OFF / 4)});
    };
    for (uint32_t v = 0; v < nTreeBuckets; v++) addMesh(treeModel.data.meshes[v]);
    for (uint32_t v = 0; v < nPlantBuckets; v++) addMesh(plantModel.data.meshes[v]);
    if (ml.empty()) return;
    vkDeviceWaitIdle(dev);  // (rare: startup and vegetation regeneration)
    for (Buffer* b : {&meshletBuf, &meshletDataBuf, &vegBucketBuf}) if (b->b) freeBuf(*b);
    meshletBuf = uploadBuf(ml.data(), ml.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    meshletDataBuf = uploadBuf(data.data(), data.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    vegBucketBuf = uploadBuf(buckets.data(), buckets.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    writeVegDescriptors();
    printf("meshlets: %zu for %u tree + %u plant variants\n", ml.size() / 8, nTreeBuckets, nPlantBuckets);
}
}  // namespace

void writeVegDescriptors() {
    VkDescriptorBufferInfo bi[6] = {{vegCmdBuf.b, 0, VK_WHOLE_SIZE}, {vegInstBuf.b, 0, VK_WHOLE_SIZE}, {rockVisBuf.b, 0, VK_WHOLE_SIZE},
                                    {meshletBuf.b, 0, VK_WHOLE_SIZE}, {meshletDataBuf.b, 0, VK_WHOLE_SIZE}, {vegBucketBuf.b, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet w[6];
    for (uint32_t i = 0; i < 6; i++) {
        w[i] = {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        w[i].dstSet = descSet;
        w[i].dstBinding = 107 + i;
        w[i].descriptorCount = 1;
        w[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        w[i].pBufferInfo = &bi[i];
    }
    vkUpdateDescriptorSets(dev, 6, w, 0, nullptr);
}

void createVegGpu() {
    VkMemoryPropertyFlags host = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    vegInstBuf = mkBuf((VkDeviceSize)VEG_SLOTS * REC_FLOATS * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, host, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    vegCmdBuf = mkBuf(CMD_BYTES * 2, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                      VK_BUFFER_USAGE_TRANSFER_DST_BIT, host, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    memset(vegCmdBuf.map, 0, CMD_BYTES * 2);
    rockVisBuf = mkBuf((MAX_ROCKS + MAX_THROWN) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    for (Buffer* b : {&meshletBuf, &meshletDataBuf, &vegBucketBuf})  // placeholders until the meshlets are built
        *b = mkBuf(16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

bool vegMeshShading() { return vegGpuOn() && S.meshShaders && hasMeshShaders && rasterModelMesh && meshletsFor[0]; }

bool vegGpuOn() { return S.gpuCull && S.rasterPrimary && treeModel.gpu; }

void uploadVegInstances() {
    auto* tg = (GpuModel*)treeModel.gpu;
    auto* pg = (GpuModel*)plantModel.gpu;
    if (!tg) return;
    nTreeBuckets = (uint32_t)numTreeVariants;
    nPlantBuckets = pg ? (uint32_t)trees::PLANT_VARIANTS : 0;
    uint32_t nb = nTreeBuckets + nPlantBuckets;
    // counting sort into buckets (one indirect draw each)
    uint32_t start[MAX_BUCKETS + 1] = {};
    for (const TreeInst& t : treeList) if (t.v < nTreeBuckets) start[t.v + 1]++;
    for (const PlantInst& p : plantList) if (p.v < nPlantBuckets) start[nTreeBuckets + p.v + 1]++;
    for (uint32_t k = 0; k < nb; k++) start[k + 1] += start[k];
    numInst = std::min(start[nb], VEG_SLOTS);
    if (hasMeshShaders && (meshletsFor[0] != tg || meshletsFor[1] != pg)) {
        buildMeshlets();
        meshletsFor[0] = tg;
        meshletsFor[1] = pg;
    }
    uint32_t base[MAX_BUCKETS];
    memcpy(base, start, sizeof base);
    float* rec = (float*)vegInstBuf.map;
    auto put = [&](uint32_t k, const float r[REC_FLOATS]) {
        uint32_t i = start[k]++;
        if (i < VEG_SLOTS) memcpy(rec + (size_t)i * REC_FLOATS, r, REC_FLOATS * 4);
    };
    for (const TreeInst& t : treeList) {
        if (t.v >= nTreeBuckets) continue;
        uint32_t v = t.v;
        float c = cosf(t.yaw), sn = sinf(t.yaw), sc = t.scale;
        V3 lc{treeC[v][0], treeC[v][1], treeC[v][2]};
        V3 C = t.p + V3{c * lc.x + sn * lc.z, lc.y, -sn * lc.x + c * lc.z} * sc;
        uint32_t slot = tg->meshes[v].slot, kb = v | 0x80000000u;
        float r[REC_FLOATS] = {c * sc, 0, sn * sc, t.p.x, 0, sc, 0, t.p.y, -sn * sc, 0, c * sc, t.p.z,
                               0, treeTop[v] * sc, swayPhase(t.p.x, t.p.z), 0,
                               C.x, C.y, C.z, treeR[v] * sc,
                               lc.x, lc.y, lc.z, t.yaw};
        memcpy(&r[12], &slot, 4);
        memcpy(&r[15], &kb, 4);
        put(v, r);
    }
    for (const PlantInst& p : plantList) {
        if (p.v >= nPlantBuckets) continue;
        uint32_t v = p.v, k = nTreeBuckets + v;
        const float* m = p.m;
        V3 C{m[0] * plantC[v][0] + m[1] * plantC[v][1] + m[2] * plantC[v][2] + m[3], m[4] * plantC[v][0] + m[5] * plantC[v][1] + m[6] * plantC[v][2] + m[7],
             m[8] * plantC[v][0] + m[9] * plantC[v][1] + m[10] * plantC[v][2] + m[11]};
        int flex = plantFlex((int)v);
        uint32_t slot = pg->meshes[v].slot;
        float r[REC_FLOATS] = {};
        memcpy(r, m, 48);
        memcpy(&r[12], &slot, 4);
        r[13] = flex >= 0 ? plantTop[v] * m[5] : 0.f;
        r[14] = flex >= 0 ? (float)flex + swayPhase(m[3], m[11]) : 0.f;
        memcpy(&r[15], &k, 4);
        r[16] = C.x; r[17] = C.y; r[18] = C.z; r[19] = plantR[v] * m[5];
        put(k, r);
    }
    // the reset template (second half): instance counts 0, each bucket's first slot
    uint32_t* tpl = (uint32_t*)((char*)vegCmdBuf.map + CMD_BYTES);
    memset(tpl, 0, CMD_BYTES);
    for (uint32_t k = 0; k < nb; k++) {
        const GpuMesh& gm = k < nTreeBuckets ? tg->meshes[k] : pg->meshes[k - nTreeBuckets];
        uint32_t* c = tpl + k * CMD_UINTS;
        c[0] = gm.indexCount; c[1] = 0; c[2] = gm.firstIndex; c[3] = 0; c[4] = VEG_XF_BASE + base[k];
    }
    for (uint32_t k = 0; k < nb; k++) {  // mesh-task draws: x grows by the bucket's task groups per visible instance
        uint32_t* c = tpl + MESH_CMD_OFF / 4 + k * 3;
        c[0] = 0; c[1] = 1; c[2] = 1;
    }
    uint32_t* ic = (uint32_t*)((char*)tpl + IMP_CMD_OFF);
    ic[0] = 6; ic[1] = 0; ic[2] = 0; ic[3] = 0;
    ready = true;
}

void readVegStats() {
    if (!vegGpuOn() || !ready) return;
    const uint32_t* c = (const uint32_t*)vegCmdBuf.map;
    numTreeNear = numPlantsDrawn = 0;
    for (uint32_t k = 0; k < nTreeBuckets; k++) numTreeNear += c[k * CMD_UINTS + 1];
    for (uint32_t k = nTreeBuckets; k < nTreeBuckets + nPlantBuckets; k++) numPlantsDrawn += c[k * CMD_UINTS + 1];
    numImpostors = c[IMP_CMD_OFF / 4 + 1];
}

void recordVegCull(VkCommandBuffer cb) {
    // the rock draw's template: its index count can change (rock LOD), so it's written every frame
    uint32_t* rc = (uint32_t*)((char*)vegCmdBuf.map + CMD_BYTES + ROCK_CMD_OFF);
    rc[0] = rockTris * 3; rc[1] = 0; rc[2] = 0; rc[3] = 0; rc[4] = 0;
    uint32_t* fl = (uint32_t*)((char*)vegCmdBuf.map + CMD_BYTES + FLAGS_OFF);  // read by raster_model.task
    fl[0] = occlusionValid() ? 1u : 0u; fl[1] = (uint32_t)(MESH_CMD_OFF / 4); fl[2] = vegMeshShading() ? 1u : 0u; fl[3] = 0;
    VkBufferCopy cp{CMD_BYTES, 0, CMD_BYTES};
    vkCmdCopyBuffer(cb, vegCmdBuf.b, vegCmdBuf.b, 1, &cp);
    memBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT, VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_INDIRECT_COMMAND_READ_BIT,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    float d0 = S.impostorDist;
    uint32_t d0b;
    memcpy(&d0b, &d0, 4);
    uint32_t nr = numRocks + numThrownRock;
    if (nr) dispatch(cb, P_ROCKCULL, (nr + 63) / 64, 1, {nr, (uint32_t)(ROCK_CMD_OFF / 4), occlusionValid() ? 1u : 0u, S.gpuCull ? 1u : 0u});
    if (vegGpuOn() && ready && numInst) dispatch(cb, P_VEGCULL, (numInst + 63) / 64, 1, {numInst, d0b, occlusionValid() ? 1u : 0u, (uint32_t)(IMP_CMD_OFF / 4)});
    memBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT,
               VK_PIPELINE_STAGE_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_VERTEX_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
               VK_ACCESS_INDIRECT_COMMAND_READ_BIT | VK_ACCESS_SHADER_READ_BIT);
}

void drawVegModels(VkCommandBuffer cb) {
    if (!vegGpuOn() || !ready) return;
    if (vegMeshShading()) {  // one indirect mesh-task draw per bucket (gl_DrawID = bucket)
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, rasterModelMesh);
        pCmdDrawMeshTasksIndirect(cb, vegCmdBuf.b, MESH_CMD_OFF, nTreeBuckets + nPlantBuckets, 12);
        return;
    }
    auto* tg = (GpuModel*)treeModel.gpu;
    auto* pg = (GpuModel*)plantModel.gpu;
    vkCmdBindIndexBuffer(cb, tg->geom.b, 0, VK_INDEX_TYPE_UINT32);
    if (nTreeBuckets) vkCmdDrawIndexedIndirect(cb, vegCmdBuf.b, 0, nTreeBuckets, CMD_UINTS * 4);
    if (pg && nPlantBuckets) {
        vkCmdBindIndexBuffer(cb, pg->geom.b, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexedIndirect(cb, vegCmdBuf.b, nTreeBuckets * CMD_UINTS * 4, nPlantBuckets, CMD_UINTS * 4);
    }
}

void drawRocks(VkCommandBuffer cb) { vkCmdDrawIndexedIndirect(cb, vegCmdBuf.b, ROCK_CMD_OFF, 1, 20); }

void drawVegImpostors(VkCommandBuffer cb) {
    if (!vegGpuOn() || !ready) return;
    vkCmdDrawIndirect(cb, vegCmdBuf.b, IMP_CMD_OFF, 1, 16);
}

#include "scene/model_instances.h"
#include <vector>
#include <cstdio>
#include <cstdint>
#include <utility>
#include <vulkan/vulkan.h>
#include "world/world.h"
#include "core/limits.h"
#include "gpu/accel.h"
#include "scene/models.h"
#include "scene/scene_gpu.h"

std::vector<uint32_t> emisItems;                  // drawItems with emissive triangles

std::vector<DrawItem> drawItems;

std::vector<float> aabb[6];

std::vector<uint8_t> visibleItems;

std::vector<DrawRun> drawRuns;

uint32_t numModelInst = 0, numModelVisible = 0;

// Streamed placements -> TLAS instances (static, rewritten only on change) + SoA bounds for culling.
void rebuildModelInstances(const std::vector<std::pair<const MeshData*, DrawItem>>& items) {
    drawItems.clear();
    for (auto& a : aabb) a.clear();
    auto* inst = (VkAccelerationStructureInstanceKHR*)instBuf.map;
    for (auto& [md, it] : items) {
        if (drawItems.size() >= MAX_MODEL_INST) { fprintf(stderr, "stream: more than %u model instances\n", MAX_MODEL_INST); break; }
        const GpuMesh& gm = it.m->meshes[it.mesh];
        setInstance(inst[INST_MODELS + drawItems.size()], 3 + gm.slot, gm.blasAddr, it.xf.m);  // 3 = MAT_MODEL
        float mn[3], mx[3];
        xfAabb(it.xf, md->bmin, md->bmax, mn, mx);
        for (int k = 0; k < 3; k++) { aabb[k].push_back(mn[k]); aabb[3 + k].push_back(mx[k]); }
        drawItems.push_back(it);
    }
    numModelInst = (uint32_t)drawItems.size();
    visibleItems.resize(numModelInst);
    emisItems.clear();
    for (uint32_t i = 0; i < numModelInst; i++)
        if (!drawItems[i].m->meshes[drawItems[i].mesh].emis.empty()) emisItems.push_back(i);
}

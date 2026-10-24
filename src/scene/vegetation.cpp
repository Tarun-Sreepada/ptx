#include "scene/vegetation.h"
#include "app/settings_io.h"
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstring>
#include <thread>
#include <cstdint>
#include <vulkan/vulkan.h>
#include "core/jobs.h"
#include "core/prof.h"
#include "physics/physics.h"
#include "app/settings.h"
#include "vegetation/trees.h"
#include "world/texcomp.h"
#include "world/world.h"
#include "gpu/accel.h"
#include "gpu/context.h"
#include "gpu/memory.h"
#include "gpu/pipelines.h"
#include "gpu/vk.h"
#include "scene/forest.h"
#include "scene/models.h"
#include "scene/scene_gpu.h"

Model treeModel;

int numTreeVariants = trees::VARIANTS;

std::vector<ModelData> forestAssets;              // variant trees::VARIANTS + i, merged into treeModel by initTrees

float treeWeight[MAX_TREE_VARIANTS], treeWeightSum = 0.f, forestSpacing = 0.8f;

float treeFoot[MAX_TREE_VARIANTS];                // footprint radius (local, m) about the trunk; 0 = never blocks

float treeC[MAX_TREE_VARIANTS][3], treeR[MAX_TREE_VARIANTS];
float treeTop[MAX_TREE_VARIANTS];

std::vector<TreeInst> treeList;

uint32_t numTrees = 0, numTreeNear = 0, numImpostors = 0;

bool placeDirty = true;  // re-run rock / tree placement next frame (world streamed in, zones changed)

Buffer impBuf;                          // impostor instances (impostor.glsl), also the bake's parameters

Image impAlbedoImg, impNormalImg;       // octahedral atlas, one layer per variant

void* trunkShape[MAX_TREE_VARIANTS];

std::unordered_map<uint64_t, uint32_t> treeBodies;  // trunk colliders near the player, keyed by grid cell

Model plantModel;

float plantC[trees::PLANT_VARIANTS][3], plantR[trees::PLANT_VARIANTS];
float plantTop[trees::PLANT_VARIANTS];

std::vector<PlantInst> plantList;

uint32_t numPlants = 0, numPlantsDrawn = 0;

static void uploadVegetation();

void generateVegetation() {
    PROF_FUNC();
    trees::defaults();
    applyVegetationSettings();  // the sliders as they were left
    trees::seasonDay = S.dayCycle ? trees::seasonBucketDay(S.dayOfYear) : -1.f;  // leaves as the season has them
    trees::build(treeModel.data);
    trees::buildPlants(plantModel.data);
}

void initTrees() {
    treeModel.path = "<procedural trees>";
    treeModel.state = 1;
    // imported forest trees after the procedural variants (mesh v = variant v), their materials / textures appended
    for (auto& a : forestAssets) {
        uint32_t matBase = (uint32_t)treeModel.data.materials.size(), texBase = (uint32_t)treeModel.data.textures.size();
        for (MaterialData mt : a.materials) {
            for (int* t : {&mt.tex, &mt.normalTex, &mt.ormTex, &mt.occTex, &mt.emisTex}) if (*t >= 0) *t += (int)texBase;
            treeModel.data.materials.push_back(mt);
        }
        treeModel.data.textures.insert(treeModel.data.textures.end(), a.textures.begin(), a.textures.end());
        treeModel.data.meshes.push_back(mergeAsset(a, matBase));
    }
    forestAssets.clear();
    forestAssets.shrink_to_fit();
    uploadVegetation();
}

// Trees (procedural variants + merged forest assets in treeModel.data) and plants -> GPU, trunk colliders,
// impostor bake. Submits and waits on uploadCmd, so it runs at startup or in the upload window with the GPU idle.
static void uploadVegetation() {
    for (int v = 0; v < numTreeVariants; v++) {  // mesh v = variant v
        const MeshData& m = treeModel.data.meshes[v];
        trees::bounds(m, treeC[v], treeR[v]);
        treeTop[v] = std::max(m.bmax[1], 0.f);
        treeFoot[v] = v < trees::VARIANTS ? 0.f : std::max({-m.bmin[0], m.bmax[0], -m.bmin[2], m.bmax[2]});
    }
    {   // the forest assets carry 4k textures: BC7 them in parallel (cached in texcache/ after the first run)
        PROF_ZONE("compress tree textures");
        auto& tex = treeModel.data.textures;
        jobs::parallelFor((uint32_t)tex.size(), 1, [&](uint32_t a, uint32_t b) { for (uint32_t i = a; i < b; i++) texcomp::prepare(tex[i]); });
    }
    uploadModel(treeModel);
    plantModel.path = "<procedural plants>";
    plantModel.state = 1;
    for (int v = 0; v < trees::PLANT_VARIANTS; v++) {
        trees::bounds(plantModel.data.meshes[v], plantC[v], plantR[v]);
        plantTop[v] = std::max(plantModel.data.meshes[v].bmax[1], 0.f);
    }
    uploadModel(plantModel);
    VK_CHECK(vkEndCommandBuffer(uploadCmd));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &uploadCmd;
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    VK_CHECK(vkQueueWaitIdle(queue));
    uploadsRecorded = false;
    for (auto& f : retireNext) f();
    retireNext.clear();
    frameUploadBytes = 0;

    std::vector<float> cv;
    std::vector<uint32_t> ci;
    for (int v = 0; v < numTreeVariants; v++) {
        if (v < trees::VARIANTS) trees::trunkCollider(treeModel.data, v, cv, ci);
        else assetTrunk(treeModel.data.meshes[v], cv, ci);
        trunkShape[v] = physics::meshShape(cv.data(), 3, (uint32_t)cv.size() / 3, ci.data(), (uint32_t)ci.size() / 3);
    }

    auto* g = (GpuModel*)treeModel.gpu;
    auto* inst = (VkAccelerationStructureInstanceKHR*)instBuf.map;
    float* ib = (float*)impBuf.map;
    for (int v = 0; v < numTreeVariants; v++) {
        float y = -5000.f - 100.f * v;
        float m[3][4] = {{1, 0, 0, 0}, {0, 1, 0, y}, {0, 0, 1, 0}};
        setInstance(inst[v], 3 + g->meshes[v].slot, g->meshes[v].blasAddr, m);
        float e[8] = {0, y, 0, 0, treeC[v][0], treeC[v][1], treeC[v][2], treeR[v]};
        memcpy(ib + v * 8, e, 32);
    }
    oneShot([](VkCommandBuffer cb) {
        cmdBuild(cb, tlas, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR, &tlasGeom,
                 (uint32_t)numTreeVariants, false);
        VkImageMemoryBarrier b[2];
        for (int k = 0; k < 2; k++) {
            b[k] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b[k].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b[k].newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b[k].srcQueueFamilyIndex = b[k].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b[k].image = k ? impNormalImg.img : impAlbedoImg.img;
            b[k].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, (uint32_t)numTreeVariants};
            b[k].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
        }
        VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
        mb.srcAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
        mb.dstAccessMask = VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT,
                             VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 1, &mb, 0, nullptr, 2, b);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &descSet, 0, nullptr);
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[P_IMPOSTOR_BAKE]);
        vkCmdDispatch(cb, 1024 / 8, 1024 / 8, (uint32_t)numTreeVariants);
        memBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT,
                   VK_ACCESS_SHADER_READ_BIT);
    });
    const float id[3][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};
    setInstance(inst[0], 0, terrainAS.addr, id);
    setInstance(inst[1], 1, grassAS.addr, id);
}

// Vegetation sliders changed: regrow the procedural trees and plants in place (imported forest assets kept), then
// re-upload, re-bake impostors and rebuild colliders. Needs the GPU idle; done in the upload window.
bool vegDirty = false;

int vegRegenFrame = -1;

void regenVegetation() {
    PROF_FUNC();
    vkDeviceWaitIdle(dev);
    std::vector<uint32_t> ids;
    for (auto& [k, id] : treeBodies) ids.push_back(id);
    if (!ids.empty()) physics::removeBodies(ids.data(), ids.size());
    treeBodies.clear();
    for (int v = 0; v < numTreeVariants; v++)
        if (trunkShape[v]) { physics::releaseShape(trunkShape[v]); trunkShape[v] = nullptr; }
    releaseModel(treeModel);
    releaseModel(plantModel);
    ModelData fresh;
    trees::build(fresh);
    trees::buildPlants(plantModel.data);
    // same procedural texture / material / mesh counts every time, so the assets' offsets after them still hold
    auto& d = treeModel.data;
    std::copy(fresh.textures.begin(), fresh.textures.end(), d.textures.begin());
    std::copy(fresh.materials.begin(), fresh.materials.end(), d.materials.begin());
    std::copy(fresh.meshes.begin(), fresh.meshes.end(), d.meshes.begin());
    uploadVegetation();
    placeDirty = true;
}

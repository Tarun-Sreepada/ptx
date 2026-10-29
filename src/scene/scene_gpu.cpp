#include "scene/scene_gpu.h"
#include "render/veg_gpu.h"

#include <cstring>
#include <random>
#include <vector>

#include "core/jobs.h"
#include "core/limits.h"
#include "core/math.h"
#include "core/prof.h"
#include "destruction/thrown.h"
#include "destruction/tree_damage.h"
#include "gpu/accel.h"
#include "gpu/context.h"
#include "gpu/memory.h"
#include "gpu/textures.h"
#include "render/bake.h"
#include "render/crater_bins.h"
#include "render/frame_ubo.h"
#include "render/ground_vt.h"
#include "render/lights.h"
#include "render/lighttree.h"
#include "render/occlusion.h"
#include "render/targets.h"
#include "scene/characters.h"
#include "scene/grass.h"
#include "scene/model_instances.h"
#include "scene/models.h"
#include "scene/rocks.h"
#include "scene/vegetation.h"
#include "sim/debris.h"
#include "sim/world_sim.h"
#include "world/worldmap.h"

Buffer terrainV, terrainI, grassV, grassI, bladeBuf, lightBuf, uboBuf, instBuf, grassCounter, countReadback, rockV, rockI, rockXf;
Image skyImg;
Image windImg[2];
Accel terrainAS, grassAS, rockAS[16], tlas;  // rockAS: one per variant (ROCK_VARIANTS)
VkAccelerationStructureGeometryKHR grassGeom, terrainGeom, rockGeom[16], tlasGeom;
uint32_t terrainTris, rockTris, rockVerts, numRocks = 0;

namespace {
constexpr uint32_t WIND_MAP = 256;              // wind map cells per side (wind.glsl)
constexpr uint32_t VT_REQUESTS = 4096;           // ground virtual texture tile requests per frame (capacity)
constexpr uint32_t IMPOSTOR_ATLAS = 1024;        // octahedral impostor atlas texels per side (impostor.glsl IMP_ATLAS)
constexpr uint32_t EXPOSURE_BYTES = 128 * 4 + 16; // histogram bins + adapted exposure (exposure.comp)
constexpr float MAX_ANISOTROPY = 8.f;
constexpr uint32_t GRASS_VERTS = 7, GRASS_TRIS = 5;  // per blade slot (grass.comp)

const VkBufferUsageFlags GEOM_USAGE = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                      VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
const VkMemoryPropertyFlags HOST = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

// prepareSceneGeometry -> buildScene
std::vector<uint32_t> grassIdx, rockIdx;
std::vector<float> rockVertsAll;

void createTerrainAndGrass() {
    // terrain clipmap: vertices and indices are both written on the GPU (terrain.comp), sized for every level
    terrainTris = TERRAIN_LEVEL_TRIS;  // one level until the first frame sets the count
    terrainI = mkBuf((VkDeviceSize)TERRAIN_LEVELS_MAX * TERRAIN_LEVEL_TRIS * 3 * 4, GEOM_USAGE, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    terrainV = mkBuf((VkDeviceSize)TERRAIN_LEVELS_MAX * TERRAIN_LEVEL_VERTS * 24, GEOM_USAGE, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    grassI = uploadBuf(grassIdx.data(), grassIdx.size() * 4, GEOM_USAGE);
    grassV = mkBuf((VkDeviceSize)maxBlades * GRASS_VERTS * 3 * 4, GEOM_USAGE, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    bladeBuf = mkBuf((VkDeviceSize)maxBlades * 12,  // seed + cell xy per slot (grass.comp)
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    grassCounter = mkBuf(64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    countReadback = mkBuf(16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST);
    std::vector<uint32_t>().swap(grassIdx);
}

void createRocks() {
    rockVerts = (uint32_t)rockVertsAll.size() / 6 / ROCK_VARIANTS;
    rockPts = rockVertsAll;
    rockTriIdx = rockIdx;
    rockTris = (uint32_t)rockIdx.size() / 3;
    rockV = uploadBuf(rockVertsAll.data(), rockVertsAll.size() * 4, GEOM_USAGE);
    rockI = uploadBuf(rockIdx.data(), rockIdx.size() * 4, GEOM_USAGE);
    // 4 rows per rock (3x4 transform + variant base vertex); the second half holds last frame's transforms
    rockXf = mkBuf((MAX_ROCKS + MAX_THROWN) * 128, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

// images every pass reads / writes in GENERAL layout; cleared to their resting value
struct InitImage { Image* im; uint32_t layers; VkClearColorValue clear; bool doClear; };

void createImages(std::vector<InitImage>& init) {
    skyImg = mkImage(SKY_W, SKY_H, VK_FORMAT_R32G32B32A32_SFLOAT);  // sky-view LUT
    init.push_back({&skyImg, 1, {}, false});
    for (auto& w : windImg) {  // wind map around the camera: velocity + pressure
        w = mkImage(WIND_MAP, WIND_MAP, VK_FORMAT_R16G16B16A16_SFLOAT);
        init.push_back({&w, 1, {{0, 0, 0, 0}}, true});
    }
    // world map + simulation grid
    const int N = worldmap::N;
    mapHImg = mkImage(N, N, VK_FORMAT_R32_SFLOAT);
    mapPaintImg = mkImage(N, N, VK_FORMAT_R8G8B8A8_UNORM);
    mapWaterImg = mkImage(N / 2, N / 2, VK_FORMAT_R32G32B32A32_SFLOAT);
    simImg = mkImage(N, N, VK_FORMAT_R16G16B16A16_SFLOAT);
    fluxWImg = mkImage(N, N, VK_FORMAT_R16G16B16A16_SFLOAT);
    fluxSImg = mkImage(N, N, VK_FORMAT_R16G16B16A16_SFLOAT);
    simGroundImg = mkImage(N, N, VK_FORMAT_R32_SFLOAT);
    for (Image* im : {&mapHImg, &simImg, &fluxWImg, &fluxSImg, &simGroundImg}) init.push_back({im, 1, {{0, 0, 0, 0}}, true});
    init.push_back({&mapPaintImg, 1, {{0.5f, 0.5f, 0, 0}}, true});  // 1x grass and trees
    init.push_back({&mapWaterImg, 1, {{-1000.f, 0, 0, 0}}, true});  // no drawn water
    // ground virtual texture clipmap
    vtImgA = mkImageArray(VT_N, VT_N, VT_L, VK_FORMAT_R8G8B8A8_UNORM);
    vtImgB = mkImageArray(VT_N, VT_N, VT_L, VK_FORMAT_R8G8B8A8_UNORM);
    vtImgH = mkImageArray(VT_N, VT_N, VT_L, VK_FORMAT_R16_UNORM);
    for (Image* im : {&vtImgA, &vtImgB, &vtImgH}) init.push_back({im, VT_L, {}, false});
}

void createSmallBuffers() {
    vtTileBuf = mkBuf(VT_REQUESTS * sizeof(VtTileReq), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST);
    craterBinBuf = mkBuf((2 * CB_BINS + CB_LIST) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST);
    memset(craterBinBuf.map, 0, (2 * CB_BINS + CB_LIST) * 4);
    simRB = mkBuf(SIM_RB * SIM_RB * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST);
    memset(simRB.map, 0, SIM_RB * SIM_RB * 8);
    exposureBuf = mkBuf(EXPOSURE_BYTES, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST);
    memset(exposureBuf.map, 0, EXPOSURE_BYTES);
    createOcclusionBuffer();
    createVegGpu();
}

// light orbs drift in a square tile (S.orbRange wide) that wraps around the camera; positions in tile units
void createLights() {
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    const V3 palette[6] = {{1, 0.25f, 0.6f}, {0.2f, 0.8f, 1}, {1, 0.6f, 0.15f}, {0.4f, 1, 0.3f}, {0.6f, 0.3f, 1}, {1, 0.9f, 0.7f}};
    for (int i = 0; i < MAX_ORBS; i++)
        lightBase.push_back({{U(rng) - 0.5f, 0.6f + U(rng) * 1.4f, U(rng) - 0.5f}, U(rng) * 100.f, palette[i % 6], 0.6f + 0.8f * U(rng)});
    lightBuf = mkBuf(MAX_LIGHTS * 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    uboBuf = mkBuf(sizeof(UBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT, HOST, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    triLightBuf = mkBuf(MAX_TRI_LIGHTS * 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    lightAliasBuf = mkBuf((MAX_LIGHTS + MAX_TRI_LIGHTS) * sizeof(lighttree::AliasEntry), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST,
                          VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    lightTreeBuf = mkBuf((MAX_LIGHTS + MAX_TRI_LIGHTS) * 2 * sizeof(lighttree::Node), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST,
                         VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    std::vector<float> bn;
    lighttree::blueNoise(BLUE_NOISE, bn);
    blueNoiseBuf = mkBuf(bn.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    memcpy(blueNoiseBuf.map, bn.data(), bn.size() * 4);
}

// streamed model tables (written from the CPU only for slots no in-flight frame uses), bake volumes, GI cache
void createModelTables() {
    meshInfoBuf = mkBuf(MAX_MESHES * 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    memset(meshInfoBuf.map, 0, MAX_MESHES * 32);
    materialBuf = mkBuf(MAX_MATERIALS * sizeof(GpuMat), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    // second half: each instance's previous-frame transform (motion vectors)
    modelXfBuf = mkBuf((VkDeviceSize)XF_SLOTS * 128, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    zoneBuf = mkBuf(MAX_ZONES * 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    memset(zoneBuf.map, 0, MAX_ZONES * 32);
    volBuf = mkBuf(MAX_VOLUMES * 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    memset(volBuf.map, 0, MAX_VOLUMES * 64);
    probeBuf = mkBuf((VkDeviceSize)MAX_PROBES * 2 * 64,
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    probeTmp = mkBuf((VkDeviceSize)MAX_PROBES * 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    radCacheBuf = mkBuf((VkDeviceSize)RC_ENTRIES * RC_STRIDE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    impBuf = mkBuf(MAX_TREES * 48, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, HOST, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
}

// impostor atlas: one layer per tree variant, albedo and normal + depth
void createImpostorAtlas() {
    for (int k = 0; k < 2; k++) {
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = k == 0 ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_R16G16B16A16_SFLOAT;
        ci.extent = {IMPOSTOR_ATLAS, IMPOSTOR_ATLAS, 1};
        ci.mipLevels = 1;
        ci.arrayLayers = (uint32_t)numTreeVariants;
        ci.samples = VK_SAMPLE_COUNT_1_BIT;
        ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;
        Image im = mkImageCI(ci, VK_IMAGE_ASPECT_COLOR_BIT);
        vkDestroyImageView(dev, im.view, nullptr);
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = im.img;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
        vi.format = ci.format;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, (uint32_t)numTreeVariants};
        VK_CHECK(vkCreateImageView(dev, &vi, nullptr, &im.view));
        (k == 0 ? impAlbedoImg : impNormalImg) = im;
    }
}

void createSamplers() {
    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.anisotropyEnable = VK_TRUE;
    sci.maxAnisotropy = MAX_ANISOTROPY;
    sci.maxLod = VK_LOD_CLAMP_NONE;
    VK_CHECK(vkCreateSampler(dev, &sci, nullptr, &texSampler));
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.anisotropyEnable = VK_FALSE;
    VK_CHECK(vkCreateSampler(dev, &sci, nullptr, &postSampler));
}
}  // namespace

void prepareSceneGeometry() {
    PROF_FUNC();
    // grass: per-frame GPU generated; fixed 7-vertex / 5-triangle topology per slot
    const uint32_t pat[GRASS_TRIS * 3] = {0, 1, 2, 1, 3, 2, 2, 3, 4, 3, 5, 4, 4, 5, 6};
    grassIdx.resize((size_t)maxBlades * GRASS_TRIS * 3);
    jobs::parallelFor(maxBlades, 16384, [&](uint32_t b0, uint32_t b1) {
        for (uint32_t b = b0; b < b1; b++)
            for (uint32_t k = 0; k < GRASS_TRIS * 3; k++) grassIdx[(size_t)b * GRASS_TRIS * 3 + k] = b * GRASS_VERTS + pat[k];
    });
    // rocks: every variant on its own worker, same topology (the index list comes out identical)
    std::vector<std::vector<float>> verts(ROCK_VARIANTS);
    std::vector<std::vector<uint32_t>> idx(ROCK_VARIANTS);
    jobs::parallelFor(ROCK_VARIANTS, 1, [&](uint32_t a, uint32_t b) {
        for (uint32_t v = a; v < b; v++) buildRockMesh(v, verts[v], idx[v]);
    });
    rockVertsAll.clear();
    for (auto& v : verts) rockVertsAll.insert(rockVertsAll.end(), v.begin(), v.end());
    rockIdx = std::move(idx[0]);
}

void buildScene() {
    PROF_FUNC();
    createTerrainAndGrass();
    createRocks();
    std::vector<InitImage> init;
    createImages(init);
    createSmallBuffers();
    createLights();
    createModelTables();
    createImpostorAtlas();
    createSamplers();
    // one submit for every image's first layout + clear, and the radiance cache reset
    oneShot([&](VkCommandBuffer cb) {
        std::vector<VkImageMemoryBarrier> bs;
        for (const InitImage& ii : init) {
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = ii.im->img;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, ii.layers};
            b.dstAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            bs.push_back(b);
        }
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0,
                             nullptr, 0, nullptr, (uint32_t)bs.size(), bs.data());
        for (const InitImage& ii : init) {
            if (!ii.doClear) continue;
            VkImageSubresourceRange rr{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, ii.layers};
            vkCmdClearColorImage(cb, ii.im->img, VK_IMAGE_LAYOUT_GENERAL, &ii.clear, 1, &rr);
        }
        vkCmdFillBuffer(cb, radCacheBuf.b, 0, VK_WHOLE_SIZE, 0);
        vkCmdFillBuffer(cb, terrainI.b, 0, VK_WHOLE_SIZE, 0);
        vkCmdFillBuffer(cb, terrainV.b, 0, VK_WHOLE_SIZE, 0);
        memBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
    });
}

void createAccels() {
    PROF_FUNC();
    terrainGeom = triGeom(terrainV, 24, TERRAIN_LEVELS_MAX * TERRAIN_LEVEL_VERTS, terrainI);
    grassGeom = triGeom(grassV, 12, maxBlades * GRASS_VERTS, grassI);
    for (int v = 0; v < ROCK_VARIANTS; v++) {
        rockGeom[v] = triGeom(rockV, 24, rockVerts, rockI);
        rockGeom[v].geometry.triangles.vertexData.deviceAddress = rockV.addr + (VkDeviceSize)v * rockVerts * 24;
    }
    const VkBuildAccelerationStructureFlagsKHR TRACE = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    terrainAS = mkAccel(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, TRACE, &terrainGeom, TERRAIN_LEVELS_MAX * TERRAIN_LEVEL_TRIS);
    grassAS = mkAccel(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, grassFlags, &grassGeom, maxBlades * GRASS_TRIS);
    for (int v = 0; v < ROCK_VARIANTS; v++) rockAS[v] = mkAccel(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, TRACE, &rockGeom[v], rockTris);
    oneShot([&](VkCommandBuffer cb) {
        for (int v = 0; v < ROCK_VARIANTS; v++) cmdBuild(cb, rockAS[v], VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, TRACE, &rockGeom[v], rockTris, false);
    });

    instBuf = mkBuf(TLAS_CAPACITY * sizeof(VkAccelerationStructureInstanceKHR), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR, HOST);
    auto* inst = (VkAccelerationStructureInstanceKHR*)instBuf.map;
    const float id[3][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};
    setInstance(inst[0], 0, terrainAS.addr, id);
    setInstance(inst[1], 1, grassAS.addr, id);

    tlasGeom = VkAccelerationStructureGeometryKHR{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlasGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    tlasGeom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tlasGeom.geometry.instances.data.deviceAddress = instBuf.addr;
    tlas = mkAccel(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, TRACE, &tlasGeom, TLAS_CAPACITY);
}

// in three steps, each its own render graph pass (so each is timed): the terrain BLAS (when its mesh changed), the
// grass BLAS, then the TLAS over everything
void cmdBuildTerrainBLAS(VkCommandBuffer cb) {
    const VkBuildAccelerationStructureFlagsKHR TRACE = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    cmdBuild(cb, terrainAS, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, TRACE, &terrainGeom, terrainTris, false);
}

void cmdBuildGrassBLAS(VkCommandBuffer cb) {
    cmdBuild(cb, grassAS, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, grassFlags, &grassGeom, bladeCap * GRASS_TRIS, grassRefit);
    grassBuilt = true;
}

void cmdBuildTLAS(VkCommandBuffer cb) {
    const VkBuildAccelerationStructureFlagsKHR TRACE = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
    uint32_t instances = INST_MODELS + numModelInst + numRocks + numTrees + numPlants + numThrown + numSprayInst + numFallen + numCharInst;
    cmdBuild(cb, tlas, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, TRACE, &tlasGeom, instances, false);
    memBarrier(cb, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT);
}

// ptx — Vulkan ray-query path tracer prototype.
// Bumpy terrain + wind-animated grass (BLAS refit every frame), floating light orbs,
// ReSTIR DI, N-ray first-bounce GI, SVGF-style denoiser, GPU benchmark + fps-target auto scaler.
// Streamed glTF world (cells + shared assets, see world.h), Jolt physics, first-person walking.
#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>
#include <vk_mem_alloc.h>
#include "prof.h"

#include <algorithm>
#include <map>
#include <array>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <unistd.h>

#include <fstream>
#include <sstream>
#include <functional>
#include <random>
#include <string>
#include <thread>
#include <mutex>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "audio.h"
#include "dlss.h"
#include "input.h"
#include "jobs.h"
#include "lighttree.h"
#include "physics.h"
#include "rendergraph.h"
#include "simd.h"
#include "terrain.h"
#include "trees.h"
#include "texgen.h"
#include "texcomp.h"
#include "fracture.h"
#include "editor.h"
#include "matlib.h"
#include <stb_image_write.h>
#include "worldmap.h"
#include "character.h"
#include "world.h"

#ifndef SHADER_DIR
#define SHADER_DIR "shaders/"
#endif
#ifndef WORLD_DIR
#define WORLD_DIR "world/"
#endif

#define VK_CHECK(x)                                                                    \
    do {                                                                               \
        VkResult r_ = (x);                                                             \
        if (r_ != VK_SUCCESS) {                                                        \
            fprintf(stderr, "Vulkan error %d at %s:%d\n", (int)r_, __FILE__, __LINE__); \
            abort();                                                                   \
        }                                                                              \
    } while (0)

// ---------------------------------------------------------------- math
struct V3 { float x = 0, y = 0, z = 0; };
static V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
static V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
static V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }
static float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
static V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
static V3 normalize(V3 a) { return a * (1.f / sqrtf(dot(a, a))); }
static void set4(float* d, V3 v, float w) { d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = w; }

// ---------------------------------------------------------------- vulkan globals
static GLFWwindow* window;
static VkInstance instance;
static VkSurfaceKHR surface;
static VkPhysicalDevice phys;
static VkDevice dev;
static VkQueue queue, computeQueue;
static uint32_t qfam, computeFam;    // computeFam == qfam when there is no separate async compute family
static uint32_t shareFams[2];       // buffers are CONCURRENT across both families (no ownership transfers)
static VkPhysicalDeviceMemoryProperties memProps;
static VmaAllocator vma;
static VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};
static float timestampPeriod;
static char gpuName[256];
static VkCommandPool cmdPool;
static VkCommandBuffer uploadCmd;  // streamed model uploads + BLAS builds, submitted ahead of the frame graph
static VkFence frameFence;
static VkSemaphore acquireSem;
static rg::Graph graph;
static bool asyncCompute = true;    // BVH build on the async compute queue, overlapping the raster G-buffer
static bool asyncFog = true;        // volumetric fog on the async compute queue, overlapping ReSTIR DI + GI

static PFN_vkCreateAccelerationStructureKHR pCreateAS;
static PFN_vkGetAccelerationStructureBuildSizesKHR pGetASSizes;
static PFN_vkCmdBuildAccelerationStructuresKHR pCmdBuildAS;
static PFN_vkGetAccelerationStructureDeviceAddressKHR pGetASAddr;
static PFN_vkDestroyAccelerationStructureKHR pDestroyAS;

static VkSwapchainKHR swapchain;
static VkFormat swapFmt;
// screenshots (F12 scene, Shift+F12 with the UI): <repo>/screenshots/*.png, path copied to the clipboard
static bool swapCanCopy = false, swapBGR = true;
static bool shotUIReq = false, shotUIPending = false, shotSceneReq = false;
static uint32_t uiShotW = 0, uiShotH = 0;
static std::string shotMsg;
static double shotMsgT = -100.0;
static VkExtent2D swapExtent;
static std::vector<VkImage> swapImages;
static std::vector<VkImageView> swapViews;
static std::vector<VkSemaphore> renderDone;
static bool swapDirty = false;

struct Buffer { VkBuffer b = VK_NULL_HANDLE; VmaAllocation m = VK_NULL_HANDLE; VkDeviceSize size = 0; void* map = nullptr; VkDeviceAddress addr = 0; };
struct Image { VkImage img = VK_NULL_HANDLE; VmaAllocation mem = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE; };
struct Accel { VkAccelerationStructureKHR as = VK_NULL_HANDLE; Buffer buf, scratch; VkDeviceAddress addr = 0, scratchAddr = 0; VkDeviceSize size = 0; };

// Screen images, in binding order 10..24 (must match common.glsl)
enum { I_GPOS, I_GNORM, I_GALBEDO, I_GEMIS, I_PREVPOS, I_PREVNORM, I_DIRECT, I_INDIRECT, I_ACCUM, I_ACCUMMOM,
       I_HISTCOLOR, I_HISTMOM, I_PINGA, I_PINGB, I_OUT, I_COUNT };
static const VkFormat imgFormats[I_COUNT] = {
    VK_FORMAT_R32_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
    VK_FORMAT_R32_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
    VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT,
    VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM};
static Image imgs[I_COUNT];
// More screen images, bindings 54 + index (common.glsl and the passes that use them).
enum SizeClass { RENDER_RES, HALF_RES, OUTPUT_RES };
struct XImgDef { VkFormat fmt; SizeClass size; };
enum { X_GMAT, X_GMAT2, X_GTAN, X_SPEC, X_HDR, X_HDROUT, X_FOG, X_FOGHIST, X_GVEL, X_COUNT };
static const XImgDef xDefs[X_COUNT] = {
    {VK_FORMAT_R16G16B16A16_SFLOAT, RENDER_RES}, {VK_FORMAT_R16G16B16A16_SFLOAT, RENDER_RES}, {VK_FORMAT_R16G16B16A16_SFLOAT, RENDER_RES},
    {VK_FORMAT_R16G16B16A16_SFLOAT, RENDER_RES},
    {VK_FORMAT_R16G16B16A16_SFLOAT, OUTPUT_RES},   // HDR scene before post (post.glsl hdrTex)
    {VK_FORMAT_R16G16B16A16_SFLOAT, OUTPUT_RES},   // HDR display encoding (PQ / scRGB)
    {VK_FORMAT_R16G16B16A16_SFLOAT, RENDER_RES}, {VK_FORMAT_R16G16B16A16_SFLOAT, RENDER_RES},  // volumetric fog + its history (top-left 1/fogDiv used)
    {VK_FORMAT_R16G16B16A16_SFLOAT, RENDER_RES}};  // G-buffer velocity: world-space offset to where the surface was last frame
// post-processing (post.glsl)
static constexpr uint32_t BLOOM_MIPS = 7;
static VkSampler postSampler;          // bilinear, clamp
static Image bloomImg;                 // half output resolution, BLOOM_MIPS mips
static VkImageView bloomViews[8];
static Buffer exposureBuf;             // histogram + adapted exposure (host visible: the UI shows the metered EV)
static Image uiImg, presentImg;        // HDR present: UI rendered SDR, then composed into presentImg (swapchain size)
static VkFormat uiFmt = VK_FORMAT_B8G8R8A8_UNORM;
static bool hdrModeAvail[3] = {true, false, false};  // SDR, HDR10 (PQ), scRGB offered by the surface
static int hdrActive = 0;                            // what the swapchain currently is
static bool haveColorspaceExt = false;
static constexpr uint32_t X_BINDING = 54;
static Image ximgs[X_COUNT];
// raster G-buffer targets: gDepth, gNorm, gAlbedo, gEmis, gMat, gMat2, gTan, gVel
static constexpr uint32_t GBUF_TARGETS = 8;
static int gbufX(int i) { return i == 7 ? X_GVEL : X_GMAT + i - 4; }
static VkFormat gbufFormat(int i) { return i < 4 ? imgFormats[I_GPOS + i] : xDefs[gbufX(i)].fmt; }
static Image& gbufImage(int i) { return i < 4 ? imgs[I_GPOS + i] : ximgs[gbufX(i)]; }
static Buffer resCur, resPrev, giCur, giPrev;
static Image depthImg;
static const VkFormat depthFmt = VK_FORMAT_D32_SFLOAT;
static uint32_t W = 1920, H = 1080;   // render resolution (what rays are traced at)
static uint32_t OW = 1920, OH = 1080; // output resolution (DLSS-RR upscales W x H to this; equal without it)
static dlss::Mode upscaler = dlss::QUALITY;
static bool rrActive = false;         // upscaler != OFF and the feature was created
// DLSS-RR inputs (bindings 37..42, render res) and output (43, output res); see dlss_prep.comp
enum { RR_MV, RR_VIEWZ, RR_DIFF, RR_SPEC, RR_NRM, RR_COLOR, RR_OUT, RR_COUNT };
static const VkFormat rrFormats[RR_COUNT] = {VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
                                              VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
                                              VK_FORMAT_R16G16B16A16_SFLOAT};
static Image rrImgs[RR_COUNT];
static float jitter[2] = {0, 0}, prevJitter[2] = {0, 0};

static Buffer terrainV, terrainI, grassV, grassI, bladeBuf, lightBuf, uboBuf, instBuf, grassCounter, countReadback, rockV, rockI, rockXf;
static Image skyImg;
static Image windImg[2];
static Buffer uiShotBuf;  // screenshot with the UI: swapchain readback
// Crater bins (material.glsl craterRange): 32 x 32 cells of 16 m around the camera, each listing the craters whose
// reach touches it, so terrain / shadow lookups loop over a few craters instead of all of them.
static constexpr int CB_N = 32, CB_BINS = CB_N * CB_N, CB_LIST = 16384;
static constexpr float CB_CELL = 16.f;
static Buffer craterBinBuf;
static int cbOrigin[2] = {0, 0};
static bool cbValid = false;
static void buildCraterBins(float cx, float cz) {
    if (!craterBinBuf.map) return;
    uint32_t* b = (uint32_t*)craterBinBuf.map;
    int ox = (int)floorf(cx / CB_CELL) - CB_N / 2, oz = (int)floorf(cz / CB_CELL) - CB_N / 2;
    int n = std::min(gCraterCount.load(std::memory_order_relaxed), MAX_CRATERS);
    static std::vector<uint32_t> lists[CB_BINS];
    for (auto& l : lists) l.clear();
    for (int i = 0; i < n; i++) {
        const Crater& c = gCraters[i];
        float reach = 1.9f * c.r + 1.f;
        int x0 = (int)floorf((c.x - reach) / CB_CELL) - ox, x1 = (int)floorf((c.x + reach) / CB_CELL) - ox;
        int z0 = (int)floorf((c.z - reach) / CB_CELL) - oz, z1 = (int)floorf((c.z + reach) / CB_CELL) - oz;
        for (int z = std::max(z0, 0); z <= std::min(z1, CB_N - 1); z++)
            for (int x = std::max(x0, 0); x <= std::min(x1, CB_N - 1); x++) lists[z * CB_N + x].push_back((uint32_t)i);
    }
    uint32_t at = 0;
    bool overflow = false;
    for (int k = 0; k < CB_BINS; k++) {
        if (at + lists[k].size() > CB_LIST) { overflow = true; break; }
        b[2 * k] = at;
        b[2 * k + 1] = (uint32_t)lists[k].size();
        memcpy(b + 2 * CB_BINS + at, lists[k].data(), lists[k].size() * 4);
        at += (uint32_t)lists[k].size();
    }
    cbOrigin[0] = ox; cbOrigin[1] = oz;
    cbValid = !overflow;  // (overflow: the shaders fall back to looping over all craters)
}
// Ground virtual texture (groundvt.glsl): a clipmap of VT_L levels of VT_N^2 texels around the camera, the finest
// 2^-11 m (0.49 mm), each next one twice as coarse; the coarsest (4 m texels) covers +-2 km, past any view distance. Levels are split into 64^2 tiles, stored toroidally; tiles that
// come into the window (or are out of date after a generator setting changed) are queued and generated by
// vtgen.comp, a budget of them per frame, coarse levels first so there is always something to fall back to.
static constexpr int VT_N = 1024, VT_L = 14, VT_TILE = 64, VT_T = VT_N / VT_TILE;
static constexpr float VT_TEXEL0 = 1.f / 2048.f;
struct VtTileReq { uint32_t level, slot; int32_t tx, tz; };
static Image vtImgA, vtImgB, vtImgH;
static Buffer vtTileBuf;
static struct VtSlot { int tx = INT32_MIN, tz = INT32_MIN; uint32_t ver = 0; } vtSlots[VT_L][VT_T * VT_T];
static uint32_t vtVersion = 1, vtQueued = 0;
static float vtGenKey[2] = {-1, -1};  // generator inputs: pebbles, warp (changing them regenerates everything)

// world map (worldmap.h): edits uploaded from the CPU, and the GPU-owned simulation grid (worldsim.comp)
static Image mapHImg, mapPaintImg, mapWaterImg;  // height delta (r32f), paint (rgba8), drawn water (rgba32f, half res)
static Image simImg, fluxWImg, fluxSImg, simGroundImg;
static float gSimBrush[4] = {0, 0, 0, 0};  // editor snow brush this frame (x, z, radius, m/s)
static float frameDt = 1.f / 60.f;
// Rain swells the river: its surface rises toward ~0.35 m x precipitation over a minute or so, and falls back more
// slowly once it stops (the catchment keeps draining). Raises the water, not the bed; the current speeds up with it.
static float riverSwell = 0.f;
static bool simClear = false;
static int simActiveFrames = 0;
// a 64 x 64 cell (128 m) patch of the simulation around the camera, read back each frame: the physics' view of
// snow and wetness (ground friction / bounce)
static constexpr int SIM_RB = 64;
static Buffer simRB;
static int simRBx = 0, simRBz = 0, simSnapX = 0, simSnapZ = 0;
static float simSnap[SIM_RB * SIM_RB][2];  // snow depth, wetness
static bool simSnapValid = false;
static float halfToFloat(uint16_t h) {
    uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, f;
    if (e == 0) { if (!m) f = s; else { e = 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3ff; f = s | ((e + 112) << 23) | (m << 13); } }
    else if (e == 31) f = s | 0x7f800000u | (m << 13);
    else f = s | ((e + 112) << 23) | (m << 13);
    float r;
    memcpy(&r, &f, 4);
    return r;
}  // keep refreshing the terrain mesh a while after snow was added (avalanches settle)         // snow depth / water depth / wetness / avalanche; water + snow fluxes  // wind map ping-pong (wind.glsl)
static Image groundImg[4];
static uint32_t groundSlots[4];
static float groundMaxH[2];
static bool groundReady = false;
static uint32_t groundTexSize = 4096;
static Accel terrainAS, grassAS, rockAS[16], tlas;  // rockAS: one per variant (ROCK_VARIANTS)
static VkAccelerationStructureGeometryKHR grassGeom, terrainGeom, rockGeom[16], tlasGeom;
static uint32_t terrainTris, rockTris, rockVerts, numRocks = 0;
static uint32_t maxBlades = 400000;  // buffer / BLAS capacity
static uint32_t bladeCap = 0;        // slots built this frame (from last frame's count + headroom)
static uint32_t lastBladeCount = 0, grassGridN = 0;
static float terrainCenter[2] = {1e9f, 1e9f};
static bool terrainDirty = true;
// grass is regenerated (culled, LOD'd, compacted) every frame, so its BLAS is rebuilt rather than refit
static const VkBuildAccelerationStructureFlagsKHR grassFlags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR |
                                                               VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;
// Grass BLAS refit: while the camera stays put the blade set is kept and only re-animated (grass.comp pc.a = 1),
// and the BLAS is updated in place; a full rebuild every GRASS_REBUILD frames or when the camera moves / turns.
static constexpr int GRASS_REBUILD = 8;
static bool grassRefit = false, grassBuilt = false;
static int grassFrames = 0;
static uint32_t grassBuildCap = 0;
static float grassBuildPos[3], grassBuildFwd[3];

static VkDescriptorSetLayout setLayout;
static VkDescriptorPool descPool, imguiPool;
static VkDescriptorSet descSet;
static VkPipelineLayout pipeLayout;
static VkPipelineCache pipeCache;  // persisted next to the binary: warm starts skip driver shader compilation
static const char* PIPE_CACHE_FILE = "pipeline_cache.bin";
enum { P_SKY, P_TERRAIN, P_GRASS, P_GRASS_TAIL, P_PRIMARY, P_RESTIR_INIT, P_RESTIR_SPATIAL, P_GI, P_GI_SPATIAL, P_TEMPORAL, P_ATROUS, P_COMPOSITE,
       P_DLSS_PREP, P_DLSS_TONEMAP, P_IMPOSTOR_BAKE, P_PROBE_BAKE, P_PROBE_FILTER, P_WIND, P_EXPOSURE, P_BLOOM, P_POST, P_PRESENT_HDR, P_VOLFOG, P_SURFACE_FX, P_WORLDSIM, P_RADCACHE, P_SKIN, P_VTGEN, P_COUNT };
static const char* pipeNames[P_COUNT] = {"sky", "terrain", "grass", "grass_tail", "primary", "restir_initial", "restir_spatial", "gi", "restir_gi_spatial", "temporal", "atrous", "composite",
                                         "dlss_prep", "dlss_tonemap", "impostor_bake", "probe_bake", "probe_filter", "wind",
                                         "exposure", "bloom", "post", "present_hdr", "volfog", "surface_fx", "worldsim", "radcache_resolve", "skin", "vtgen"};
static VkPipeline pipes[P_COUNT];
static VkPipeline rasterTerrain, rasterGrass, rasterRock, rasterModel, rasterImpostor;

// model.glsl Material (std430, 128 bytes)
struct GpuMat {
    float color[4];
    float emissive[3]; uint32_t tex;
    float rough, metal, normalScale; uint32_t nrmTex;
    uint32_t ormTex, emisTex, occTex; float occStrength;
    float heightScale, clearcoat, ccRough, sheen;
    float sheenColor[3]; float aniso;
    float anisoRot, transmission, ior, specular;
    float transl, pad[3];
};
static_assert(sizeof(GpuMat) == 128);

// ---------------------------------------------------------------- scene / settings
struct Settings {
    int raysFirst = 1, bounces = 2;
    bool rasterPrimary = true;
    bool restir = true, restirTemporal = true, restirSpatial = true;
    int candidates = 16, spatialSamples = 4;
    float spatialRadius = 20.f;
    bool restirGI = true, giTemporal = true, giSpatial = true;
    bool giEnabled = true;  // off: direct light only (I key)
    bool giCache = true;     // radiance cache ends paths after the first secondary hit (radcache.glsl)
    bool giChecker = true;   // new GI candidates in half the pixels per frame (ReSTIR GI fills the rest)
    bool giProbes = true;    // secondary hits in baked volumes read the probes
    int giSpatialSamples = 4;
    float giRadius = 30.f, giMaxM = 20.f;
    bool accumulate = true, denoise = true;
    int historyMax = 32, atrousIters = 4;
    float fireflyClamp = 20.f;
    float sunElev = 25.f, sunAzim = 40.f, sunIntensity = 8.f, sunRadiusDeg = 0.6f, skyIntensity = 1.f;
    float fogScale = 20.f;
    bool clouds = true;
    bool groundTextures = true, parallax = true;
    float groundNormal = 1.f, parallaxDist = 12.f;
    int groundMode = 3;  // 0 tiled textures, 1 fractal (generated per pixel, never repeats), 2 hybrid, 3 layered virtual texture
    int vtBudget = 96;   // ground virtual texture tiles (64^2 texels) generated per frame
    float fractalRelief = 1.f, fractalPebbles = 0.6f, fractalFinest = 0.001f;
    bool windMap = true;
    float gustStrength = 0.3f, gustSpreadDeg = 45.f, windWaveSpeed = 12.f, windDamping = 0.7f, windRelax = 1.6f;  // calm, plausible gusts
    float cloudCoverage = 0.45f, cloudDensity = 0.03f, cloudSpeed = 1.f, cloudBase = 1500.f, cloudThickness = 1800.f, cloudShadow = 1.f;
    // cloud genera besides the cumulus slab (clouds.glsl): low stratus / stratocumulus deck, cumulonimbus towers
    // rising out of the cumulus layer to an anvil, high cirrus sheet
    // each genus is a share of the one coverage: amount = coverage * weight, so coverage 0 is a clear sky
    float cumulus = 1.f;
    float stratus = 0.55f, stratusLumpy = 0.6f, stratusBase = 700.f, stratusThick = 350.f;
    float cumulonimbus = 0.55f, cbTop = 10000.f, cirrus = 0.8f, cirrusAlt = 9000.f;
    float cirrusLength = 1.f, cirrusFine = 0.5f, cirrusPuff = 0.5f;  // strand length, strand fineness, strands <-> tufts
    // grass blades (grass.comp): clumping, shape, variation
    float grassClump = 0.65f, grassClumpSize = 0.3f, grassWidth = 1.f, grassCurve = 1.f, grassTwist = 0.5f, grassTilt = 0.35f;
    float grassHeightVar = 0.6f, grassDead = 0.12f, grassColorVar = 1.f;
    // ground detail (ground.glsl)
    float groundWarp = 1.f, groundMottle = 1.f, groundMottleScale = 1.f, darkSpots = 0.4f, darkSpotSize = 1.f;
    float groundGrit = 1.f, groundCavity = 1.f, tileBreakup = 1.f, groundSat = 1.f;
    float bushAmount = 1.f, berryShare = 0.35f;  // bushes per forest edge / meadow, fraction of them berry bushes
    float fovMargin = 20.f, keepRadius = 5.f, offscreenDensity = 0.2f;
    float grassDensity = 120.f, grassLength = 1.f, lodStart = 8.f, drawDist = 60.f;
    float craterInflow = 0.25f;
    float riverRainRise = 0.35f;  // river surface rise at full precipitation (m)
    float groundFriction = 0.75f, groundBounce = 0.12f;  // natural ground (soil / grass); other surfaces in groundSurfaceAt
    // world simulation (worldsim.comp): snow above the snow line, avalanches, melt water running to the river
    bool worldSim = true, snowFollowsClouds = true;
    float snowLine = 1500.f, snowfall = 2.f /* m/h at full precipitation */, meltRate = 0.6f /* m/h */;
    float reposeDeg = 38.f, avalanche = 1.f, riverRange = 250.f, dryRate = 0.004f, waterMobility = 1.f, mapTexTile = 2.f;
    float simSpeed = 1.f;   // world sim time scale (fast-forward snow, melt and runoff)
    int snowMeshEvery = 6;  // frames between terrain mesh refreshes while the snow changes  // share of the river discharge into a connected crater
    float viewDist = 800.f;  // world radius drawn: terrain mesh, rocks / trees, streamed cells
    float treeDensity = 0.25f, impostorDist = 110.f;  // impostors from here (+ a 20 m crossfade band)
    float clutter = 1.f, clutterRange = 45.f;      // ground clutter density and distance (ferns, pebbles, twigs, leaf litter)
    int lightCount = 48;
    float orbRange = 60.f;  // width of the tile the orbs drift in around the camera (m)
    float lightIntensity = 1.5f;
    float windStrength = 0.35f, windSpeed = 1.6f, windDirDeg = 30.f;
    float exposure = 1.4f;                          // manual exposure (auto off)
    bool autoExposure = true;
    float evComp = -2.f, adaptUp = 2.5f, adaptDown = 1.2f;  // EV compensation, eye adaptation speed (1/s)
    // physical camera (post.comp): exposure from f-stop, shutter and ISO (EV100) instead of metering; the lens
    // (full-frame thin lens, focal length from the FOV) sets depth of field and aperture vignetting, the ISO the noise
    bool physCam = false;
    float fStop = 8.f, shutterDen = 125.f, iso = 100.f, sceneLum = 11500.f;  // sceneLum: cd/m^2 per scene radiance unit
    bool dof = false, motionBlur = false;
    float focusDist = 0.f;                     // m, 0 = autofocus on the centre
    float mbShutter = 0.5f;                    // fraction of the frame the shutter is open (no physical camera)
    float vignette = 0.35f, chromAb = 0.f, grain = 0.f;
    int tonemapper = 0;                             // 0 AgX, 1 AgX punchy, 2 ACES, 3 none
    float bloom = 0.04f, sharpen = 0.35f;
    int hdrMode = 0;                                // 0 SDR, 1 HDR10 (PQ), 2 scRGB
    float paperWhite = 203.f, peakNits = 1000.f;
    bool grassRefit = true;                          // grass BLAS update in place between rebuilds
    bool fog = true, fogLights = true;               // volumetric fog / god rays (volfog.comp)
    int fogDiv = 2;                                  // fog resolution = render resolution / fogDiv (1..4)
    float fogDensity = 0.0007f, fogFalloff = 0.08f, fogAniso = 0.6f, fogNoise = 0.6f, fogDist = 150.f, fogBlend = 0.12f;
    float rain = 0.f, wetness = 0.f, puddles = 0.f;  // puddles: the old noise-placed ones; the world sim makes real ones  // weather (surface_fx.comp, post.comp)
    bool rainWets = true;                            // wetness follows the rain (dries slowly after)
    float caustics = 1.f;                            // river-bed caustics strength
    bool decals = true;
    float foamTrail = 20.f;
    float flow = 1.f, clarity = 1.f, foam = 1.f, splash = 1.f, absorb = 1.4f;
    float murk[3] = {0.03f, 0.07f, 0.065f};
    float throwSize = 0.4f, throwSpeed = 12.f;
    int viewMode = 0;
};
static Settings S;

static constexpr int MAX_SPRAY = 2048;  // splash droplets
static constexpr int MAX_THROWN = 256;  // thrown rocks and their fragments (physics bodies), also the water shader's list
static constexpr int MAX_DECALS = 128;  // impact decals (frame.glsl decals[])
struct CraterWater { bool pond = false; float level = 0, fill = 1, divert = 0, inflowSpeed = 0, lastR = 0, ground[3] = {}; };  // updateCraterWater
static CraterWater craterWater[MAX_CRATERS];
static constexpr int MAX_DUST = 32;     // dust clouds from impacts (frame.glsl dust[])
struct Decal { float p[3], r, n[3]; int kind; double t0; };  // kind: 0 wall impact, 1 mud splatter, 2 bark scar
struct Dust { float p[3], r, density, rise; double t0; };
static std::vector<Decal> decalList;  // ring: the oldest goes when full
static std::vector<Dust> dustList;
struct UBO {
    float camPos[4], camRight[4], camUp[4], camFwd[4];
    float prevCamPos[4], prevRight[4], prevUp[4], prevFwd[4];
    float sunDir[4], sunColor[4], wind[4], params[4], params2[4];
    uint32_t res[4], settings[4], counts[4];
    float grassA[4], grassB[4];
    int32_t grassGrid[4];
    float terrainP[4];
    float jitter[4];
    uint32_t bake[4];  // probes.glsl: volumes, dynamic lights, flags, lights incl. switched-off ones
    float river[4], water[4], water2[4], waterCol[4];  // water.glsl
    float dynRocks[MAX_THROWN][4];          // thrown rocks / fragments: centre, size (rock-mesh scale equivalent)
    float dynRockV[MAX_THROWN][4];          // their velocity (m/s)
    float splashEv[128][4];                 // water impacts: (x, z, time, impact), (size, speed, -, -)
    float craters[MAX_CRATERS][4];          // terrain.h Crater
    uint32_t dynInfo[4];                    // rocks, splash events, craters
    float clouds[4], clouds2[4];            // clouds.glsl
    uint32_t groundTex[4];                  // ground.glsl
    float groundP[4], groundS[4];
    int32_t windGrid[4];                    // wind.glsl
    float windSim[4], windSim2[4];
    uint32_t lightInfo[4];                  // tri lights, light tree nodes, all emitters listed, decals
    float post0[4], post1[4], post2[4], post3[4], post4[4];  // frame.glsl: exposure, bloom / lens, tonemap, DoF / motion blur, display
    float fog0[4], fog1[4];                 // volumetric fog
    float weather[4];                       // wetness, rain, puddles, caustics
    float decals[2 * MAX_DECALS][4];
    uint32_t fxInfo[4];                     // dust puffs
    float dust[2 * MAX_DUST][4];
    float craterW[MAX_CRATERS][4];          // frame.glsl craterW
    float craterG[MAX_CRATERS][4];          // frame.glsl craterG
    uint32_t mapPal[8];                     // world map ground textures: bindless slot + 1 (0 = none), index 1..7
    float mapPalTile[8];                    // metres per repeat
    float mapP[4];                          // x origin, y cell size, z cells per side, w -
    float sim0[4];                          // x snow line (m), y precipitation 0..1, z melt rate (m/s at +10 m below the line), w river search range (m)
    float sim1[4];                          // x angle of repose (tan), y avalanche mobility, z sim step (s), w sim on
    float sim2[4];                          // x snowfall (m/s at full precipitation), y wetness decay (1/s), z water mobility, w -
    float simBrush[4];                      // editor snow brush: x, z, radius (0 = off), m/s
    float fractal[4];                       // ground detail: mode (0 textures, 1 fractal, 2 hybrid), relief, pebbles, finest scale (m)
    int32_t craterGrid[4];                  // crater bins: first bin xz (16 m), bins per side, valid
    int32_t fogRes[4];                      // volumetric fog: resolution divisor, grid width, height
    float clouds3[4], clouds4[4];           // clouds.glsl: stratus, cumulonimbus, cirrus
    float clouds5[4];                       // cirrus: strand length, fineness, puffiness
    float grassC[4], grassD[4];             // grass.comp: clumping / shape, variation
    float ground2[4], ground3[4];           // ground.glsl: noise, dark spots, grit
    float vegCol[4];                        // x grass colour variation, y ground colour variation
    float vtP[4];                           // ground virtual texture: finest texel (m), texels per side, levels, on
    float vtRect[14][4];                    // per level: resident world xz rectangle
};

struct Camera {
    V3 pos{0, 2.2f, -18.f};
    float yaw = 0.f, pitch = -0.08f, fovDeg = 60.f;
    V3 fwd() const { return {cosf(pitch) * sinf(yaw), sinf(pitch), cosf(pitch) * cosf(yaw)}; }
    V3 right() const { return normalize(cross(fwd(), {0, 1, 0})); }
    V3 up() const { return cross(right(), fwd()); }
};
static Camera cam, prevCam;

static constexpr uint32_t TNV = 257;                           // terrain grid verts per side (matches terrain.comp)
static constexpr float TERRAIN_INNER = 64.f;  // half-extent = inner*|u| + outer*|u|^4, outer = S.viewDist - inner
static constexpr int MAX_LIGHTS = 256;           // orbs + nearest/strongest streamed lights, sampled by ReSTIR
static constexpr int MAX_ORBS = 64;
static constexpr uint32_t MAX_TRI_LIGHTS = 8192;  // emissive triangles in the light list (lights.glsl)
static constexpr int BLUE_NOISE = 64;              // blue noise tile size (lights.glsl bn())
static constexpr int MAX_ROCKS = 16384;
static constexpr int MAX_TREES = 65536;  // a 4 km view over forest
static constexpr int MAX_PLANTS = 16384;
static constexpr uint32_t MAX_MESHES = 16384;      // bindless mesh slots (model.glsl)
static constexpr uint32_t MAX_MATERIALS = 65536;
static constexpr uint32_t MAX_TEXTURES = 4096;     // matches model.glsl
static constexpr uint32_t MAX_MODEL_INST = 32768;  // TLAS / raster instances of streamed meshes
static constexpr uint32_t MAX_ZONES = 256;
static constexpr uint32_t INST_MODELS = 2;          // TLAS layout: terrain, grass, model instances, rocks, trees, plants, thrown rocks, spray
static constexpr uint32_t SKY_W = 256, SKY_H = 128;             // matches atmosphere.glsl

struct LightBase { V3 p; float phase; V3 col; float inten; };
static std::vector<LightBase> lightBase;
static float orbFade[64];  // per orb, 0 at the wrap edge of the tile (MAX_ORBS)
static uint32_t numLights = 0, numWorldLights = 0;

// ---------------------------------------------------------------- streamed models (GPU side)
static Buffer meshInfoBuf, materialBuf, modelXfBuf, zoneBuf;
static Buffer triLightBuf, lightTreeBuf, blueNoiseBuf, lightAliasBuf;  // lights.glsl
static std::vector<uint32_t> emisItems;                  // drawItems with emissive triangles
static uint32_t numTriLights = 0, numTreeNodes = 0;
static bool allEmittersListed = true;
static Image defaultTex;
static VkSampler texSampler;
struct SlotPool {  // free-list allocator for bindless slots
    std::vector<uint32_t> freeList;
    uint32_t next = 0, cap = 0;
    uint32_t alloc() {
        if (!freeList.empty()) { uint32_t s = freeList.back(); freeList.pop_back(); return s; }
        return next < cap ? next++ : UINT32_MAX;
    }
    void release(uint32_t s) { freeList.push_back(s); }
    uint32_t used() const { return next - (uint32_t)freeList.size(); }
};
static SlotPool meshSlots{{}, 0, MAX_MESHES}, matSlots{{}, 0, MAX_MATERIALS}, texSlots{{}, 1, MAX_TEXTURES};  // tex slot 0 = white
struct EmisTri { float v[3][3]; float Le[3]; };  // object-space emissive triangle and its radiance (light list)
struct GpuMesh {
    uint32_t slot = UINT32_MAX; uint32_t firstIndex = 0, indexCount = 0; VkAccelerationStructureKHR blas = VK_NULL_HANDLE; VkDeviceAddress blasAddr = 0;
    std::vector<EmisTri> emis;
};
struct GpuModel {
    Buffer geom, blasMem;               // all meshes' vertices | indices | triangle materials; all BLASes
    std::vector<GpuMesh> meshes;
    std::vector<uint32_t> mats, texs;   // slots
    std::vector<Image> images;
    std::vector<uint32_t> audioMeshes;  // acoustic (simplified) copies, see audio.h
    size_t bytes = 0;
};
static std::vector<std::function<void()>> retireNext;  // freed after the next frame fence (staging, scratch)
static bool uploadsRecorded = false;
static size_t gpuModelBytes = 0;
static float uploadBudgetMB = 96.f;

// placed instances of streamed meshes, rebuilt when the stream changes; bounds in SoA for SIMD culling
struct DrawItem { GpuModel* m; uint32_t mesh; Xf xf; };
static std::vector<DrawItem> drawItems;
static std::vector<float> aabb[6];
static std::vector<uint8_t> visibleItems;
struct DrawRun { GpuModel* m; uint32_t mesh, first, count; };
static std::vector<DrawRun> drawRuns;
static uint32_t numModelInst = 0, numModelVisible = 0;

// procedural trees (trees.h): one non-streamed model, mesh v = variant v
struct TreeInst { V3 p; float yaw, scale; uint32_t v; int ix, iz; };
static Model treeModel;
// World forest: imported assets join the procedural variants (trees::VARIANTS..numTreeVariants-1), so placement,
// the BVH, impostors, trunk colliders and breaking treat them alike. <world>/forest.txt, one "<weight> <asset>" per
// line, plus optional "builtin <weight>" (procedural trees' share, default 0 once assets are listed) and
// "spacing <x>" (bounding footprints x this may not overlap). Read at startup.
static constexpr int MAX_TREE_VARIANTS = trees::VARIANTS + 12;
static int numTreeVariants = trees::VARIANTS;
static std::vector<ModelData> forestAssets;              // variant trees::VARIANTS + i, merged into treeModel by initTrees
static float treeWeight[MAX_TREE_VARIANTS], treeWeightSum = 0.f, forestSpacing = 0.8f;
static float treeFoot[MAX_TREE_VARIANTS];                // footprint radius (local, m) about the trunk; 0 = never blocks
static float treeC[MAX_TREE_VARIANTS][3], treeR[MAX_TREE_VARIANTS];
static std::vector<TreeInst> treeList;
static uint32_t numTrees = 0, numTreeNear = 0, numImpostors = 0;
static bool placeDirty = true;  // re-run rock / tree placement next frame (world streamed in, zones changed)
static Buffer impBuf;                          // impostor instances (impostor.glsl), also the bake's parameters
static Image impAlbedoImg, impNormalImg;       // octahedral atlas, one layer per variant
static void* trunkShape[MAX_TREE_VARIANTS];
static std::unordered_map<uint64_t, uint32_t> treeBodies;  // trunk colliders near the player, keyed by grid cell
// small plants (trees::buildPlants): same idea without impostors or colliders, placed within PLANT_RANGE
static constexpr float PLANT_RANGE = 140.f;
struct PlantInst { float m[12]; uint32_t v; };
static Model plantModel;
static float plantC[trees::PLANT_VARIANTS][3], plantR[trees::PLANT_VARIANTS];
static std::vector<PlantInst> plantList;
static uint32_t numPlants = 0, numPlantsDrawn = 0;
// thrown rocks: dynamic physics bodies sharing the rock mesh (hull for collision), pushed around by the river
// Fragments of broken rocks live in a FragSet: one uploaded model per fracture event, one mesh per piece.
struct FragSet { Model model; int live = 0; };
static std::vector<std::unique_ptr<FragSet>> fragSets;
struct Thrown {
    uint32_t body; float sc; Xf xf; bool wet = false; float v[3] = {}; float cooldown = 0.f, trail = 0.f;
    float radius = 0.f;                          // bounding radius (m)
    FragSet* frag = nullptr; uint32_t fragMesh = 0;  // a fragment: own mesh, own hull; else the rock mesh at scale sc
    uint32_t rockVar = 0;                        // which generated rock mesh (not a fragment)
    void* shape = nullptr;
    std::shared_ptr<fracture::Poly> poly;        // the solid in body space (unscaled), what breaks next time
    float age = 0.f;
    float density = 2600.f;                      // kg/m^3 of its hull (buoyancy)
    bool wood = false;                           // a piece of a broken tree: no craters, floats
    Xf prevXf{}; bool hasPrev = false;           // last frame's pose (motion vectors)
    float rest = 0.f;                            // seconds at rest (retire settled debris)
    std::shared_ptr<trees::Wood> woodPart;       // a loose part of a procedural tree: breaks again on landing
    float woodScale = 1.f; V3 woodCtr{};         // its frame: body space = woodScale * tree local - woodCtr
    uint32_t woodSeed = 0;                       // its tree's (splinters stay the same on every rebuild)
};
// breakable buildings (breakStatic): connected components of a static mesh (triangles, local AABB)
struct MeshParts { std::vector<std::vector<uint32_t>> tris; std::vector<std::array<float, 6>> box; };
static std::unordered_map<const MeshData*, MeshParts> meshPartsCache;  // cleared when the streamed set changes
static std::vector<std::shared_ptr<Model>> damagedModels;               // keeps override copies' GPU side alive
struct FractureReq { uint32_t body; V3 contact, dir; float En; };
// Scattered ground rocks get static colliders near thrown rocks (like tree trunks); a hard enough hit knocks one
// loose: it leaves the placement (brokenRocks) and becomes a thrown rock that fractures by the same energy rule.
struct RockBody { uint32_t body; Xf xf; float sc; uint32_t var; };
static std::unordered_map<uint64_t, RockBody> rockBodies;
static std::unordered_set<uint64_t> brokenRocks;
static uint32_t numBreakableRocks = 0;  // the first rocks in rockXf: grid + buried + hand-placed (not river rocks)
static inline uint64_t rockKey(float x, float z) { return ((uint64_t)(uint32_t)(int)lroundf(x * 8.f) << 32) | (uint32_t)(int)lroundf(z * 8.f); }
static std::vector<FractureReq> pendingFractures;
// trees / walls broken during the physics update, carried out in the next upload window (processFractures)
struct TreeBreakReq { TreeInst t; V3 impulse, at; float severity; };  // imported trees
struct WallBreakReq { uint32_t body; V3 p, dir, v; float En, reach; };
static std::vector<TreeBreakReq> pendingTreeBreaks;
static std::vector<WallBreakReq> pendingWallBreaks;
static uint32_t numThrownRock = 0;  // thrown that use the rock mesh (raster rock list)
static uint32_t fragsDrawn = 0;
static std::vector<Thrown> thrown;
static uint32_t numThrown = 0;
static int autoThrow = 0;
static bool treeLog = false;
static int treeThrow = 0;  // test: rocks at the nearest tree (trunk, limbs, crown in turn), hits logged
// water impacts for water.glsl (crown, cavity, rings, foam): every entry, skip and wake puff of anything thrown
struct SplashEvent { float x, z, t, impact, sc, speed; };
static constexpr int MAX_SPLASH_EV = 64;
static constexpr float SPLASH_LIFE = 40.f, SPLASH_FOAM_TAU = 12.f;  // s, water.glsl dropLocal / dropFoam
static std::vector<SplashEvent> splashEvents;
// debris: ballistic particles (CPU), drawn and ray traced as instances of the plant model's debris meshes
enum DebrisKind : uint8_t { DEB_WATER, DEB_DIRT, DEB_WOOD, DEB_LEAF, DEB_KINDS };
static const int debrisMesh[DEB_KINDS] = {trees::PLANT_SPRAY, trees::PLANT_CLOD, trees::PLANT_SPLINTER, trees::PLANT_LEAF};
struct Droplet { V3 p, v; float size, life, delay; uint8_t kind = DEB_WATER; V3 spin{0, 1, 0}; V3 prevP{0, 0, 0}; bool hasPrev = false; };
static std::vector<Droplet> spray;
static uint32_t numSprayInst = 0;
// Particle transform. Water stretches along its velocity (reads as a streak of spray); clods and splinters
// tumble about their spin axis.
static void dropletXf(const Droplet& d, float m[3][4]) {
    float sp = sqrtf(dot(d.v, d.v));
    V3 a;
    float s = d.size, sa = s;
    if (d.kind == DEB_WATER) {
        a = sp > 1e-3f ? d.v * (1.f / sp) : V3{0, 1, 0};
        sa = s * (1.f + 0.06f * sp);
    } else {
        V3 q = normalize(cross(d.spin, fabsf(d.spin.y) < 0.9f ? V3{0, 1, 0} : V3{1, 0, 0}));
        float ang = d.life * 9.f;
        a = normalize(q * cosf(ang) + cross(d.spin, q) * sinf(ang));
    }
    V3 t = normalize(cross(fabsf(a.y) < 0.99f ? V3{0, 1, 0} : V3{1, 0, 0}, a)), b = cross(a, t);
    V3 cols[3] = {t * s, a * sa, b * s};  // local x, y (along velocity / tumble axis), z
    for (int r = 0; r < 3; r++) {
        m[r][0] = (&cols[0].x)[r]; m[r][1] = (&cols[1].x)[r]; m[r][2] = (&cols[2].x)[r];
        m[r][3] = (&d.p.x)[r];
    }
}
// broken trees: the pieces above the snap are thrown (wood = true); the stump stays, a static body in the
// placed tree's grid cell
struct Stump { FragSet* frag; uint32_t mesh, body; void* shape; Xf xf; uint64_t key = 0; };  // key: damaged procedural tree
static constexpr int MAX_FALLEN = 256;
static constexpr int MAX_CHAR_INST = 128;  // skinned meshes of all characters (character.h), after the stumps in the TLAS
static uint32_t numCharInst = 0;
static constexpr VkBuildAccelerationStructureFlagsKHR charBlasFlags =
    VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR | VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR;
static std::vector<Stump> stumps;
static std::unordered_set<uint64_t> brokenTrees;  // grid cells of broken trees (placement skips them)
// procedural trees hit but standing (what's left of them, a Stump with this key): processWood
static std::unordered_map<uint64_t, std::pair<TreeInst, trees::Wood>> damagedTrees;
static uint32_t numFallen = 0;
static inline uint64_t treeKey(int ix, int iz) { return ((uint64_t)(uint32_t)ix << 32) | (uint32_t)iz; }

static std::vector<float> rockPts;  // rock mesh vertices of all variants (pos3 nrm3, rockVerts each), for the hulls
static void* rockHull[16] = {};

// baked interiors (probes.glsl): a probe grid per bake volume, baked progressively on the GPU for both switch
// states and cached in <world>/bake/
static constexpr uint32_t MAX_VOLUMES = 64, MAX_PROBES = 400000, BAKE_PASSES = 16, BAKE_CHUNK = 8192, BAKE_FILTERS = 2;
struct BakeVol {
    ProbeVolume v;
    float spacing = 0.25f;
    uint32_t dims[3] = {}, first = 0, count = 0;
    bool on = true;
    bool ready[2] = {}, saved[2] = {};
    uint32_t pass[2] = {}, next[2] = {};
};
static std::vector<BakeVol> bakeVols;
static Buffer volBuf, probeBuf, probeTmp;
static constexpr uint32_t RC_ENTRIES = 1u << 20, RC_STRIDE = 64;  // radiance cache (radcache.glsl RC_SIZE, RC_STRIDE * 4)
static Buffer radCacheBuf;
static bool bakedGI = true, bakeEnabled = true, startLightsOff = false;
static uint64_t bakeEnv = 0;   // hash of everything the bakes depend on (sun, sky, light scale)
static std::string bakeDir;
static uint32_t bakeOffLights = 0;
static bool modelInstDirty = true;
static uint32_t zoneVersionSeen = 0;

// player / streaming settings
static bool walkMode = false, grounded = false, walkReset = false;
static float flySpeed = 1.f;  // fly-mode speed multiplier (UI), Ctrl = x8 on top
static bool gyroAim = true;
static float gyroSens = 1.0f;
static V3 playerFeet;
static float lightScale = 4.66e-4f;  // glTF candela -> renderer intensity: 1 / (683 pi), so Blender watts match the sun scale
static int profFrames = 0;
static bool mapTest = false;           // --prof N
static int ragdollFrame = -1;           // --ragdoll-at F
static int mapClickFrame = -1;         // --map-click F: test the world map brush through ImGui input (M, then hold left mouse at the screen centre)
static int loadRadius = 12;          // cells (64 m): ~768 m, matching the terrain horizon so buildings don't vanish in open ground

static uint32_t frameIndex = 0;
static double simTime = 0.0;
static bool resetHistory = true;
static bool resetRR = true;

// ---------------------------------------------------------------- timing / benchmark
static float gpuTotal = 0, gpuEmaTotal = 0, gpuEmaGI = 0;
static float cpuFrameMs = 16.f;
static bool haveTimestamps = false;

struct Bench {
    static constexpr int WARM = 60, MEAS = 240;
    bool running = false, done = false;
    int tier = 0, frame = 0;
    double vt = 0;
    std::vector<float> samples[5];
    float avg[5] = {}, p95[5] = {};
    Settings saved;
    Camera savedCam;
} bench;
static int submittedTier = 0, submittedFrame = 0;

static int targetFps = 0;          // 0 = uncapped/manual
static bool vsync = true;          // FIFO: frames paced to the display, no uneven-present judder
static int scaleDown = 0, scaleUp = 0;

// ---------------------------------------------------------------- vulkan helpers
// All GPU memory goes through VMA: sub-allocated from large blocks, budget-aware.
static Buffer mkBuf(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags req, VkMemoryPropertyFlags pref = 0) {
    Buffer b;
    b.size = size;
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size;
    ci.usage = usage | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT;
    ci.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    if (computeFam != qfam) {
        ci.sharingMode = VK_SHARING_MODE_CONCURRENT;
        ci.queueFamilyIndexCount = 2;
        ci.pQueueFamilyIndices = shareFams;
    }
    VmaAllocationCreateInfo ac{};
    ac.requiredFlags = req;
    ac.preferredFlags = pref;
    if (req & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ac.flags = VMA_ALLOCATION_CREATE_MAPPED_BIT;
    VmaAllocationInfo info;
    VK_CHECK(vmaCreateBuffer(vma, &ci, &ac, &b.b, &b.m, &info));
    b.map = info.pMappedData;
    VkBufferDeviceAddressInfo dai{VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO};
    dai.buffer = b.b;
    b.addr = vkGetBufferDeviceAddress(dev, &dai);
    return b;
}

static void freeBuf(Buffer& b) {
    if (b.b) vmaDestroyBuffer(vma, b.b, b.m);
    b = {};
}

static Image mkImageCI(const VkImageCreateInfo& ci, VkImageAspectFlags aspect) {
    Image im;
    VmaAllocationCreateInfo ac{};
    ac.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    // screen-sized render targets get their own block so resizes don't fragment the shared heap
    if ((uint64_t)ci.extent.width * ci.extent.height >= 1024 * 1024 && ci.mipLevels == 1)
        ac.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    VK_CHECK(vmaCreateImage(vma, &ci, &ac, &im.img, &im.mem, nullptr));
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = im.img;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
    vi.format = ci.format;
    vi.subresourceRange = {aspect, 0, ci.mipLevels, 0, 1};
    VK_CHECK(vkCreateImageView(dev, &vi, nullptr, &im.view));
    return im;
}

static void oneShot(const std::function<void(VkCommandBuffer)>& f) {
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = cmdPool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VkCommandBuffer cb;
    VK_CHECK(vkAllocateCommandBuffers(dev, &ai, &cb));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    f(cb);
    vkEndCommandBuffer(cb);
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(queue, 1, &si, VK_NULL_HANDLE));
    vkQueueWaitIdle(queue);
    vkFreeCommandBuffers(dev, cmdPool, 1, &cb);
}

static Buffer uploadBuf(const void* data, size_t size, VkBufferUsageFlags usage) {
    Buffer st = mkBuf(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memcpy(st.map, data, size);
    Buffer b = mkBuf(size, usage | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    oneShot([&](VkCommandBuffer cb) {
        VkBufferCopy c{0, 0, size};
        vkCmdCopyBuffer(cb, st.b, b.b, 1, &c);
    });
    freeBuf(st);
    return b;
}

static void memBarrier(VkCommandBuffer cb, VkPipelineStageFlags ss, VkAccessFlags sa, VkPipelineStageFlags ds, VkAccessFlags da) {
    VkMemoryBarrier m{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    m.srcAccessMask = sa;
    m.dstAccessMask = da;
    vkCmdPipelineBarrier(cb, ss, ds, 0, 1, &m, 0, nullptr, 0, nullptr);
}

static void imgBarrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags ss,
                       VkAccessFlags sa, VkPipelineStageFlags ds, VkAccessFlags da) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
    b.srcAccessMask = sa;
    b.dstAccessMask = da;
    vkCmdPipelineBarrier(cb, ss, ds, 0, 0, nullptr, 0, nullptr, 1, &b);
}

static void computeBarrier(VkCommandBuffer cb) {
    memBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
               VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
}

static Image mkImage(uint32_t w, uint32_t h, VkFormat fmt) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = fmt;
    ci.extent = {w, h, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = fmt == depthFmt ? VK_IMAGE_USAGE_DEPTH_STENCIL_ATTACHMENT_BIT
                               : VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT |
                                     VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT;  // sampled: DLSS-RR inputs
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    return mkImageCI(ci, fmt == depthFmt ? VK_IMAGE_ASPECT_DEPTH_BIT : VK_IMAGE_ASPECT_COLOR_BIT);
}

static Image mkImageArray(uint32_t w, uint32_t h, uint32_t layers, VkFormat fmt) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = fmt;
    ci.extent = {w, h, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = layers;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    Image im;
    VmaAllocationCreateInfo ac{};
    ac.usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
    ac.flags = VMA_ALLOCATION_CREATE_DEDICATED_MEMORY_BIT;
    VK_CHECK(vmaCreateImage(vma, &ci, &ac, &im.img, &im.mem, nullptr));
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = im.img;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    vi.format = fmt;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
    VK_CHECK(vkCreateImageView(dev, &vi, nullptr, &im.view));
    return im;
}

static void freeImage(Image& im) {
    if (im.view) vkDestroyImageView(dev, im.view, nullptr);
    if (im.img) vmaDestroyImage(vma, im.img, im.mem);
    im = {};
}

static std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); abort(); }
    std::vector<char> d((size_t)f.tellg());
    f.seekg(0);
    f.read(d.data(), d.size());
    return d;
}

// ---------------------------------------------------------------- device setup
static bool hasExt(VkPhysicalDevice pd, const char* name) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> e(n);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, e.data());
    for (auto& x : e)
        if (!strcmp(x.extensionName, name)) return true;
    return false;
}

// --validate: validation layer messages (warnings and errors), each distinct message id printed only a few times
static VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;
static std::atomic<uint32_t> validationErrors{0};
static VKAPI_ATTR VkBool32 VKAPI_CALL validationCb(VkDebugUtilsMessageSeverityFlagBitsEXT sev, VkDebugUtilsMessageTypeFlagsEXT,
                                                   const VkDebugUtilsMessengerCallbackDataEXT* d, void*) {
    static std::mutex m;
    static std::unordered_map<int32_t, int> seen;
    bool err = sev & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    if (err) validationErrors++;
    std::lock_guard<std::mutex> lk(m);
    if (++seen[d->messageIdNumber] > 3) return VK_FALSE;
    fprintf(stderr, "[vk %s] %s\n", err ? "ERROR" : "warn", d->pMessage);
    return VK_FALSE;
}

static void initVulkan(bool validate) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "ptx";
    app.apiVersion = VK_API_VERSION_1_3;
    uint32_t n = 0;
    const char** gext = glfwGetRequiredInstanceExtensions(&n);
    std::vector<const char*> exts(gext, gext + n);
    std::vector<const char*> ngxInst, ngxDev;
    dlss::requiredExtensions(ngxInst, ngxDev);
    {
        uint32_t ne = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &ne, nullptr);
        std::vector<VkExtensionProperties> av(ne);
        vkEnumerateInstanceExtensionProperties(nullptr, &ne, av.data());
        for (auto e : ngxInst) {
            bool have = std::any_of(av.begin(), av.end(), [&](auto& x) { return !strcmp(x.extensionName, e); });
            bool dup = std::any_of(exts.begin(), exts.end(), [&](auto x) { return !strcmp(x, e); });
            if (have && !dup) exts.push_back(e);
        }
    }
    {   // HDR swapchain colour spaces
        uint32_t ne = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &ne, nullptr);
        std::vector<VkExtensionProperties> av(ne);
        vkEnumerateInstanceExtensionProperties(nullptr, &ne, av.data());
        for (auto& x : av)
            if (!strcmp(x.extensionName, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME)) haveColorspaceExt = true;
        if (haveColorspaceExt) exts.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
    }
    bool haveDebugUtils = false;
    {   // debug labels: Nsight Systems / RenderDoc show GPU passes by name
        uint32_t ne = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &ne, nullptr);
        std::vector<VkExtensionProperties> av(ne);
        vkEnumerateInstanceExtensionProperties(nullptr, &ne, av.data());
        for (auto& x : av) if (!strcmp(x.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) haveDebugUtils = true;
        if (haveDebugUtils && !std::any_of(exts.begin(), exts.end(), [](auto x) { return !strcmp(x, VK_EXT_DEBUG_UTILS_EXTENSION_NAME); }))
            exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    const char* layer = "VK_LAYER_KHRONOS_validation";
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = (uint32_t)exts.size();
    ici.ppEnabledExtensionNames = exts.data();
    if (validate) { ici.enabledLayerCount = 1; ici.ppEnabledLayerNames = &layer; }
    VkDebugUtilsMessengerCreateInfoEXT dmi{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    dmi.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    dmi.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
    dmi.pfnUserCallback = validationCb;
    if (validate && haveDebugUtils) ici.pNext = &dmi;  // also covers vkCreateInstance / vkDestroyInstance themselves
    VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));
    if (validate && haveDebugUtils)
        if (auto f = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"))
            f(instance, &dmi, nullptr, &debugMessenger);
    if (validate) fprintf(stderr, "validation: %s\n", debugMessenger ? "on" : "layer on, no debug utils messenger");
    if (haveDebugUtils)
        rg::setDebugLabels((PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetInstanceProcAddr(instance, "vkCmdBeginDebugUtilsLabelEXT"),
                           (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetInstanceProcAddr(instance, "vkCmdEndDebugUtilsLabelEXT"));
    VK_CHECK(glfwCreateWindowSurface(instance, window, nullptr, &surface));

    uint32_t pdc = 0;
    vkEnumeratePhysicalDevices(instance, &pdc, nullptr);
    std::vector<VkPhysicalDevice> pds(pdc);
    vkEnumeratePhysicalDevices(instance, &pdc, pds.data());
    for (auto pd : pds) {
        if (!hasExt(pd, VK_KHR_RAY_QUERY_EXTENSION_NAME) || !hasExt(pd, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)) continue;
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qp(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qp.data());
        for (uint32_t i = 0; i < qn; i++) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface, &present);
            if ((qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && present) {
                VkPhysicalDeviceProperties pr;
                vkGetPhysicalDeviceProperties(pd, &pr);
                if (!phys || pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { phys = pd; qfam = i; }
                break;
            }
        }
    }
    if (!phys) { fprintf(stderr, "No GPU with VK_KHR_ray_query found\n"); exit(1); }

    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &asProps;
    vkGetPhysicalDeviceProperties2(phys, &p2);
    timestampPeriod = p2.properties.limits.timestampPeriod;
    snprintf(gpuName, sizeof gpuName, "%s", p2.properties.deviceName);
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);

    // async compute: a compute-only family (NVIDIA: 8 queues, runs alongside the graphics queue)
    computeFam = qfam;
    {
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qp(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, qp.data());
        for (uint32_t i = 0; i < qn; i++)
            if ((qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && qp[i].timestampValidBits) { computeFam = i; break; }
    }
    shareFams[0] = qfam;
    shareFams[1] = computeFam;
    float prio = 1.f;
    VkDeviceQueueCreateInfo qcis[2] = {{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}, {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}};
    for (int i = 0; i < 2; i++) {
        qcis[i].queueFamilyIndex = i ? computeFam : qfam;
        qcis[i].queueCount = 1;
        qcis[i].pQueuePriorities = &prio;
    }

    VkPhysicalDeviceRayQueryFeaturesKHR rqf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    rqf.rayQuery = VK_TRUE;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR asf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    asf.accelerationStructure = VK_TRUE;
    asf.pNext = &rqf;
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    f13.shaderDemoteToHelperInvocation = VK_TRUE;  // NGX (DLSS-RR) shader modules declare it
    f13.pNext = &asf;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE;
    f12.timelineSemaphore = VK_TRUE;   // render graph cross-queue sync
    f12.hostQueryReset = VK_TRUE;      // timestamps reset from the CPU, so any queue can write them
    f12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;  // model textures
    f12.pNext = &f13;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f12;
    f2.features.geometryShader = VK_TRUE;     // gl_PrimitiveID in raster_model.frag (per-triangle material)
    f2.features.samplerAnisotropy = VK_TRUE;
    f2.features.shaderFloat64 = VK_TRUE;                       // groundgen.glsl: kilometre positions at sub-mm precision
    f2.features.shaderStorageImageExtendedFormats = VK_TRUE;   // vtgen.comp writes r16
    f2.features.fragmentStoresAndAtomics = VK_TRUE;  // material.glsl declares the world sim image (binding 93) writable for worldsim.comp

    std::vector<const char*> dext = {VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                                     VK_KHR_RAY_QUERY_EXTENSION_NAME, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};
    // ImGui's Vulkan backend loads the KHR alias of vkCmdBeginRendering
    if (hasExt(phys, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME)) dext.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
    for (auto e : ngxDev)  // DLSS-RR (NGX) device extensions, where the driver has them (BDA is core 1.2, enabled above)
        if (strcmp(e, VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) && hasExt(phys, e) && std::none_of(dext.begin(), dext.end(), [&](auto x) { return !strcmp(x, e); })) dext.push_back(e);
    bool memBudget = hasExt(phys, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    if (memBudget) dext.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2;
    dci.queueCreateInfoCount = computeFam != qfam ? 2 : 1;
    dci.pQueueCreateInfos = qcis;
    dci.enabledExtensionCount = (uint32_t)dext.size();
    dci.ppEnabledExtensionNames = dext.data();
    VK_CHECK(vkCreateDevice(phys, &dci, nullptr, &dev));
    vkGetDeviceQueue(dev, qfam, 0, &queue);
    vkGetDeviceQueue(dev, computeFam, 0, &computeQueue);

    VmaAllocatorCreateInfo vci{};
    vci.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT | (memBudget ? VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT : 0);
    vci.physicalDevice = phys;
    vci.device = dev;
    vci.instance = instance;
    vci.vulkanApiVersion = VK_API_VERSION_1_3;
    VK_CHECK(vmaCreateAllocator(&vci, &vma));

    pCreateAS = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(dev, "vkCreateAccelerationStructureKHR");
    pGetASSizes = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(dev, "vkGetAccelerationStructureBuildSizesKHR");
    pCmdBuildAS = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(dev, "vkCmdBuildAccelerationStructuresKHR");
    pGetASAddr = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(dev, "vkGetAccelerationStructureDeviceAddressKHR");
    pDestroyAS = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(dev, "vkDestroyAccelerationStructureKHR");

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = qfam;
    VK_CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &cmdPool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = cmdPool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(dev, &cai, &uploadCmd));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VK_CHECK(vkCreateFence(dev, &fci, nullptr, &frameFence));
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VK_CHECK(vkCreateSemaphore(dev, &sci, nullptr, &acquireSem));
    {
        std::vector<char> blob;
        if (std::ifstream f{PIPE_CACHE_FILE, std::ios::binary | std::ios::ate}) {
            blob.resize((size_t)f.tellg());
            f.seekg(0);
            f.read(blob.data(), blob.size());
        }
        VkPipelineCacheCreateInfo pci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
        pci.initialDataSize = blob.size();  // the driver validates the header and ignores a stale/foreign blob
        pci.pInitialData = blob.data();
        if (vkCreatePipelineCache(dev, &pci, nullptr, &pipeCache) != VK_SUCCESS) {
            pci.initialDataSize = 0;
            VK_CHECK(vkCreatePipelineCache(dev, &pci, nullptr, &pipeCache));
        }
    }
    if (!dlss::init(instance, phys, dev)) upscaler = dlss::OFF;
    rg::QueueInfo qs[rg::QUEUE_COUNT] = {{queue, qfam}, {computeQueue, computeFam}};
    graph.init(phys, dev, qs, timestampPeriod);
}

static Image mkImageUsage(uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage, uint32_t mips = 1) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = fmt;
    ci.extent = {w, h, 1};
    ci.mipLevels = mips;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    ci.initialLayout = VK_IMAGE_LAYOUT_UNDEFINED;
    return mkImageCI(ci, VK_IMAGE_ASPECT_COLOR_BIT);
}
static void writePresentDescriptors();
// Swapchain-sized images for the HDR present (UI target + composed frame); always present so descriptors stay valid.
static void createPresentImages() {
    if (uiImg.img) { vkDeviceWaitIdle(dev); freeImage(uiImg); freeImage(presentImg); }
    uiImg = mkImageUsage(swapExtent.width, swapExtent.height, uiFmt, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    presentImg = mkImageUsage(swapExtent.width, swapExtent.height, VK_FORMAT_R16G16B16A16_SFLOAT,
                              VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    oneShot([](VkCommandBuffer cb) {
        for (Image* im : {&uiImg, &presentImg})
            imgBarrier(cb, im->img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    });
    if (descSet) writePresentDescriptors();
}

static void createSwapchain() {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys, surface, &caps);
    int fw, fh;
    glfwGetFramebufferSize(window, &fw, &fh);
    swapExtent = caps.currentExtent;
    if (swapExtent.width == 0xFFFFFFFF) {
        swapExtent.width = std::clamp((uint32_t)fw, caps.minImageExtent.width, caps.maxImageExtent.width);
        swapExtent.height = std::clamp((uint32_t)fh, caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    uint32_t n = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface, &n, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface, &n, fmts.data());
    VkSurfaceFormatKHR fmt = fmts[0];
    for (auto& f : fmts)
        if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { fmt = f; break; }
    uiFmt = fmt.format;  // ImGui's target format: the SDR swapchain, or the UI image in HDR mode
    // HDR: HDR10 (10-bit PQ, Rec.2020) or scRGB (fp16 linear, extended sRGB), when the surface offers them
    hdrModeAvail[1] = hdrModeAvail[2] = false;
    VkSurfaceFormatKHR hdrFmt[3] = {fmt, fmt, fmt};
    for (auto& f : fmts) {
        if (f.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT && (f.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || f.format == VK_FORMAT_A2R10G10B10_UNORM_PACK32)) { hdrModeAvail[1] = true; hdrFmt[1] = f; }
        if (f.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT && f.format == VK_FORMAT_R16G16B16A16_SFLOAT) { hdrModeAvail[2] = true; hdrFmt[2] = f; }
    }
    hdrActive = S.hdrMode > 0 && S.hdrMode < 3 && hdrModeAvail[S.hdrMode] ? S.hdrMode : 0;
    fmt = hdrFmt[hdrActive];
    swapFmt = fmt.format;

    vkGetPhysicalDeviceSurfacePresentModesKHR(phys, surface, &n, nullptr);
    std::vector<VkPresentModeKHR> modes(n);
    vkGetPhysicalDeviceSurfacePresentModesKHR(phys, surface, &n, modes.data());
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    if (!vsync) {
        for (auto m : modes) if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) mode = m;
        for (auto m : modes) if (m == VK_PRESENT_MODE_MAILBOX_KHR) mode = m;
    }

    uint32_t count = caps.minImageCount + 1;
    if (caps.maxImageCount && count > caps.maxImageCount) count = caps.maxImageCount;

    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = surface;
    ci.minImageCount = count;
    ci.imageFormat = fmt.format;
    ci.imageColorSpace = fmt.colorSpace;
    ci.imageExtent = swapExtent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    swapCanCopy = (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;  // screenshots with the UI
    if (swapCanCopy) ci.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    swapBGR = fmt.format == VK_FORMAT_B8G8R8A8_UNORM || fmt.format == VK_FORMAT_B8G8R8A8_SRGB;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = mode;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = swapchain;
    VkSwapchainKHR sc;
    VK_CHECK(vkCreateSwapchainKHR(dev, &ci, nullptr, &sc));
    for (auto v : swapViews) vkDestroyImageView(dev, v, nullptr);
    for (auto s : renderDone) vkDestroySemaphore(dev, s, nullptr);
    if (swapchain) vkDestroySwapchainKHR(dev, swapchain, nullptr);
    swapchain = sc;

    vkGetSwapchainImagesKHR(dev, swapchain, &n, nullptr);
    swapImages.resize(n);
    vkGetSwapchainImagesKHR(dev, swapchain, &n, swapImages.data());
    swapViews.resize(n);
    renderDone.resize(n);
    for (uint32_t i = 0; i < n; i++) {
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = swapImages[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = swapFmt;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(dev, &vi, nullptr, &swapViews[i]));
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(dev, &sci, nullptr, &renderDone[i]));
    }
    createPresentImages();
}

// ---------------------------------------------------------------- scene generation
// terrain height including the currently streamed flatten zones
static float heightFn(float x, float z) {
    auto& zs = stream::zones();
    return terrainHeight(x, z, zs.data(), (int)zs.size());
}
// Sun transmittance through the atmosphere at the viewer (mirrors atmosphere.glsl), for the sun colour.
static V3 sunTransmittance(V3 sd) {
    const double RG = 6360e3, RT = 6460e3;
    double py = RG + 300.0;
    double b = py * sd.y, c = py * py - RG * RG, disc = b * b - c;
    if (disc >= 0 && -b - sqrt(disc) > 0) return {0, 0, 0};
    double L = -b + sqrt(b * b - (py * py - RT * RT)), ds = L / 32;
    double od[3] = {0, 0, 0};
    for (int i = 0; i < 32; i++) {
        double t = (i + 0.5) * ds, x = sd.x * t, y = py + sd.y * t, z = sd.z * t;
        double h = sqrt(x * x + y * y + z * z) - RG;
        double r = exp(-h / 8000.0), m = exp(-h / 1200.0), o = std::max(0.0, 1.0 - fabs(h - 25000.0) / 15000.0);
        const double br[3] = {5.802e-6, 13.558e-6, 33.1e-6}, bo[3] = {0.650e-6, 1.881e-6, 0.085e-6};
        for (int k = 0; k < 3; k++) od[k] += br[k] * r + 4.40e-6 * m + bo[k] * o;
    }
    return {(float)exp(-od[0] * ds), (float)exp(-od[1] * ds), (float)exp(-od[2] * ds)};
}

// Procedural boulder, instanced as rocks. Real stones are broken blocks worn round, not lumpy spheres:
//  1. fracture: the radius in each direction is a soft minimum over a dozen random cleavage planes (plus a bounding
//     sphere), so the stone has flat-ish faces; the softness is the weathering that rounds the edges and corners
//  2. bedding: shallow ledges along one tilted axis (layered rock splits along its strata)
//  3. cracks: a few joints cut in as narrow grooves
//  4. erosion: ridged 3D noise (sharp crests, soft hollows) plus broad lumps; the base is flattened where it sits
static float rockHash(int x, int y, int z) {
    uint32_t h = (uint32_t)x * 73856093u ^ (uint32_t)y * 19349663u ^ (uint32_t)z * 83492791u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return (h & 0xFFFFFF) / 16777216.f;
}
static float rockNoise(V3 p) {
    int ix = (int)floorf(p.x), iy = (int)floorf(p.y), iz = (int)floorf(p.z);
    float fx = p.x - ix, fy = p.y - iy, fz = p.z - iz;
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy); fz = fz * fz * (3 - 2 * fz);
    auto L = [](float a, float b, float t) { return a + (b - a) * t; };
    float c[2][2];
    for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++) c[j][k] = L(rockHash(ix, iy + j, iz + k), rockHash(ix + 1, iy + j, iz + k), fx);
    return L(L(c[0][0], c[1][0], fy), L(c[0][1], c[1][1], fy), fz);
}
static float rockSmooth(float e0, float e1, float x) { x = std::clamp((x - e0) / (e1 - e0), 0.f, 1.f); return x * x * (3 - 2 * x); }
// Every rock variant has its own seed, so its own cleavage planes, wear, proportions, bedding and erosion. All
// share the subdivided-icosahedron topology (one index list, ROCK_VERTS vertices each); instances pick a variant.
constexpr int ROCK_VARIANTS = 16;
static void buildRockMesh(uint32_t seed, std::vector<float>& verts, std::vector<uint32_t>& idx) {
    const float t = 1.618034f;
    std::vector<V3> v = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t},
                         {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
    idx = {0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
           3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1};
    for (int sub = 0; sub < 4; sub++) {
        std::vector<uint32_t> out;
        std::unordered_map<uint64_t, uint32_t> cache;
        auto mid = [&](uint32_t a, uint32_t b) {
            uint64_t key = ((uint64_t)std::min(a, b) << 32) | std::max(a, b);
            auto it = cache.find(key);
            if (it != cache.end()) return it->second;
            v.push_back((v[a] + v[b]) * 0.5f);
            return cache[key] = (uint32_t)v.size() - 1;
        };
        for (size_t i = 0; i < idx.size(); i += 3) {
            uint32_t a = idx[i], b = idx[i + 1], c = idx[i + 2], ab = mid(a, b), bc = mid(b, c), ca = mid(c, a);
            out.insert(out.end(), {a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca});
        }
        idx = out;
    }
    std::mt19937 rng(seed * 2654435761u + 20261023u);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    // per-variant character: few big facets (blocky) .. many (rounded), sharp .. worn, flat slab .. tall, elongated
    const int NP = 6 + (int)(U(rng) * 11);
    const float k = 8.f + 22.f * U(rng);                     // edge softness (lower = more worn)
    const float jitter = 0.3f + 0.6f * U(rng), depthVar = 0.15f + 0.3f * U(rng);
    const float bedAmt = U(rng) < 0.5f ? 0.f : 0.01f + 0.03f * U(rng), bedFreq = 4.f + 6.f * U(rng);
    const float ridgeAmt = 0.02f + 0.07f * U(rng), lumpAmt = 0.05f + 0.15f * U(rng);
    const float sy = 0.45f + 0.45f * U(rng), sx = 0.8f + 0.5f * U(rng), sz = 0.8f + 0.3f * U(rng);
    const V3 noiseOff{U(rng) * 100.f, U(rng) * 100.f, U(rng) * 100.f};
    // cleavage planes: spread over the sphere (golden spiral), jittered, at varied depths
    V3 pn[16]; float pd[16];
    for (int i = 0; i < NP; i++) {
        float y = 1.f - 2.f * (i + 0.5f) / NP, r = sqrtf(1.f - y * y), a = i * 2.39996f;
        pn[i] = normalize(V3{cosf(a) * r + (U(rng) - 0.5f) * jitter, y + (U(rng) - 0.5f) * jitter, sinf(a) * r + (U(rng) - 0.5f) * jitter});
        pd[i] = 1.f - depthVar * U(rng);
    }
    V3 bed = normalize(V3{U(rng) - 0.5f, 1.f, U(rng) - 0.5f});  // strata axis
    V3 jn[3]; float jd[3];
    for (int i = 0; i < 3; i++) { jn[i] = normalize(V3{U(rng) - 0.5f, U(rng) - 0.5f, U(rng) - 0.5f}); jd[i] = (U(rng) - 0.5f) * 0.9f; }
    for (auto& p : v) {
        V3 d = normalize(p);
        float sum = expf(-k * 1.2f);
        for (int i = 0; i < NP; i++) sum += expf(-k * pd[i] / std::max(dot(d, pn[i]), 0.06f));
        float r = -logf(sum) / k;
        V3 q = d * r;
        q = q + noiseOff;
        float s = dot(q, bed) * bedFreq + 0.6f * rockNoise(q * 2.f);
        float f = s - floorf(s);
        r -= bedAmt * rockSmooth(0.75f, 1.f, f);  // a step at each bedding plane
        for (int i = 0; i < 3; i++) {
            float w = fabsf(dot(q - noiseOff, jn[i]) - jd[i] - 0.04f * (rockNoise(q * 5.f + V3{3, 1, 7}) - 0.5f));
            r -= 0.015f * std::max(0.f, 1.f - w / 0.015f) * rockSmooth(0.2f, 0.6f, rockNoise(q * 1.5f + V3{(float)i * 9, 0, 0}));
        }
        float ridged = 0.f, amp = 0.5f, fr = 3.f;
        for (int o = 0; o < 4; o++, amp *= 0.5f, fr *= 2.1f) ridged += amp * (1.f - fabsf(2.f * rockNoise(q * fr + V3{(float)o * 17, 5, 3}) - 1.f));
        r += ridgeAmt * (ridged - 0.5f) + lumpAmt * (rockNoise(q * 1.2f + V3{11, 3, 5}) - 0.5f);
        p = d * r;
        p.x *= sx; p.z *= sz;
        p.y = p.y * sy + 0.1f;
        if (p.y < -0.3f) p.y = -0.3f + (p.y + 0.3f) * 0.35f;  // sat flat in the ground
    }
    std::vector<V3> n(v.size());
    for (size_t i = 0; i < idx.size(); i += 3) {
        V3 fn = cross(v[idx[i + 1]] - v[idx[i]], v[idx[i + 2]] - v[idx[i]]);
        for (int k = 0; k < 3; k++) n[idx[i + k]] = n[idx[i + k]] + fn;
    }
    for (size_t i = 0; i < v.size(); i++) {
        V3 nn = normalize(n[i]);
        verts.insert(verts.end(), {v[i].x, v[i].y, v[i].z, nn.x, nn.y, nn.z});
    }
}

static void buildScene() {
    const VkBufferUsageFlags geomUsage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                         VK_BUFFER_USAGE_VERTEX_BUFFER_BIT | VK_BUFFER_USAGE_INDEX_BUFFER_BIT;
    // terrain: vertices are generated on the GPU around the camera (terrain.comp); topology is fixed
    std::vector<uint32_t> tidx;
    for (uint32_t j = 0; j + 1 < TNV; j++)
        for (uint32_t i = 0; i + 1 < TNV; i++) {
            uint32_t v00 = j * TNV + i, v10 = v00 + 1, v01 = v00 + TNV, v11 = v01 + 1;
            tidx.insert(tidx.end(), {v00, v01, v10, v10, v01, v11});
        }
    terrainTris = (uint32_t)tidx.size() / 3;
    terrainI = uploadBuf(tidx.data(), tidx.size() * 4, geomUsage);
    terrainV = mkBuf((VkDeviceSize)TNV * TNV * 24, geomUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // grass: per-frame GPU generated; fixed 7-vertex / 5-triangle topology per slot
    std::vector<uint32_t> gidx;
    gidx.reserve((size_t)maxBlades * 15);
    const uint32_t pat[15] = {0, 1, 2, 1, 3, 2, 2, 3, 4, 3, 5, 4, 4, 5, 6};
    for (uint32_t b = 0; b < maxBlades; b++)
        for (uint32_t k : pat) gidx.push_back(b * 7 + k);
    grassI = uploadBuf(gidx.data(), gidx.size() * 4, geomUsage);
    grassV = mkBuf((VkDeviceSize)maxBlades * 7 * 3 * 4, geomUsage, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    bladeBuf = mkBuf((VkDeviceSize)maxBlades * 12,  // seed + cell xy per slot (grass.comp)
                     VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    grassCounter = mkBuf(64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_INDIRECT_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                 VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    countReadback = mkBuf(16, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    // rocks
    std::vector<float> rverts;
    std::vector<uint32_t> ridx;
    for (int v = 0; v < ROCK_VARIANTS; v++) buildRockMesh((uint32_t)v, rverts, ridx);  // same topology: ridx is rebuilt identically
    rockVerts = (uint32_t)rverts.size() / 6 / ROCK_VARIANTS;
    rockPts = rverts;
    rockTris = (uint32_t)ridx.size() / 3;
    rockV = uploadBuf(rverts.data(), rverts.size() * 4, geomUsage);
    rockI = uploadBuf(ridx.data(), ridx.size() * 4, geomUsage);
    rockXf = mkBuf((MAX_ROCKS + MAX_THROWN) * 128,  // 4 rows each (3x4 transform + variant base vertex);  // second half: previous-frame transforms
 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // sky-view LUT
    skyImg = mkImage(SKY_W, SKY_H, VK_FORMAT_R16G16B16A16_SFLOAT);
    oneShot([](VkCommandBuffer cb) {
        imgBarrier(cb, skyImg.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                   VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT);
    });
    // wind map (256 x 256 cells around the camera, velocity + pressure)
    for (auto& w : windImg) {
        w = mkImage(256, 256, VK_FORMAT_R16G16B16A16_SFLOAT);
        oneShot([&](VkCommandBuffer cb) {
            imgBarrier(cb, w.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            VkClearColorValue z{{0, 0, 0, 0}};
            VkImageSubresourceRange rr{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cb, w.img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rr);
        });
    }

    // world map + simulation grid
    mapHImg = mkImage(worldmap::N, worldmap::N, VK_FORMAT_R32_SFLOAT);
    mapPaintImg = mkImage(worldmap::N, worldmap::N, VK_FORMAT_R8G8B8A8_UNORM);
    mapWaterImg = mkImage(worldmap::N / 2, worldmap::N / 2, VK_FORMAT_R32G32B32A32_SFLOAT);
    simImg = mkImage(worldmap::N, worldmap::N, VK_FORMAT_R16G16B16A16_SFLOAT);
    fluxWImg = mkImage(worldmap::N, worldmap::N, VK_FORMAT_R16G16B16A16_SFLOAT);
    fluxSImg = mkImage(worldmap::N, worldmap::N, VK_FORMAT_R16G16B16A16_SFLOAT);
    simGroundImg = mkImage(worldmap::N, worldmap::N, VK_FORMAT_R32_SFLOAT);
    vtImgA = mkImageArray(VT_N, VT_N, VT_L, VK_FORMAT_R8G8B8A8_UNORM);
    vtImgB = mkImageArray(VT_N, VT_N, VT_L, VK_FORMAT_R8G8B8A8_UNORM);
    vtImgH = mkImageArray(VT_N, VT_N, VT_L, VK_FORMAT_R16_UNORM);
    for (Image* im : {&vtImgA, &vtImgB, &vtImgH})
        oneShot([&](VkCommandBuffer cb) {
            VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
            b.oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
            b.newLayout = VK_IMAGE_LAYOUT_GENERAL;
            b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
            b.image = im->img;
            b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, VT_L};
            b.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
            vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        });
    vtTileBuf = mkBuf(4096 * sizeof(VtTileReq), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    craterBinBuf = mkBuf((2 * CB_BINS + CB_LIST) * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memset(craterBinBuf.map, 0, (2 * CB_BINS + CB_LIST) * 4);
    simRB = mkBuf(SIM_RB * SIM_RB * 8, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memset(simRB.map, 0, SIM_RB * SIM_RB * 8);
    for (Image* im : {&mapHImg, &mapPaintImg, &mapWaterImg, &simImg, &fluxWImg, &fluxSImg, &simGroundImg})
        oneShot([&](VkCommandBuffer cb) {
            imgBarrier(cb, im->img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            VkClearColorValue z{{0, 0, 0, 0}};
            if (im == &mapPaintImg) z = {{0.5f, 0.5f, 0, 0}};  // 1x grass and trees
            if (im == &mapWaterImg) z = {{-1000.f, 0, 0, 0}};
            VkImageSubresourceRange rr{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cb, im->img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rr);
        });

    // lights: drift in a square tile (S.orbRange wide) that wraps around the camera; positions in tile units
    std::mt19937 rng(1337);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    const V3 palette[6] = {{1, 0.25f, 0.6f}, {0.2f, 0.8f, 1}, {1, 0.6f, 0.15f}, {0.4f, 1, 0.3f}, {0.6f, 0.3f, 1}, {1, 0.9f, 0.7f}};
    for (int i = 0; i < MAX_ORBS; i++)
        lightBase.push_back({{U(rng) - 0.5f, 0.6f + U(rng) * 1.4f, U(rng) - 0.5f}, U(rng) * 100.f, palette[i % 6], 0.6f + 0.8f * U(rng)});
    lightBuf = mkBuf(MAX_LIGHTS * 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    uboBuf = mkBuf(sizeof(UBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
                   VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);

    // streamed model tables (written from the CPU only for slots no in-flight frame uses)
    const VkMemoryPropertyFlags hv = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    meshInfoBuf = mkBuf(MAX_MESHES * 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hv, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    triLightBuf = mkBuf(MAX_TRI_LIGHTS * 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hv, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    lightAliasBuf = mkBuf((MAX_LIGHTS + MAX_TRI_LIGHTS) * sizeof(lighttree::AliasEntry), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hv, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    lightTreeBuf = mkBuf((MAX_LIGHTS + MAX_TRI_LIGHTS) * 2 * sizeof(lighttree::Node), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hv, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    {
        std::vector<float> bn;
        lighttree::blueNoise(BLUE_NOISE, bn);
        blueNoiseBuf = mkBuf(bn.size() * 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hv, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        memcpy(blueNoiseBuf.map, bn.data(), bn.size() * 4);
    }
    materialBuf = mkBuf(MAX_MATERIALS * sizeof(GpuMat), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hv, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    modelXfBuf = mkBuf(MAX_MODEL_INST * 128,  // second half: each instance's previous-frame transform (motion vectors)
 VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hv, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    zoneBuf = mkBuf(MAX_ZONES * 32, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hv, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    memset(meshInfoBuf.map, 0, MAX_MESHES * 32);
    volBuf = mkBuf(MAX_VOLUMES * 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hv, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    memset(volBuf.map, 0, MAX_VOLUMES * 64);
    probeBuf = mkBuf((VkDeviceSize)MAX_PROBES * 2 * 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                     VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    probeTmp = mkBuf((VkDeviceSize)MAX_PROBES * 64, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    radCacheBuf = mkBuf((VkDeviceSize)RC_ENTRIES * RC_STRIDE, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    oneShot([&](VkCommandBuffer cb) { vkCmdFillBuffer(cb, radCacheBuf.b, 0, VK_WHOLE_SIZE, 0); });
    impBuf = mkBuf(MAX_TREES * 48, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, hv, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    for (int k = 0; k < 2; k++) {
        VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
        ci.imageType = VK_IMAGE_TYPE_2D;
        ci.format = k == 0 ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_R16G16B16A16_SFLOAT;
        ci.extent = {1024, 1024, 1};  // IMP_ATLAS
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
    memset(zoneBuf.map, 0, MAX_ZONES * 32);

    VkSamplerCreateInfo sci{VK_STRUCTURE_TYPE_SAMPLER_CREATE_INFO};
    sci.magFilter = sci.minFilter = VK_FILTER_LINEAR;
    sci.mipmapMode = VK_SAMPLER_MIPMAP_MODE_LINEAR;
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_REPEAT;
    sci.anisotropyEnable = VK_TRUE;
    sci.maxAnisotropy = 8.f;
    sci.maxLod = VK_LOD_CLAMP_NONE;
    VK_CHECK(vkCreateSampler(dev, &sci, nullptr, &texSampler));
    sci.addressModeU = sci.addressModeV = sci.addressModeW = VK_SAMPLER_ADDRESS_MODE_CLAMP_TO_EDGE;
    sci.anisotropyEnable = VK_FALSE;
    VK_CHECK(vkCreateSampler(dev, &sci, nullptr, &postSampler));
    exposureBuf = mkBuf(128 * 4 + 16, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memset(exposureBuf.map, 0, 128 * 4 + 16);
}

// ---------------------------------------------------------------- acceleration structures
static VkAccelerationStructureGeometryKHR triGeom(const Buffer& vb, uint32_t stride, uint32_t nv, const Buffer& ib) {
    VkAccelerationStructureGeometryKHR g{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    g.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    g.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    auto& t = g.geometry.triangles;
    t.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    t.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    t.vertexData.deviceAddress = vb.addr;
    t.vertexStride = stride;
    t.maxVertex = nv - 1;
    t.indexType = VK_INDEX_TYPE_UINT32;
    t.indexData.deviceAddress = ib.addr;
    return g;
}

static VkAccelerationStructureBuildGeometryInfoKHR buildInfo(VkAccelerationStructureTypeKHR type, VkBuildAccelerationStructureFlagsKHR flags,
                                                            const VkAccelerationStructureGeometryKHR* g) {
    VkAccelerationStructureBuildGeometryInfoKHR bi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    bi.type = type;
    bi.flags = flags;
    bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bi.geometryCount = 1;
    bi.pGeometries = g;
    return bi;
}

static Accel mkAccel(VkAccelerationStructureTypeKHR type, VkBuildAccelerationStructureFlagsKHR flags, const VkAccelerationStructureGeometryKHR* g, uint32_t prims) {
    auto bi = buildInfo(type, flags, g);
    VkAccelerationStructureBuildSizesInfoKHR sz{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    pGetASSizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &prims, &sz);
    Accel a;
    a.size = sz.accelerationStructureSize;
    a.buf = mkBuf(sz.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkAccelerationStructureCreateInfoKHR ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    ci.buffer = a.buf.b;
    ci.size = sz.accelerationStructureSize;
    ci.type = type;
    VK_CHECK(pCreateAS(dev, &ci, nullptr, &a.as));
    VkAccelerationStructureDeviceAddressInfoKHR ai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    ai.accelerationStructure = a.as;
    a.addr = pGetASAddr(dev, &ai);
    VkDeviceSize align = asProps.minAccelerationStructureScratchOffsetAlignment;
    a.scratch = mkBuf(std::max(sz.buildScratchSize, sz.updateScratchSize) + align, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    a.scratchAddr = (a.scratch.addr + align - 1) & ~(align - 1);
    return a;
}

static void cmdBuild(VkCommandBuffer cb, Accel& a, VkAccelerationStructureTypeKHR type, VkBuildAccelerationStructureFlagsKHR flags,
                     const VkAccelerationStructureGeometryKHR* g, uint32_t prims, bool update) {
    auto bi = buildInfo(type, flags, g);
    if (update) {
        bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        bi.srcAccelerationStructure = a.as;
    }
    bi.dstAccelerationStructure = a.as;
    bi.scratchData.deviceAddress = a.scratchAddr;
    VkAccelerationStructureBuildRangeInfoKHR r{prims, 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR* pr = &r;
    pCmdBuildAS(cb, 1, &bi, &pr);
}

static void setInstance(VkAccelerationStructureInstanceKHR& in, uint32_t customIndex, VkDeviceAddress blas, const float m[3][4]) {
    in = {};
    memcpy(in.transform.matrix, m, sizeof in.transform.matrix);
    in.instanceCustomIndex = customIndex; // material id in shaders
    in.mask = customIndex == 0 ? 0x02 : 0x01;  // terrain on its own bit: crater rays can leave it out (common.glsl visibleEx)
    in.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR | VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
    in.accelerationStructureReference = blas;
}

// A rock instance: TLAS entry (the variant's BLAS; its base vertex in the SBT record offset, which ray queries
// don't use otherwise, so hit shading finds its normals in rv) and the raster transform (rockXf slot, 4 rows).
static uint32_t rockVarOf(float x, float z) { return (uint32_t)(hash2((int)lroundf(x * 8.f) + 71, (int)lroundf(z * 8.f) + 313) * ROCK_VARIANTS) % ROCK_VARIANTS; }
static void setRockInstance(VkAccelerationStructureInstanceKHR& in, uint32_t var, const float m[3][4]) {
    setInstance(in, 2, rockAS[var].addr, m);
    in.instanceShaderBindingTableRecordOffset = var * rockVerts;
}
static void writeRockXf(float* xf, uint32_t slot, const float m[3][4], const float prev[3][4], uint32_t var) {
    for (int half = 0; half < 2; half++) {
        float* o = xf + (half ? (MAX_ROCKS + MAX_THROWN) * 16 : 0) + slot * 16;
        memcpy(o, half ? prev : m, 48);
        uint32_t base = var * rockVerts;
        memcpy(o + 12, &base, 4);
        o[13] = o[14] = o[15] = 0.f;
    }
}

static void createAccels() {
    terrainGeom = triGeom(terrainV, 24, TNV * TNV, terrainI);
    grassGeom = triGeom(grassV, 12, maxBlades * 7, grassI);
    for (int v = 0; v < ROCK_VARIANTS; v++) {
        rockGeom[v] = triGeom(rockV, 24, rockVerts, rockI);
        rockGeom[v].geometry.triangles.vertexData.deviceAddress = rockV.addr + (VkDeviceSize)v * rockVerts * 24;
    }
    terrainAS = mkAccel(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR, &terrainGeom, terrainTris);
    grassAS = mkAccel(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, grassFlags, &grassGeom, maxBlades * 5);
    for (int v = 0; v < ROCK_VARIANTS; v++)
        rockAS[v] = mkAccel(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR, &rockGeom[v], rockTris);
    oneShot([](VkCommandBuffer cb) {
        for (int v = 0; v < ROCK_VARIANTS; v++)
            cmdBuild(cb, rockAS[v], VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
                     &rockGeom[v], rockTris, false);
    });

    instBuf = mkBuf((INST_MODELS + MAX_MODEL_INST + MAX_ROCKS + MAX_TREES + MAX_PLANTS + MAX_THROWN + MAX_SPRAY + MAX_FALLEN + MAX_CHAR_INST) * sizeof(VkAccelerationStructureInstanceKHR), VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
                    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    auto* inst = (VkAccelerationStructureInstanceKHR*)instBuf.map;
    const float id[3][4] = {{1, 0, 0, 0}, {0, 1, 0, 0}, {0, 0, 1, 0}};
    setInstance(inst[0], 0, terrainAS.addr, id);
    setInstance(inst[1], 1, grassAS.addr, id);

    tlasGeom = VkAccelerationStructureGeometryKHR{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    tlasGeom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
    tlasGeom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    tlasGeom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
    tlasGeom.geometry.instances.data.deviceAddress = instBuf.addr;
    tlas = mkAccel(VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR, &tlasGeom,
                   INST_MODELS + MAX_MODEL_INST + MAX_ROCKS + MAX_TREES + MAX_PLANTS + MAX_THROWN + MAX_SPRAY + MAX_FALLEN + MAX_CHAR_INST);
}

static void cmdBuildScene(VkCommandBuffer cb) {
    if (terrainDirty)
        cmdBuild(cb, terrainAS, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR,
                 &terrainGeom, terrainTris, false);
    cmdBuild(cb, grassAS, VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, grassFlags, &grassGeom, bladeCap * 5, grassRefit);
    grassBuilt = true;
    memBarrier(cb, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
               VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);
    cmdBuild(cb, tlas, VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR, &tlasGeom, INST_MODELS + numModelInst + numRocks + numTrees + numPlants + numThrown + numSprayInst + numFallen + numCharInst, false);
    memBarrier(cb, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR | VK_ACCESS_SHADER_READ_BIT);
}

// ---------------------------------------------------------------- streamed models: upload / release / instances
static uint32_t mipCount(uint32_t w, uint32_t h) { return 32 - (uint32_t)__builtin_clz(std::max(w, h)); }

static Image mkTexture(uint32_t w, uint32_t h, uint32_t mips, VkFormat fmt = VK_FORMAT_R8G8B8A8_SRGB) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = fmt;
    ci.extent = {w, h, 1};
    ci.mipLevels = mips;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    return mkImageCI(ci, VK_IMAGE_ASPECT_COLOR_BIT);
}

static void mipBarrier(VkCommandBuffer cb, VkImage img, uint32_t base, uint32_t count, VkImageLayout from, VkImageLayout to, VkAccessFlags sa,
                       VkAccessFlags da, VkPipelineStageFlags ds = VK_PIPELINE_STAGE_TRANSFER_BIT) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, base, count, 0, 1};
    b.srcAccessMask = sa;
    b.dstAccessMask = da;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, ds, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// copy level 0 from staging, build the mip chain with blits, leave it shader-readable
static void recordTextureUpload(VkCommandBuffer cb, const Image& im, uint32_t w, uint32_t h, uint32_t mips, VkBuffer src, VkDeviceSize off) {
    mipBarrier(cb, im.img, 0, mips, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkBufferImageCopy r{};
    r.bufferOffset = off;
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent = {w, h, 1};
    vkCmdCopyBufferToImage(cb, src, im.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    for (uint32_t i = 1; i < mips; i++) {
        mipBarrier(cb, im.img, i - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_ACCESS_TRANSFER_READ_BIT);
        VkImageBlit bl{};
        bl.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1};
        bl.srcOffsets[1] = {(int32_t)std::max(1u, w >> (i - 1)), (int32_t)std::max(1u, h >> (i - 1)), 1};
        bl.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1};
        bl.dstOffsets[1] = {(int32_t)std::max(1u, w >> i), (int32_t)std::max(1u, h >> i), 1};
        vkCmdBlitImage(cb, im.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, im.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_LINEAR);
    }
    const VkPipelineStageFlags shaders = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    if (mips > 1)
        mipBarrier(cb, im.img, 0, mips - 1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
                   VK_ACCESS_SHADER_READ_BIT, shaders);
    mipBarrier(cb, im.img, mips - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
               VK_ACCESS_SHADER_READ_BIT, shaders);
}

// BC7 texture: every mip level comes precomputed (texcomp.h), so the upload is copies only
static void recordCompressedUpload(VkCommandBuffer cb, const Image& im, const TextureData& t, VkBuffer src, VkDeviceSize off) {
    mipBarrier(cb, im.img, 0, t.bcMips, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    std::vector<VkBufferImageCopy> rs(t.bcMips);
    for (uint32_t l = 0; l < t.bcMips; l++) {
        rs[l] = {};
        rs[l].bufferOffset = off + t.bcLevels[l];
        rs[l].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, 1};
        rs[l].imageExtent = {std::max(1u, t.w >> l), std::max(1u, t.h >> l), 1};
    }
    vkCmdCopyBufferToImage(cb, src, im.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, t.bcMips, rs.data());
    mipBarrier(cb, im.img, 0, t.bcMips, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
               VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
}

static void writeTexDescriptor(uint32_t slot, VkImageView view) {
    VkDescriptorImageInfo ii{texSampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = descSet;
    w.dstBinding = 32;
    w.dstArrayElement = slot;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &ii;
    vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
}

static void beginUploads() {
    if (uploadsRecorded) return;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(uploadCmd, 0);
    vkBeginCommandBuffer(uploadCmd, &bi);
    uploadsRecorded = true;
}

template <class F> static GpuMat gpuMaterial(const MaterialData& mt, F slot) {
    GpuMat g{};
    memcpy(g.color, mt.color, sizeof g.color);
    memcpy(g.emissive, mt.emissive, sizeof g.emissive);
    g.tex = slot(mt.tex);
    g.rough = mt.rough; g.metal = mt.metal; g.normalScale = mt.normalScale; g.nrmTex = slot(mt.normalTex);
    g.ormTex = slot(mt.ormTex); g.emisTex = slot(mt.emisTex); g.occTex = slot(mt.occTex); g.occStrength = mt.occStrength;
    g.heightScale = mt.normalTex >= 0 ? mt.heightScale : 0.f;
    g.clearcoat = mt.clearcoat; g.ccRough = mt.clearcoatRough; g.sheen = mt.sheen;
    memcpy(g.sheenColor, mt.sheenColor, sizeof g.sheenColor);
    g.aniso = mt.anisotropy; g.anisoRot = mt.anisoRot; g.transmission = mt.transmission; g.ior = mt.ior; g.specular = mt.specular;
    g.transl = mt.translucency;
    return g;
}
static bool isGlass(const MaterialData& m) { return m.transmission > 0.5f; }

static size_t frameUploadBytes = 0;

// Model CPU data -> one geometry buffer, one BLAS buffer (a BLAS per mesh), mip-mapped textures, bindless slots.
// Recorded into uploadCmd; returns false once this frame's upload budget is spent.
static bool uploadModel(Model& m) {
    PROF_FUNC();
    const ModelData& d = m.data;
    if (meshSlots.cap - meshSlots.used() < d.meshes.size() || matSlots.cap - matSlots.used() < d.materials.size() ||
        texSlots.cap - texSlots.used() < d.textures.size()) {
        static bool warned = false;
        if (!warned) fprintf(stderr, "stream: out of mesh/material/texture slots, %s not uploaded\n", m.path.c_str());
        warned = true;
        return false;
    }
    beginUploads();
    VkCommandBuffer cb = uploadCmd;
    auto* g = new GpuModel;

    // slots: textures, then materials (pointing at texture slots), then per-triangle material slots
    for (auto& t : d.textures) {
        g->texs.push_back(texSlots.alloc());
        if (!t.bc7.empty()) g->images.push_back(mkTexture(t.w, t.h, t.bcMips, t.linear ? VK_FORMAT_BC7_UNORM_BLOCK : VK_FORMAT_BC7_SRGB_BLOCK));
        else g->images.push_back(mkTexture(t.w, t.h, mipCount(t.w, t.h), t.linear ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_R8G8B8A8_SRGB));
    }
    for (auto& mt : d.materials) {
        uint32_t slot = matSlots.alloc();
        GpuMat gm = gpuMaterial(mt, [&](int t) { return t >= 0 && t < (int)g->texs.size() ? g->texs[t] : 0xFFFFFFFFu; });
        memcpy((char*)materialBuf.map + slot * sizeof(GpuMat), &gm, sizeof gm);
        g->mats.push_back(slot);
    }

    // layout: per mesh [verts | indices | triangle material slots], 16-byte aligned
    auto al = [](VkDeviceSize v, VkDeviceSize a) { return (v + a - 1) & ~(a - 1); };
    std::vector<VkDeviceSize> vOff(d.meshes.size()), iOff(d.meshes.size()), tOff(d.meshes.size());
    VkDeviceSize geomSize = 0;
    for (size_t i = 0; i < d.meshes.size(); i++) {
        auto& md = d.meshes[i];
        vOff[i] = geomSize = al(geomSize, 16); geomSize += md.verts.size() * 4;
        iOff[i] = geomSize = al(geomSize, 16); geomSize += md.idx.size() * 4;
        tOff[i] = geomSize = al(geomSize, 16); geomSize += md.triMat.size() * 4;
    }
    geomSize = std::max<VkDeviceSize>(al(geomSize, 16), 16);
    VkDeviceSize texSize = 0;
    std::vector<VkDeviceSize> texOff;
    auto texBytes = [](const TextureData& t) -> const std::vector<uint8_t>& { return t.bc7.empty() ? t.rgba : t.bc7; };
    for (auto& t : d.textures) { texOff.push_back(geomSize + texSize); texSize += al(texBytes(t).size(), 16); }

    Buffer st = mkBuf(geomSize + texSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    char* sp = (char*)st.map;
    for (size_t i = 0; i < d.meshes.size(); i++) {
        auto& md = d.meshes[i];
        memcpy(sp + vOff[i], md.verts.data(), md.verts.size() * 4);
        memcpy(sp + iOff[i], md.idx.data(), md.idx.size() * 4);
        uint32_t* tm = (uint32_t*)(sp + tOff[i]);
        for (size_t t = 0; t < md.triMat.size(); t++) tm[t] = g->mats[md.triMat[t]];
    }
    for (size_t i = 0; i < d.textures.size(); i++) memcpy(sp + texOff[i], texBytes(d.textures[i]).data(), texBytes(d.textures[i]).size());

    g->geom = mkBuf(geomSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                  VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkBufferCopy cp{0, 0, geomSize};
    vkCmdCopyBuffer(cb, st.b, g->geom.b, 1, &cp);
    for (size_t i = 0; i < d.textures.size(); i++) {
        auto& t = d.textures[i];
        if (!t.bc7.empty()) recordCompressedUpload(cb, g->images[i], t, st.b, texOff[i]);
        else recordTextureUpload(cb, g->images[i], t.w, t.h, mipCount(t.w, t.h), st.b, texOff[i]);
        writeTexDescriptor(g->texs[i], g->images[i].view);
    }
    memBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
               VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT);

    // one BLAS per mesh, all in one buffer and one build call
    size_t n = d.meshes.size();
    g->meshes.resize(n);
    std::vector<VkAccelerationStructureGeometryKHR> geoms(n);
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> infos;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
    std::vector<size_t> which;
    std::vector<VkDeviceSize> asOff, asSize, scOff;
    VkDeviceSize asTotal = 0, scTotal = 0, scAlign = asProps.minAccelerationStructureScratchOffsetAlignment;
    for (size_t i = 0; i < n; i++) {
        auto& md = d.meshes[i];
        uint32_t tris = (uint32_t)md.idx.size() / 3;
        if (!tris) continue;
        auto& gg = geoms[i];
        gg = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        gg.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        // meshes with glass are non-opaque: shadow rays (common.glsl visible()) pass through the glass triangles
        bool glass = false;
        for (uint32_t tm : md.triMat) glass |= tm < d.materials.size() && isGlass(d.materials[tm]);
        gg.flags = glass ? 0 : VK_GEOMETRY_OPAQUE_BIT_KHR;
        auto& t = gg.geometry.triangles;
        t.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        t.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        t.vertexData.deviceAddress = g->geom.addr + vOff[i];
        t.vertexStride = 32;
        t.maxVertex = (uint32_t)(md.verts.size() / 8) - 1;
        t.indexType = VK_INDEX_TYPE_UINT32;
        t.indexData.deviceAddress = g->geom.addr + iOff[i];
        auto bi = buildInfo(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR, &gg);
        VkAccelerationStructureBuildSizesInfoKHR sz{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        pGetASSizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &tris, &sz);
        asOff.push_back(asTotal = al(asTotal, 256));
        asSize.push_back(sz.accelerationStructureSize);
        asTotal += sz.accelerationStructureSize;
        scOff.push_back(scTotal = al(scTotal, scAlign));
        scTotal += sz.buildScratchSize;
        infos.push_back(bi);
        ranges.push_back({tris, 0, 0, 0});
        which.push_back(i);
    }
    Buffer scratch;
    if (!infos.empty()) {
        g->blasMem = mkBuf(asTotal, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        scratch = mkBuf(scTotal + scAlign, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceAddress scBase = al(scratch.addr, scAlign);
        std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> pr;
        for (size_t k = 0; k < infos.size(); k++) {
            GpuMesh& gm = g->meshes[which[k]];
            VkAccelerationStructureCreateInfoKHR ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
            ci.buffer = g->blasMem.b;
            ci.offset = asOff[k];
            ci.size = asSize[k];
            ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            VK_CHECK(pCreateAS(dev, &ci, nullptr, &gm.blas));
            VkAccelerationStructureDeviceAddressInfoKHR ai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
            ai.accelerationStructure = gm.blas;
            gm.blasAddr = pGetASAddr(dev, &ai);
            infos[k].pGeometries = &geoms[which[k]];
            infos[k].dstAccelerationStructure = gm.blas;
            infos[k].scratchData.deviceAddress = scBase + scOff[k];
            pr.push_back(&ranges[k]);
        }
        pCmdBuildAS(cb, (uint32_t)infos.size(), infos.data(), pr.data());
    }
    memBarrier(cb, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
               VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
               VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);

    // emissive triangles (radiance = factor x the emissive texture's mean colour) for the light list
    std::vector<std::array<float, 3>> emisAvg(d.materials.size(), {1, 1, 1});
    for (size_t k = 0; k < d.materials.size(); k++) {
        int t = d.materials[k].emisTex;
        if (t < 0 || t >= (int)d.textures.size() || d.textures[t].rgba.empty()) continue;
        double sum[3] = {0, 0, 0};
        const auto& px = d.textures[t].rgba;
        size_t np = px.size() / 4, step = std::max<size_t>(1, np / 4096);
        size_t cnt = 0;
        for (size_t q = 0; q < np; q += step, cnt++)
            for (int c = 0; c < 3; c++) { float v = px[q * 4 + c] / 255.f; sum[c] += v <= 0.04045f ? v / 12.92f : powf((v + 0.055f) / 1.055f, 2.4f); }
        for (int c = 0; c < 3; c++) emisAvg[k][c] = (float)(sum[c] / std::max<size_t>(cnt, 1));
    }
    for (size_t i = 0; i < n; i++) {
        const MeshData& md = d.meshes[i];
        for (size_t t = 0; t < md.triMat.size(); t++) {
            uint32_t mi = md.triMat[t];
            if (mi >= d.materials.size()) continue;
            const MaterialData& mt = d.materials[mi];
            float Le[3] = {mt.emissive[0] * emisAvg[mi][0], mt.emissive[1] * emisAvg[mi][1], mt.emissive[2] * emisAvg[mi][2]};
            if (Le[0] + Le[1] + Le[2] <= 0.f) continue;
            EmisTri e;
            for (int v = 0; v < 3; v++) memcpy(e.v[v], &md.verts[md.idx[t * 3 + v] * 8], 12);
            memcpy(e.Le, Le, 12);
            g->meshes[i].emis.push_back(e);
        }
    }
    for (size_t i = 0; i < n; i++) {
        GpuMesh& gm = g->meshes[i];
        if (!gm.blas) continue;
        gm.slot = meshSlots.alloc();
        gm.indexCount = (uint32_t)d.meshes[i].idx.size();
        gm.firstIndex = (uint32_t)(iOff[i] / 4);  // raster binds the whole geometry buffer as the index buffer
        VkDeviceAddress info[4] = {g->geom.addr + vOff[i], g->geom.addr + iOff[i], g->geom.addr + tOff[i], 0};
        memcpy((char*)meshInfoBuf.map + gm.slot * 32, info, 32);
    }
    retireNext.push_back([st, scratch]() mutable { freeBuf(st); freeBuf(scratch); });
    for (auto& md : d.meshes)
        g->audioMeshes.push_back(audio::meshCreate(md.verts.data(), 8, (uint32_t)(md.verts.size() / 8), md.idx.data(),
                                                   (uint32_t)(md.idx.size() / 3), audio::MAT_CONCRETE));
    m.gpu = g;
    g->bytes = geomSize + asTotal + texSize * 4 / 3;  // textures + mips
    gpuModelBytes += g->bytes;
    frameUploadBytes += geomSize + texSize;
    return frameUploadBytes < (size_t)(uploadBudgetMB * 1048576.f);
}

static std::vector<uint32_t> audioMeshRetire;
// acoustic instances keyed by (mesh, placement); diffed on every stream change so only edits reach the audio scene
struct AudioKey {
    const void* mesh; float t[3], r0;
    bool operator==(const AudioKey& o) const { return !memcmp(this, &o, sizeof *this); }
};
struct AudioKeyHash {
    size_t operator()(const AudioKey& k) const {
        size_t h = std::hash<const void*>()(k.mesh);
        for (float f : {k.t[0], k.t[1], k.t[2], k.r0}) h = h * 1000003u ^ std::hash<float>()(f);
        return h;
    }
};
static std::unordered_map<AudioKey, uint32_t, AudioKeyHash> audioInsts;

static void syncAudioInstances() {
    std::unordered_map<AudioKey, uint32_t, AudioKeyHash> next;
    stream::forEachPlaced([&](const Placed& p) {
        auto* g = (GpuModel*)p.model->gpu;
        if (!g || (size_t)p.mesh >= g->audioMeshes.size()) return;
        AudioKey k{&p.model->data.meshes[p.mesh], {p.xf.m[0][3], p.xf.m[1][3], p.xf.m[2][3]}, p.xf.m[0][0]};
        auto it = audioInsts.find(k);
        if (it != audioInsts.end()) { next.emplace(k, it->second); audioInsts.erase(it); }
        else next.emplace(k, audio::instanceAdd(g->audioMeshes[p.mesh], p.xf));
    });
    for (auto& [k, id] : audioInsts) audio::instanceRemove(id);
    audioInsts.swap(next);
}

static void releaseModel(Model& m) {
    auto* g = (GpuModel*)m.gpu;
    for (auto& gm : g->meshes) {
        if (gm.blas) pDestroyAS(dev, gm.blas, nullptr);
        if (gm.slot != UINT32_MAX) meshSlots.release(gm.slot);
    }
    for (uint32_t s : g->mats) matSlots.release(s);
    for (uint32_t s : g->texs) { writeTexDescriptor(s, defaultTex.view); texSlots.release(s); }
    for (auto& im : g->images) freeImage(im);
    // instances of these meshes are removed in onStreamChanged (same frame), release the meshes after that
    audioMeshRetire.insert(audioMeshRetire.end(), g->audioMeshes.begin(), g->audioMeshes.end());
    gpuModelBytes -= g->bytes;
    freeBuf(g->geom);
    freeBuf(g->blasMem);
    delete g;
    m.gpu = nullptr;
}

// ---------------------------------------------------------------- characters (character.h): GPU skinning + BLAS refit
// A character asset is uploaded once (materials, textures, indices, bind-pose vertices). Each character then owns,
// per skinned mesh, a vertex buffer that skin.comp writes every frame, two mesh slots (the skinned mesh, and the
// skinning source: bind verts | weights | matrices, see skin.comp) and a BLAS that is refit in place most frames
// and rebuilt every CHAR_REBUILD frames (refits degrade as limbs move far from where the BLAS was built).
static constexpr uint32_t CHAR_REBUILD = 30;
struct CharAsset { std::shared_ptr<Model> model; std::vector<Buffer> skinW; };  // skinW per mesh (empty: unskinned)
static std::map<std::string, CharAsset> charAssets;
struct GpuCharMesh {
    uint32_t mesh = 0, nv = 0, ntri = 0, slot = UINT32_MAX, srcSlot = UINT32_MAX, prevSlot = UINT32_MAX;
    Buffer verts, prevVerts;  // prevVerts: last frame's skinned vertices (copied before skinning; motion vectors)
    Accel blas;
    VkAccelerationStructureGeometryKHR geom{};
};
struct GpuChar { CharAsset* asset = nullptr; Buffer mats; std::vector<GpuCharMesh> meshes; uint32_t frames = 0; Xf prevW{}; bool hasPrev = false; };
struct CharSpawn { std::string asset, clip; float pos[3], yaw, height, walk; };
static std::vector<CharSpawn> charSpawns;  // processed in the upload window
static std::string charWorldDir;

static CharAsset* charAsset(const std::string& name) {
    auto it = charAssets.find(name);
    if (it != charAssets.end()) return it->second.model ? &it->second : nullptr;
    CharAsset& a = charAssets[name];
    auto m = std::make_shared<Model>();
    m->path = assetFile(charWorldDir, name);
    std::string err;
    if (!loadGltf(m->path, m->data, err)) { fprintf(stderr, "character %s: %s\n", m->path.c_str(), err.c_str()); return nullptr; }
    if (m->data.skins.empty()) { fprintf(stderr, "character %s: no skin\n", m->path.c_str()); return nullptr; }
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

static void spawnCharacters() {
    for (const CharSpawn& sp : charSpawns) {
        CharAsset* a = charAsset(sp.asset);
        if (!a) continue;
        const ModelData& d = a->model->data;
        auto* g = (GpuModel*)a->model->gpu;
        float p[3] = {sp.pos[0], heightFn(sp.pos[0], sp.pos[2]), sp.pos[2]};
        character::Character* c = character::spawn(d, p, sp.yaw, sp.height);
        if (!c) continue;
        c->walkSpeed = sp.walk;
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

static void releaseCharacter(character::Character* c) {
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
static void writeCharacters() {
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

static void dispatch(VkCommandBuffer cb, int p, uint32_t gx, uint32_t gy, std::array<uint32_t, 4> pc);
// render graph: skin every character mesh, then build / refit its BLAS (before the scene's TLAS build)
static void recordSkinning(VkCommandBuffer cb) {
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

// Streamed placements -> TLAS instances (static, rewritten only on change) + SoA bounds for culling.
static void rebuildModelInstances(const std::vector<std::pair<const MeshData*, DrawItem>>& items) {
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

// Frustum-cull model instances (AVX2), write raster transforms, group consecutive same-mesh items into instanced draws.
static void frustumPlanes(float planes[5][4]) {
    float th = tanf(cam.fovDeg * 0.5f * 3.14159265f / 180.f), aspect = (float)W / H;
    V3 f = cam.fwd(), r = cam.right(), u = cam.up();
    V3 edges[4][2] = {{f + r * (th * aspect), u}, {f - r * (th * aspect), u}, {f + u * th, r}, {f - u * th, r}};
    for (int i = 0; i < 4; i++) {
        V3 n = normalize(cross(edges[i][0], edges[i][1]));
        if (dot(n, f) < 0) n = n * -1.f;  // orient inward regardless of handedness
        set4(planes[i], n, -dot(n, cam.pos));
    }
    set4(planes[4], f, -dot(f, cam.pos) + 0.05f);
}

// Previous-frame transform of raster instance k (second half of modelXfBuf): the vertex shader projects both and the
// difference goes into the G-buffer's velocity target. prevSlot: the mesh whose vertices it had (skinned meshes:
// last frame's skinned copy). Instances not given one get their current transform (static) in finishPrevXf.
static std::vector<uint8_t> prevSet(MAX_MODEL_INST, 0);
static void setPrevXf(uint32_t k, const float m[3][4], uint32_t prevSlot) {
    float* d = (float*)modelXfBuf.map + MAX_MODEL_INST * 16 + (size_t)k * 16;
    memcpy(d, m, 48);
    memcpy(d + 12, &prevSlot, 4);
    d[13] = d[14] = d[15] = 0;
    prevSet[k] = 1;
}
static void finishPrevXf() {
    const float* cur = (const float*)modelXfBuf.map;
    for (uint32_t k = 0; k < numModelVisible; k++) {
        if (!prevSet[k]) memcpy((float*)modelXfBuf.map + MAX_MODEL_INST * 16 + (size_t)k * 16, cur + (size_t)k * 16, 64);
        prevSet[k] = 0;
    }
}

static void cullModels() {
    PROF_FUNC();
    drawRuns.clear();
    numModelVisible = 0;
    if (!numModelInst || !S.rasterPrimary) return;
    float planes[5][4];
    frustumPlanes(planes);
    cullAabbs(aabb[0].data(), aabb[1].data(), aabb[2].data(), aabb[3].data(), aabb[4].data(), aabb[5].data(), numModelInst, planes, 5,
              visibleItems.data());
    float* xf = (float*)modelXfBuf.map;
    for (uint32_t i = 0; i < numModelInst; i++) {
        if (!visibleItems[i]) continue;
        const DrawItem& it = drawItems[i];
        uint32_t k = numModelVisible++;
        memcpy(xf + k * 16, it.xf.m, 48);
        uint32_t slot = it.m->meshes[it.mesh].slot;
        memcpy(xf + k * 16 + 12, &slot, 4);
        xf[k * 16 + 13] = xf[k * 16 + 14] = xf[k * 16 + 15] = 0;  // no impostor fade
        if (!drawRuns.empty() && drawRuns.back().m == it.m && drawRuns.back().mesh == it.mesh) drawRuns.back().count++;
        else drawRuns.push_back({it.m, it.mesh, k, 1});
    }
}

// ---------------------------------------------------------------- baked interiors
static bool inBox(const ProbeVolume& v, float x, float y, float z) {
    return x >= v.mn[0] && y >= v.mn[1] && z >= v.mn[2] && x < v.mx[0] && y < v.mx[1] && z < v.mx[2];
}
static int volumeAt(float x, float y, float z) {
    for (size_t i = 0; i < bakeVols.size(); i++) if (inBox(bakeVols[i].v, x, y, z)) return (int)i;
    return -1;
}
static uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    for (size_t i = 0; i < n; i++) h = (h ^ ((const uint8_t*)p)[i]) * 1099511628211ull;
    return h;
}
static std::string bakePath(const BakeVol& b, int st) {
    uint64_t h = fnv(b.v.mn, 24, fnv(b.dims, 12, fnv(&b.spacing, 4, bakeEnv ^ 5)));  // bump the xor to invalidate old caches
    char name[64];
    snprintf(name, sizeof name, "_%s_%016llx.bin", st ? "on" : "off", (unsigned long long)h);
    return bakeDir + b.v.name + name;
}
static uint64_t bakeEnvHash() {
    // version 2: emitters' direct light left to ReSTIR DI (lights.glsl emissionWeight) -> older bakes are redone
    float e[] = {S.sunElev, S.sunAzim, S.sunIntensity, S.sunRadiusDeg, S.skyIntensity, lightScale, 2.f};
    return fnv(e, sizeof e);
}
// Probe data of one volume + state: GPU <-> file (only called right after a frame fence)
static void bakeTransfer(BakeVol& b, int st, bool save) {
    VkDeviceSize size = (VkDeviceSize)b.count * 64, off = ((VkDeviceSize)b.first * 2 + (VkDeviceSize)st * b.count) * 64;
    Buffer stg = mkBuf(size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                       VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    std::string path = bakePath(b, st);
    if (save) {
        vkDeviceWaitIdle(dev);
        oneShot([&](VkCommandBuffer cb) { VkBufferCopy c{off, 0, size}; vkCmdCopyBuffer(cb, probeBuf.b, stg.b, 1, &c); });
        std::filesystem::create_directories(bakeDir);
        std::ofstream f(path, std::ios::binary);
        f.write((const char*)stg.map, (std::streamsize)size);
        b.saved[st] = true;
    } else {
        std::ifstream f(path, std::ios::binary);
        if (f && f.read((char*)stg.map, (std::streamsize)size)) {
            vkDeviceWaitIdle(dev);
            oneShot([&](VkCommandBuffer cb) { VkBufferCopy c{0, off, size}; vkCmdCopyBuffer(cb, stg.b, probeBuf.b, 1, &c); });
            b.ready[st] = b.saved[st] = true;
        }
    }
    freeBuf(stg);
}
// New set of volumes (stream change) or new lighting environment: lay out probe grids, load cached bakes.
static void rebuildBakeVolumes(bool envChanged) {
    const auto& vs = stream::volumes();
    bool same = !envChanged && vs.size() == bakeVols.size();
    for (size_t i = 0; same && i < vs.size(); i++) same = vs[i].name == bakeVols[i].v.name && !memcmp(vs[i].mn, bakeVols[i].v.mn, 24);
    if (same) return;
    std::vector<BakeVol> old;
    old.swap(bakeVols);
    // 25 cm probes; coarser everywhere if they don't fit
    double vol = 0;
    for (auto& v : vs) vol += (double)(v.mx[0] - v.mn[0]) * (v.mx[1] - v.mn[1]) * (v.mx[2] - v.mn[2]);
    float sp = std::max(0.25f, (float)cbrt(vol / (MAX_PROBES * 0.9)));
    uint32_t first = 0;
    for (auto& v : vs) {
        if (bakeVols.size() >= MAX_VOLUMES) break;
        BakeVol b;
        b.v = v;
        b.spacing = sp;
        b.on = !startLightsOff;
        for (int k = 0; k < 3; k++) b.dims[k] = std::max(1u, (uint32_t)ceilf((v.mx[k] - v.mn[k]) / sp));
        b.count = b.dims[0] * b.dims[1] * b.dims[2];
        if (first + b.count > MAX_PROBES) break;
        b.first = first;
        first += b.count;
        for (auto& o : old) if (o.v.name == v.name) b.on = o.on;  // keep switch states
        bakeVols.push_back(b);
    }
    for (auto& b : bakeVols) for (int st = 0; st < 2; st++) bakeTransfer(b, st, false);
}
// Lights inside a switched-off building leave the active list (they go after it, for the lights-on bake).
static bool lightOff(const float p[3]) {
    int v = volumeAt(p[0], p[1], p[2]);
    return v >= 0 && !bakeVols[v].on;
}
static void writeBakeVolumes() {
    float* o = (float*)volBuf.map;
    for (size_t i = 0; i < bakeVols.size(); i++) {
        auto& b = bakeVols[i];
        uint32_t d[4] = {b.dims[0], b.dims[1], b.dims[2], b.first};
        float e[16] = {b.v.mn[0], b.v.mn[1], b.v.mn[2], b.spacing, b.v.mx[0], b.v.mx[1], b.v.mx[2], b.on ? 1.f : 0.f,
                       0, 0, 0, 0, b.ready[0] ? 1.f : 0.f, b.ready[1] ? 1.f : 0.f, 0, 0};
        memcpy(e + 8, d, 16);
        memcpy(o + i * 16, e, sizeof e);
    }
}
// Next chunk of bake work: the volume the camera is in first (its current state first), then the rest.
struct BakeJob { int vol = -1, st = 0; uint32_t first = 0, n = 0, pass = 0; bool filter = false; uint32_t count = 0, base = 0; };
static BakeJob nextBakeJob() {
    BakeJob j;
    if (!bakeEnabled || bakeVols.empty()) return j;
    int here = volumeAt(cam.pos.x, cam.pos.y, cam.pos.z);
    std::vector<std::pair<int, int>> order;
    if (here >= 0) { order.push_back({here, bakeVols[here].on}); order.push_back({here, !bakeVols[here].on}); }
    for (int i = 0; i < (int)bakeVols.size(); i++) for (int st : {(int)bakeVols[i].on, (int)!bakeVols[i].on}) order.push_back({i, st});
    for (auto [v, st] : order) {
        BakeVol& b = bakeVols[v];
        if (b.ready[st]) continue;
        if (b.pass[st] >= BAKE_PASSES) {  // denoise: one whole-volume filter iteration per frame
            j = {v, st, 0, b.count, b.pass[st], true, b.count, b.first * 2 + (uint32_t)st * b.count};
            if (++b.pass[st] == BAKE_PASSES + BAKE_FILTERS) b.ready[st] = true;  // the GI pass sees it from the next frame on
            return j;
        }
        j = {v, st, b.next[st], std::min(BAKE_CHUNK, b.count - b.next[st]), b.pass[st]};
        b.next[st] += j.n;
        if (b.next[st] >= b.count) { b.next[st] = 0; b.pass[st]++; }
        return j;
    }
    return j;
}
static void saveFinishedBakes() {
    for (auto& b : bakeVols) for (int st = 0; st < 2; st++) if (b.ready[st] && !b.saved[st]) bakeTransfer(b, st, true);
}

// Trees for the raster G-buffer: frustum-cull bounding spheres; near ones join the model draws (one instanced
// run per variant), far ones become impostors, and the 20 m band in between draws both with a dithered split.
static void cullTrees() {
    PROF_FUNC();
    numTreeNear = numImpostors = 0;
    if (!S.rasterPrimary) return;
    auto* g = (GpuModel*)treeModel.gpu;
    float planes[5][4];
    frustumPlanes(planes);
    float* xf = (float*)modelXfBuf.map;
    float* ib = (float*)impBuf.map;
    const float d0 = S.impostorDist, band = 20.f;
    uint32_t first = numModelVisible;
    for (uint32_t v = 0; v < (uint32_t)numTreeVariants; v++) {
        uint32_t runStart = numModelVisible;
        for (const TreeInst& t : treeList) {
            if (t.v != v) continue;
            float c = cosf(t.yaw), sn = sinf(t.yaw), sc = t.scale;
            V3 lc{treeC[v][0], treeC[v][1], treeC[v][2]};
            V3 C = t.p + V3{c * lc.x + sn * lc.z, lc.y, -sn * lc.x + c * lc.z} * sc;
            float R = treeR[v] * sc;
            bool vis = true;
            for (int i = 0; i < 5 && vis; i++) vis = planes[i][0] * C.x + planes[i][1] * C.y + planes[i][2] * C.z + planes[i][3] >= -R;
            if (!vis) continue;
            V3 dd = C - cam.pos;
            float fade = std::clamp((sqrtf(dot(dd, dd)) - d0) / band, 0.f, 1.f);
            if (fade < 1.f && numModelVisible < MAX_MODEL_INST) {
                uint32_t k = numModelVisible++;
                float m[12] = {c * sc, 0, sn * sc, t.p.x, 0, sc, 0, t.p.y, -sn * sc, 0, c * sc, t.p.z};
                memcpy(xf + k * 16, m, 48);
                memcpy(xf + k * 16 + 12, &g->meshes[v].slot, 4);
                xf[k * 16 + 13] = fade;
                xf[k * 16 + 14] = xf[k * 16 + 15] = 0;
            }
            if (fade > 0.f && numImpostors < (uint32_t)MAX_TREES) {
                float e[12] = {t.p.x, t.p.y, t.p.z, sc, t.yaw, (float)v, fade, 0, lc.x, lc.y, lc.z, treeR[v]};
                memcpy(ib + numImpostors * 12, e, 48);
                numImpostors++;
            }
        }
        if (numModelVisible > runStart) drawRuns.push_back({g, v, runStart, numModelVisible - runStart});
    }
    numTreeNear = numModelVisible - first;

    // plants: always the real mesh (they're placed only where they still cover a few pixels)
    auto* pg = (GpuModel*)plantModel.gpu;
    first = numModelVisible;
    for (uint32_t v = 0; pg && v < (uint32_t)trees::PLANT_VARIANTS; v++) {
        uint32_t runStart = numModelVisible;
        for (const PlantInst& p : plantList) {
            if (p.v != v || numModelVisible >= MAX_MODEL_INST) continue;
            const float* m = p.m;
            V3 C{m[0] * plantC[v][0] + m[1] * plantC[v][1] + m[2] * plantC[v][2] + m[3], m[4] * plantC[v][0] + m[5] * plantC[v][1] + m[6] * plantC[v][2] + m[7],
                 m[8] * plantC[v][0] + m[9] * plantC[v][1] + m[10] * plantC[v][2] + m[11]};
            float R = plantR[v] * m[5];
            bool vis = true;
            for (int i = 0; i < 5 && vis; i++) vis = planes[i][0] * C.x + planes[i][1] * C.y + planes[i][2] * C.z + planes[i][3] >= -R;
            if (!vis) continue;
            uint32_t k = numModelVisible++;
            memcpy(xf + k * 16, m, 48);
            memcpy(xf + k * 16 + 12, &pg->meshes[v].slot, 4);
            xf[k * 16 + 13] = xf[k * 16 + 14] = xf[k * 16 + 15] = 0;
        }
        if (numModelVisible > runStart) drawRuns.push_back({pg, v, runStart, numModelVisible - runStart});
    }
    numPlantsDrawn = numModelVisible - first;

    // moving things, every frame: debris particles (one run per kind), fragments and stumps (a run each)
    auto push = [&](GpuModel* gm, uint32_t mesh, const float m[3][4], const float (*prev)[4] = nullptr) {
        if (numModelVisible >= MAX_MODEL_INST) return;
        uint32_t k = numModelVisible++;
        memcpy(xf + k * 16, m, 48);
        memcpy(xf + k * 16 + 12, &gm->meshes[mesh].slot, 4);
        xf[k * 16 + 13] = xf[k * 16 + 14] = xf[k * 16 + 15] = 0;
        if (prev) setPrevXf(k, prev, gm->meshes[mesh].slot);
    };
    for (int kind = 0; pg && kind < DEB_KINDS; kind++) {
        uint32_t runStart = numModelVisible;
        for (const Droplet& d : spray) {
            if (d.delay > 0.f || d.kind != kind) continue;
            float m[3][4], pm[3][4];
            dropletXf(d, m);
            Droplet pd = d;
            if (d.hasPrev) pd.p = d.prevP;
            dropletXf(pd, pm);
            push(pg, (uint32_t)debrisMesh[kind], m, pm);
        }
        if (numModelVisible > runStart) drawRuns.push_back({pg, (uint32_t)debrisMesh[kind], runStart, numModelVisible - runStart});
    }
    // rock fragments: every piece is its own mesh, so one run each
    fragsDrawn = 0;
    for (const Thrown& t : thrown) {
        auto* fg = t.frag ? (GpuModel*)t.frag->model.gpu : nullptr;
        if (!fg) continue;
        uint32_t runStart = numModelVisible;
        push(fg, t.fragMesh, t.xf.m, t.hasPrev ? t.prevXf.m : t.xf.m);
        if (numModelVisible > runStart) { drawRuns.push_back({fg, t.fragMesh, runStart, 1}); fragsDrawn++; }
    }
    for (const Stump& st : stumps) {
        auto* sg = (GpuModel*)st.frag->model.gpu;
        if (!sg) continue;
        uint32_t runStart = numModelVisible;
        push(sg, st.mesh, st.xf.m);
        if (numModelVisible > runStart) drawRuns.push_back({sg, st.mesh, runStart, 1});
    }
    // characters: the model's indices, the character's own skinned vertices (its mesh slot)
    for (character::Character* c : character::all()) {
        auto* gc = (GpuChar*)c->user;
        if (!gc) continue;
        V3 C{0.5f * (c->bmin[0] + c->bmax[0]), 0.5f * (c->bmin[1] + c->bmax[1]), 0.5f * (c->bmin[2] + c->bmax[2])};
        V3 e{c->bmax[0] - C.x, c->bmax[1] - C.y, c->bmax[2] - C.z};
        float R = sqrtf(dot(e, e));
        bool vis = true;
        for (int i = 0; i < 5 && vis; i++) vis = planes[i][0] * C.x + planes[i][1] * C.y + planes[i][2] * C.z + planes[i][3] >= -R;
        if (!vis) continue;
        Xf w = character::worldXf(*c);
        auto* mg = (GpuModel*)gc->asset->model->gpu;
        for (auto& cm : gc->meshes) {
            if (numModelVisible >= MAX_MODEL_INST) break;
            uint32_t k = numModelVisible++;
            memcpy(xf + k * 16, w.m, 48);
            memcpy(xf + k * 16 + 12, &cm.slot, 4);
            xf[k * 16 + 13] = xf[k * 16 + 14] = xf[k * 16 + 15] = 0;
            setPrevXf(k, gc->hasPrev ? gc->prevW.m : w.m, gc->frames > 1 ? cm.prevSlot : cm.slot);
            drawRuns.push_back({mg, cm.mesh, k, 1});
        }
    }
    finishPrevXf();
}

// Generate + upload the tree variants, build their trunk colliders and bake the impostor atlas by ray tracing
// the real meshes (TLAS temporarily holds only the variants, parked far below the terrain).
// forest.txt -> forestAssets (loaded now: the impostor atlas needs the variant count before buildScene)
static void loadForest(const std::string& worldDir) {
    for (int v = 0; v < trees::VARIANTS; v++) treeWeight[v] = 1.f;
    std::ifstream f(worldDir + "forest.txt");
    std::string line;
    float builtin = -1.f;
    while (std::getline(f, line)) {
        std::istringstream ls(line);
        std::string a, name;
        if (!(ls >> a) || a[0] == '#') continue;
        if (a == "builtin") { ls >> builtin; continue; }
        if (a == "spacing") { ls >> forestSpacing; continue; }
        float w = (float)atof(a.c_str());
        std::getline(ls >> std::ws, name);
        if (w <= 0.f || name.empty() || numTreeVariants >= MAX_TREE_VARIANTS) continue;
        ModelData md;
        std::string err;
        auto t0 = std::chrono::steady_clock::now();
        if (!loadGltf(assetFile(worldDir, name), md, err)) { printf("forest: %s: %s\n", name.c_str(), err.c_str()); continue; }
        printf("forest: %s as tree variant %d (%.1f s)\n", name.c_str(), numTreeVariants,
               std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count());
        forestAssets.push_back(std::move(md));
        treeWeight[numTreeVariants++] = w;
    }
    // listed assets replace the built-in trees unless "builtin <w>" keeps a share of them
    float bw = builtin >= 0.f ? builtin : (forestAssets.empty() ? 1.f : 0.f);
    for (int v = 0; v < trees::VARIANTS; v++) treeWeight[v] = bw / trees::VARIANTS;
    treeWeightSum = 0.f;
    for (int v = 0; v < numTreeVariants; v++) treeWeightSum += treeWeight[v];
}

// one mesh (instances baked in, materials offset into treeModel) for an imported forest asset
static MeshData mergeAsset(const ModelData& a, uint32_t matBase) {
    MeshData out;
    for (auto& in : a.instances) {
        if (in.mesh < 0 || in.mesh >= (int)a.meshes.size()) continue;
        const MeshData& m = a.meshes[in.mesh];
        uint32_t base = (uint32_t)out.verts.size() / 8;
        const float(*x)[4] = in.xf.m;
        for (size_t i = 0; i + 8 <= m.verts.size(); i += 8) {
            const float* v = &m.verts[i];
            float n[3];
            for (int k = 0; k < 3; k++) {
                out.verts.push_back(x[k][0] * v[0] + x[k][1] * v[1] + x[k][2] * v[2] + x[k][3]);
                n[k] = x[k][0] * v[3] + x[k][1] * v[4] + x[k][2] * v[5];
            }
            float l = sqrtf(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            for (int k = 0; k < 3; k++) out.verts.push_back(l > 0 ? n[k] / l : 0.f);
            out.verts.push_back(v[6]);
            out.verts.push_back(v[7]);
        }
        for (uint32_t i : m.idx) out.idx.push_back(base + i);
        for (uint32_t t : m.triMat) out.triMat.push_back(t + matBase);
    }
    for (int k = 0; k < 3; k++) { out.bmin[k] = 1e30f; out.bmax[k] = -1e30f; }
    for (size_t i = 0; i < out.verts.size(); i += 8)
        for (int k = 0; k < 3; k++) { out.bmin[k] = std::min(out.bmin[k], out.verts[i + k]); out.bmax[k] = std::max(out.bmax[k], out.verts[i + k]); }
    return out;
}

// Trunk collider for an imported tree: an 8-sided cylinder around the vertices 0.3-1.5 m up (the trunk), to 60% height.
static void assetTrunk(const MeshData& m, std::vector<float>& cv, std::vector<uint32_t>& ci) {
    double sx = 0, sz = 0;
    size_t n = 0;
    std::vector<float> d;
    for (size_t i = 0; i < m.verts.size(); i += 8)
        if (m.verts[i + 1] > m.bmin[1] + 0.3f && m.verts[i + 1] < m.bmin[1] + 1.5f) { sx += m.verts[i]; sz += m.verts[i + 2]; n++; }
    float cx = n ? (float)(sx / n) : 0.f, cz = n ? (float)(sz / n) : 0.f;
    for (size_t i = 0; i < m.verts.size(); i += 8)
        if (m.verts[i + 1] > m.bmin[1] + 0.3f && m.verts[i + 1] < m.bmin[1] + 1.5f)
            d.push_back(hypotf(m.verts[i] - cx, m.verts[i + 2] - cz));
    float r = 0.2f;
    if (!d.empty()) { std::nth_element(d.begin(), d.begin() + d.size() * 3 / 4, d.end()); r = std::clamp(d[d.size() * 3 / 4], 0.08f, 1.2f); }
    float y0 = m.bmin[1], y1 = m.bmin[1] + 0.6f * (m.bmax[1] - m.bmin[1]);
    cv.clear(); ci.clear();
    for (int j = 0; j < 2; j++)
        for (int k = 0; k < 8; k++) {
            float a = k * 6.2831853f / 8;
            cv.insert(cv.end(), {cx + r * cosf(a), j ? y1 : y0, cz + r * sinf(a)});
        }
    for (int k = 0; k < 8; k++) {
        uint32_t a = k, b = (k + 1) % 8;
        ci.insert(ci.end(), {a, b, 8 + b, a, 8 + b, 8 + a});
        if (k >= 2) { ci.insert(ci.end(), {0, (uint32_t)k, (uint32_t)k - 1, 8, 8 + (uint32_t)k - 1, 8 + (uint32_t)k}); }
    }
}

static void uploadVegetation();
static void initTrees() {
    trees::defaults();
    trees::build(treeModel.data);
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
        treeFoot[v] = v < trees::VARIANTS ? 0.f : std::max({-m.bmin[0], m.bmax[0], -m.bmin[2], m.bmax[2]});
    }
    {   // the forest assets carry 4k textures: BC7 them in parallel (cached in texcache/ after the first run)
        PROF_ZONE("compress tree textures");
        std::vector<std::thread> ts;
        for (auto& t : treeModel.data.textures) ts.emplace_back([&t] { texcomp::prepare(t); });
        for (auto& th : ts) th.join();
    }
    uploadModel(treeModel);
    trees::buildPlants(plantModel.data);
    plantModel.path = "<procedural plants>";
    plantModel.state = 1;
    for (int v = 0; v < trees::PLANT_VARIANTS; v++) trees::bounds(plantModel.data.meshes[v], plantC[v], plantR[v]);
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
static bool vegDirty = false;
static int vegRegenFrame = -1;
static void regenVegetation() {
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
    // same procedural texture / material / mesh counts every time, so the assets' offsets after them still hold
    auto& d = treeModel.data;
    std::copy(fresh.textures.begin(), fresh.textures.end(), d.textures.begin());
    std::copy(fresh.materials.begin(), fresh.materials.end(), d.materials.begin());
    std::copy(fresh.meshes.begin(), fresh.meshes.end(), d.meshes.begin());
    uploadVegetation();
    placeDirty = true;
}

// Trunk colliders only where the player can reach them (static bodies within ~40 m, dropped past 60 m).
static void syncTreeBodies() {
    // colliders around the player (walk mode) and around every thrown rock, so rocks hit trunks
    std::vector<V3> around;
    if (walkMode) around.push_back(cam.pos);
    for (const Thrown& t : thrown) around.push_back({t.xf.m[0][3], t.xf.m[1][3], t.xf.m[2][3]});
    if (around.empty() && treeBodies.empty()) return;
    PROF_FUNC();
    std::vector<void*> shapes;
    std::vector<Xf> xfs;
    std::vector<uint64_t> keys;
    std::unordered_map<uint64_t, bool> nearNow;
    for (const TreeInst& t : treeList) {
        float d2 = 1e30f;
        for (const V3& a : around) d2 = std::min(d2, (t.p.x - a.x) * (t.p.x - a.x) + (t.p.z - a.z) * (t.p.z - a.z));
        if (d2 > 60.f * 60.f) continue;
        uint64_t key = treeKey(t.ix, t.iz);
        nearNow[key] = true;
        if (d2 > 40.f * 40.f || treeBodies.count(key)) continue;
        float c = cosf(t.yaw) * t.scale, sn = sinf(t.yaw) * t.scale;
        xfs.push_back({{{c, 0, sn, t.p.x}, {0, t.scale, 0, t.p.y}, {-sn, 0, c, t.p.z}}});
        shapes.push_back(trunkShape[t.v]);
        keys.push_back(key);
    }
    std::vector<uint32_t> gone;
    for (auto it = treeBodies.begin(); it != treeBodies.end();)
        if (!nearNow.count(it->first)) { gone.push_back(it->second); it = treeBodies.erase(it); } else ++it;
    if (!gone.empty()) physics::removeBodies(gone.data(), gone.size());
    if (shapes.empty()) return;
    std::vector<uint32_t> ids(shapes.size());
    physics::addStatic(shapes.data(), xfs.data(), shapes.size(), ids.data());
    for (size_t i = 0; i < keys.size(); i++) treeBodies[keys[i]] = ids[i];
}

// Light SoA of every streamed light; each frame the strongest (by luminance / distance^2) fill the slots after the orbs.
static std::vector<float> wlx, wly, wlz, wlLum, wlScore;
static void rebuildWorldLights() {
    auto& L = stream::lights();
    wlx.resize(L.size()); wly.resize(L.size()); wlz.resize(L.size()); wlLum.resize(L.size()); wlScore.resize(L.size());
    for (size_t i = 0; i < L.size(); i++) {
        wlx[i] = L[i].pos[0]; wly[i] = L[i].pos[1]; wlz[i] = L[i].pos[2];
        wlLum[i] = 0.2126f * L[i].color[0] + 0.7152f * L[i].color[1] + 0.0722f * L[i].color[2];
    }
}

// a flatten zone's ground texture (matlib albedo) in a bindless slot, uploaded once per name; returns slot + 1
static uint32_t zoneTexSlot(const std::string& name) {
    static std::map<std::string, uint32_t> slots;
    if (auto it = slots.find(name); it != slots.end()) return it->second;
    const TextureData& t = matlib::texture(name);
    uint32_t r = 0;
    if (t.w && t.h && texSlots.used() < texSlots.cap) {
        uint32_t mips = mipCount(t.w, t.h);
        Image im = mkTexture(t.w, t.h, mips, t.linear ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_R8G8B8A8_SRGB);
        Buffer st = mkBuf(t.rgba.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        memcpy(st.map, t.rgba.data(), t.rgba.size());
        oneShot([&](VkCommandBuffer cb) { recordTextureUpload(cb, im, t.w, t.h, mips, st.b, 0); });
        freeBuf(st);
        uint32_t s = texSlots.alloc();
        writeTexDescriptor(s, im.view);
        r = s + 1;
    }
    return slots[name] = r;
}

// World map edits -> GPU: the dirty cell rectangle of each layer, through one staging buffer, into this frame's
// upload command buffer. Collision tiles and placement follow (throttled while a brush is held down).
static void uploadWorldMap() {
    worldmap::Rect r = worldmap::takeDirty();
    static double lastCollision = 0;
    static uint32_t pendingVer = 0;
    double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (pendingVer && now - lastCollision > 0.3) {  // collision + placement: at most ~3x a second while editing
        stream::terrainChanged();
        placeDirty = true;
        lastCollision = now;
        pendingVer = 0;
    }
    if (r.x0 > r.x1) return;
    pendingVer = worldmap::version();
    terrainDirty = true;
    const int N = worldmap::N, w = r.x1 - r.x0 + 1, h = r.z1 - r.z0 + 1;
    int wx0 = r.x0 / 2, wz0 = r.z0 / 2, wx1 = r.x1 / 2, wz1 = r.z1 / 2, ww = wx1 - wx0 + 1, wh = wz1 - wz0 + 1;
    // each layer starts 16-byte aligned (copy offsets must be a multiple of the texel size: 16 for the water layer)
    size_t bh = ((size_t)w * h * 4 + 15) & ~size_t(15), bp = bh, bw = (size_t)ww * wh * 16;
    Buffer st = mkBuf(bh + bp + bw, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    char* m = (char*)st.map;
    for (int z = 0; z < h; z++) {
        memcpy(m + (size_t)z * w * 4, worldmap::heightData() + (size_t)(r.z0 + z) * N + r.x0, (size_t)w * 4);
        memcpy(m + bh + (size_t)z * w * 4, worldmap::paintData() + (size_t)(r.z0 + z) * N + r.x0, (size_t)w * 4);
    }
    for (int z = 0; z < wh; z++) memcpy(m + bh + bp + (size_t)z * ww * 16, worldmap::waterData() + (size_t)(wz0 + z) * (N / 2) + wx0, (size_t)ww * 16);
    beginUploads();
    auto copy = [&](const Image& im, VkDeviceSize off, int x, int z, int cw, int ch) {
        VkBufferImageCopy c{};
        c.bufferOffset = off;
        c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.imageOffset = {x, z, 0};
        c.imageExtent = {(uint32_t)cw, (uint32_t)ch, 1};
        vkCmdCopyBufferToImage(uploadCmd, st.b, im.img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    };
    memBarrier(uploadCmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    copy(mapHImg, 0, r.x0, r.z0, w, h);
    copy(mapPaintImg, bh, r.x0, r.z0, w, h);
    copy(mapWaterImg, bh + bp, wx0, wz0, ww, wh);
    memBarrier(uploadCmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_READ_BIT);
    retireNext.push_back([st]() mutable { freeBuf(st); });
}

static void onStreamChanged(bool moveOnly = false) {
    PROF_FUNC();
    meshPartsCache.clear();
    std::vector<std::pair<const MeshData*, DrawItem>> items;
    stream::forEachPlaced([&](const Placed& p) {
        auto* g = (GpuModel*)p.model->gpu;
        if (g && g->meshes[p.mesh].blas) items.push_back({&p.model->data.meshes[p.mesh], DrawItem{g, (uint32_t)p.mesh, p.xf}});
    });
    std::stable_sort(items.begin(), items.end(), [](auto& a, auto& b) {
        return a.second.m->meshes[a.second.mesh].slot < b.second.m->meshes[b.second.mesh].slot;
    });
    rebuildModelInstances(items);
    rebuildWorldLights();
    if (moveOnly) return;  // editor drag of a mesh / light: terrain, vegetation, bakes unchanged
    placeDirty = true;
    rebuildBakeVolumes(false);
    syncAudioInstances();
    auto& zs = stream::zones();
    float* z = (float*)zoneBuf.map;
    for (size_t i = 0; i < std::min<size_t>(zs.size(), MAX_ZONES); i++) {
        uint32_t slot = zs[i].mat.empty() ? 0 : zoneTexSlot(zs[i].mat);  // 0 = no texture, else slot + 1
        float v[8] = {zs[i].minx, zs[i].minz, zs[i].maxx, zs[i].maxz, zs[i].height, zs[i].falloff, (float)slot, zs[i].tile};
        memcpy(z + i * 8, v, sizeof v);
    }
    terrainDirty = true;
}

// ---------------------------------------------------------------- descriptors / pipelines
static VkShaderModule loadModule(const char* name) {
    auto code = readFile(std::string(SHADER_DIR) + name);
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = code.size();
    smi.pCode = (const uint32_t*)code.data();
    VkShaderModule mod;
    VK_CHECK(vkCreateShaderModule(dev, &smi, nullptr, &mod));
    return mod;
}

// G-buffer raster pipeline: 7 color targets (gDepth, gNorm, gAlbedo, gEmis, gMat, gMat2, gTan) + reverse-Z depth.
// stride 0 = no vertex input (vertex pulling)
static VkPipeline mkRasterPipeline(const char* vert, uint32_t stride, bool hasNormal, const char* frag = "raster.frag.spv") {
    VkShaderModule vm = loadModule(vert), fm = loadModule(frag);
    VkPipelineShaderStageCreateInfo st[2] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vm, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fm, "main", nullptr}};
    VkVertexInputBindingDescription vb{0, stride, VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription va[2] = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12}};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = stride ? 1 : 0;
    vi.pVertexBindingDescriptions = &vb;
    vi.vertexAttributeDescriptionCount = stride ? (hasNormal ? 2 : 1) : 0;
    vi.pVertexAttributeDescriptions = va;
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.lineWidth = 1.f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_GREATER;
    VkPipelineColorBlendAttachmentState cba[GBUF_TARGETS]{};
    for (auto& a : cba) a.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = GBUF_TARGETS;
    cb.pAttachments = cba;
    VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dy.dynamicStateCount = 2;
    dy.pDynamicStates = dyn;
    VkPipelineRenderingCreateInfo pr{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    VkFormat fmts[GBUF_TARGETS];
    for (uint32_t i = 0; i < GBUF_TARGETS; i++) fmts[i] = gbufFormat(i);
    pr.colorAttachmentCount = GBUF_TARGETS;
    pr.pColorAttachmentFormats = fmts;
    pr.depthAttachmentFormat = depthFmt;
    VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gp.pNext = &pr;
    gp.stageCount = 2;
    gp.pStages = st;
    gp.pVertexInputState = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vp;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms;
    gp.pDepthStencilState = &ds;
    gp.pColorBlendState = &cb;
    gp.pDynamicState = &dy;
    gp.layout = pipeLayout;
    VkPipeline p;
    VK_CHECK(vkCreateGraphicsPipelines(dev, pipeCache, 1, &gp, nullptr, &p));
    vkDestroyShaderModule(dev, vm, nullptr);
    vkDestroyShaderModule(dev, fm, nullptr);
    return p;
}

static void createLayoutAndPipelines() {
    std::vector<VkDescriptorSetLayoutBinding> b;
    auto add = [&](uint32_t i, VkDescriptorType t, uint32_t n = 1) { b.push_back({i, t, n, VK_SHADER_STAGE_ALL, nullptr}); };
    add(0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
    add(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    for (uint32_t i = 2; i <= 9; i++) add(i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    for (uint32_t i = 10; i < 10 + I_COUNT; i++) add(i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    add(25, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    add(26, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    add(27, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    add(28, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    for (uint32_t i = 29; i <= 31; i++) add(i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    add(32, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_TEXTURES);  // model textures (model.glsl)
    for (uint32_t i = 33; i <= 36; i++) add(i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);  // meshes, materials, raster xf, zones
    for (uint32_t i = 37; i < 37 + RR_COUNT; i++) add(i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);  // DLSS-RR inputs + output
    add(44, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);                                               // impostor instances
    add(45, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);                                       // impostor atlas (albedo, normal+depth)
    add(46, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    add(47, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);                                                // same, written by the bake
    add(48, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    add(49, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);                                               // bake volumes, probes
    add(50, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    add(51, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);                                               // probe filter scratch
    add(52, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);                                                // wind map ping-pong
    add(53, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    for (uint32_t i = 0; i < X_COUNT; i++) add(X_BINDING + i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);  // more screen images
    for (uint32_t i = 80; i <= 82; i++) add(i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);  // triangle lights, light tree, blue noise
    add(83, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);                                      // post.glsl: exposure
    add(87, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);                                      // light alias table
    add(84, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);                              // HDR scene
    add(85, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);                              // bloom chain
    add(86, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8);                                    // bloom mips
    add(88, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);                              // UI (HDR present)
    add(89, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);                                       // composed HDR frame
    for (uint32_t i = 90; i <= 92; i++) add(i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);  // world map: height, paint, water
    for (uint32_t i = 93; i <= 96; i++) add(i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);           // world sim: state, water flux, snow flux, ground
    add(97, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);                                               // crater bins
    add(98, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);                                               // radiance cache
    for (uint32_t i = 99; i <= 101; i++) add(i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);  // ground virtual texture
    add(102, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);                                              // its tile requests
    for (uint32_t i = 103; i <= 105; i++) add(i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // vtgen.comp outputs
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = (uint32_t)b.size();
    lci.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &lci, nullptr, &setLayout));

    std::vector<VkDescriptorPoolSize> ps;  // exactly what the layout holds
    for (auto& x : b) {
        auto it = std::find_if(ps.begin(), ps.end(), [&](auto& p) { return p.type == x.descriptorType; });
        if (it == ps.end()) ps.push_back({x.descriptorType, x.descriptorCount});
        else it->descriptorCount += x.descriptorCount;
    }
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1;
    pci.poolSizeCount = (uint32_t)ps.size();
    pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(dev, &pci, nullptr, &descPool));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = descPool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &setLayout;
    VK_CHECK(vkAllocateDescriptorSets(dev, &dai, &descSet));

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &setLayout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &pipeLayout));

    for (int i = 0; i < P_COUNT; i++) {
        VkShaderModule mod = loadModule((std::string(pipeNames[i]) + ".comp.spv").c_str());
        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", nullptr};
        cpi.layout = pipeLayout;
        VK_CHECK(vkCreateComputePipelines(dev, pipeCache, 1, &cpi, nullptr, &pipes[i]));
        vkDestroyShaderModule(dev, mod, nullptr);
    }
    rasterTerrain = mkRasterPipeline("raster_terrain.vert.spv", 24, true);
    rasterGrass = mkRasterPipeline("raster_grass.vert.spv", 12, false);
    rasterRock = mkRasterPipeline("raster_rock.vert.spv", 24, true);
    rasterModel = mkRasterPipeline("raster_model.vert.spv", 0, false, "raster_model.frag.spv");
    rasterImpostor = mkRasterPipeline("raster_impostor.vert.spv", 0, false, "raster_impostor.frag.spv");
}

// 1x1 white texture in every bindless slot, so unused slots are always valid
static void initTextureSlots() {
    defaultTex = mkTexture(1, 1, 1);
    Buffer st = mkBuf(4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memset(st.map, 0xFF, 4);
    oneShot([&](VkCommandBuffer cb) { recordTextureUpload(cb, defaultTex, 1, 1, 1, st.b, 0); });
    freeBuf(st);
    std::vector<VkDescriptorImageInfo> ii(MAX_TEXTURES, {texSampler, defaultTex.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = descSet;
    w.dstBinding = 32;
    w.descriptorCount = MAX_TEXTURES;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = ii.data();
    vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
}

// Generated high-res ground sets (texgen.cpp, cached in texcache/): albedo sRGB + detail linear, into bindless slots.
static void loadGroundTextures() {
    texgen::Params p;
    p.size = groundTexSize;
    for (int k = 0; k < 2; k++) {
        texgen::Set set;
        auto t0 = std::chrono::steady_clock::now();
        bool cached = texgen::load((texgen::Kind)k, p, "texcache", set);
        printf("ground textures %d: %ux%u %s (%.1f s)\n", k, set.size, set.size, cached ? "from cache" : "generated",
               std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count());
        groundMaxH[k] = set.maxHeight;
        uint32_t n = set.size, mips = mipCount(n, n);
        size_t bytes = (size_t)n * n * 4;
        Buffer st = mkBuf(bytes * 2, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        memcpy(st.map, set.albedo.data(), bytes);
        memcpy((char*)st.map + bytes, set.detail.data(), bytes);
        for (int j = 0; j < 2; j++) {
            Image& im = groundImg[2 * k + j];
            im = mkTexture(n, n, mips, j == 0 ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM);
            groundSlots[2 * k + j] = texSlots.alloc();
        }
        oneShot([&](VkCommandBuffer cb) {
            for (int j = 0; j < 2; j++) recordTextureUpload(cb, groundImg[2 * k + j], n, n, mips, st.b, bytes * j);
        });
        freeBuf(st);
        for (int j = 0; j < 2; j++) writeTexDescriptor(groundSlots[2 * k + j], groundImg[2 * k + j].view);
    }
    groundReady = true;
}

static void writeDescriptors() {
    std::vector<VkWriteDescriptorSet> w;
    VkWriteDescriptorSetAccelerationStructureKHR asw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    asw.accelerationStructureCount = 1;
    asw.pAccelerationStructures = &tlas.as;
    VkWriteDescriptorSet ws{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    ws.dstSet = descSet;
    ws.descriptorCount = 1;

    VkWriteDescriptorSet a = ws;
    a.pNext = &asw;
    a.dstBinding = 0;
    a.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    w.push_back(a);

    VkDescriptorBufferInfo bufs[9] = {{uboBuf.b, 0, VK_WHOLE_SIZE},   {terrainV.b, 0, VK_WHOLE_SIZE}, {terrainI.b, 0, VK_WHOLE_SIZE},
                                      {grassV.b, 0, VK_WHOLE_SIZE},   {grassI.b, 0, VK_WHOLE_SIZE},   {bladeBuf.b, 0, VK_WHOLE_SIZE},
                                      {lightBuf.b, 0, VK_WHOLE_SIZE}, {resCur.b, 0, VK_WHOLE_SIZE},   {resPrev.b, 0, VK_WHOLE_SIZE}};
    for (uint32_t i = 0; i < 9; i++) {
        VkWriteDescriptorSet x = ws;
        x.dstBinding = 1 + i;
        x.descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        x.pBufferInfo = &bufs[i];
        w.push_back(x);
    }
    VkDescriptorImageInfo ii[I_COUNT];
    for (uint32_t i = 0; i < I_COUNT; i++) {
        ii[i] = {VK_NULL_HANDLE, imgs[i].view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet x = ws;
        x.dstBinding = 10 + i;
        x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        x.pImageInfo = &ii[i];
        w.push_back(x);
    }
    VkDescriptorBufferInfo gib[7] = {{giCur.b, 0, VK_WHOLE_SIZE}, {giPrev.b, 0, VK_WHOLE_SIZE}, {grassCounter.b, 0, VK_WHOLE_SIZE},
                                     {}, {rockV.b, 0, VK_WHOLE_SIZE}, {rockI.b, 0, VK_WHOLE_SIZE}, {rockXf.b, 0, VK_WHOLE_SIZE}};
    VkDescriptorImageInfo skyInfo{VK_NULL_HANDLE, skyImg.view, VK_IMAGE_LAYOUT_GENERAL};
    for (uint32_t i = 0; i < 7; i++) {
        if (i == 3) {
            VkWriteDescriptorSet x = ws;
            x.dstBinding = 28;
            x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            x.pImageInfo = &skyInfo;
            w.push_back(x);
            continue;
        }
        VkWriteDescriptorSet x = ws;
        x.dstBinding = 25 + i;
        x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        x.pBufferInfo = &gib[i];
        w.push_back(x);
    }
    VkDescriptorBufferInfo mb[4] = {{meshInfoBuf.b, 0, VK_WHOLE_SIZE}, {materialBuf.b, 0, VK_WHOLE_SIZE}, {modelXfBuf.b, 0, VK_WHOLE_SIZE},
                                    {zoneBuf.b, 0, VK_WHOLE_SIZE}};
    for (uint32_t i = 0; i < 4; i++) {
        VkWriteDescriptorSet x = ws;
        x.dstBinding = 33 + i;
        x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        x.pBufferInfo = &mb[i];
        w.push_back(x);
    }
    VkDescriptorImageInfo ri[RR_COUNT];
    for (uint32_t i = 0; i < RR_COUNT; i++) {
        ri[i] = {VK_NULL_HANDLE, rrImgs[i].view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet x = ws;
        x.dstBinding = 37 + i;
        x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        x.pImageInfo = &ri[i];
        w.push_back(x);
    }
    VkDescriptorImageInfo windInfo[2] = {{VK_NULL_HANDLE, windImg[0].view, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, windImg[1].view, VK_IMAGE_LAYOUT_GENERAL}};
    for (uint32_t i = 0; i < 2; i++) {
        VkWriteDescriptorSet x = ws;
        x.dstBinding = 52 + i;
        x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        x.pImageInfo = &windInfo[i];
        w.push_back(x);
    }
    VkDescriptorBufferInfo impInfo[4] = {{impBuf.b, 0, VK_WHOLE_SIZE}, {volBuf.b, 0, VK_WHOLE_SIZE}, {probeBuf.b, 0, VK_WHOLE_SIZE},
                                         {probeTmp.b, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet x = ws;
    for (uint32_t i = 0; i < 4; i++) {
        x = ws;
        x.dstBinding = i == 0 ? 44 : 48 + i;
        x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        x.pBufferInfo = &impInfo[i];
        w.push_back(x);
    }
    VkDescriptorImageInfo impImg[4] = {{texSampler, impAlbedoImg.view, VK_IMAGE_LAYOUT_GENERAL}, {texSampler, impNormalImg.view, VK_IMAGE_LAYOUT_GENERAL},
                                       {VK_NULL_HANDLE, impAlbedoImg.view, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, impNormalImg.view, VK_IMAGE_LAYOUT_GENERAL}};
    for (uint32_t i = 0; i < 4; i++) {
        x = ws;
        x.dstBinding = 45 + i;
        x.descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        x.pImageInfo = &impImg[i];
        w.push_back(x);
    }
    VkDescriptorImageInfo mapInfo[7] = {{postSampler, mapHImg.view, VK_IMAGE_LAYOUT_GENERAL}, {postSampler, mapPaintImg.view, VK_IMAGE_LAYOUT_GENERAL},
                                        {postSampler, mapWaterImg.view, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, simImg.view, VK_IMAGE_LAYOUT_GENERAL},
                                        {VK_NULL_HANDLE, fluxWImg.view, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, fluxSImg.view, VK_IMAGE_LAYOUT_GENERAL},
                                        {VK_NULL_HANDLE, simGroundImg.view, VK_IMAGE_LAYOUT_GENERAL}};
    for (uint32_t i = 0; i < 7; i++) {
        x = ws;
        x.dstBinding = 90 + i;
        x.descriptorType = i < 3 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        x.pImageInfo = &mapInfo[i];
        w.push_back(x);
    }
    VkDescriptorBufferInfo cbInfo{craterBinBuf.b, 0, VK_WHOLE_SIZE};
    x = ws;
    x.dstBinding = 97;
    x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    x.pBufferInfo = &cbInfo;
    w.push_back(x);
    VkDescriptorBufferInfo rcInfo{radCacheBuf.b, 0, VK_WHOLE_SIZE};
    x.dstBinding = 98;
    x.pBufferInfo = &rcInfo;
    w.push_back(x);
    VkDescriptorImageInfo vtInfo[6] = {{texSampler, vtImgA.view, VK_IMAGE_LAYOUT_GENERAL}, {texSampler, vtImgB.view, VK_IMAGE_LAYOUT_GENERAL},
                                       {texSampler, vtImgH.view, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, vtImgA.view, VK_IMAGE_LAYOUT_GENERAL},
                                       {VK_NULL_HANDLE, vtImgB.view, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, vtImgH.view, VK_IMAGE_LAYOUT_GENERAL}};
    for (uint32_t i = 0; i < 6; i++) {
        x = ws;
        x.dstBinding = i < 3 ? 99 + i : 100 + i;
        x.descriptorType = i < 3 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        x.pImageInfo = &vtInfo[i];
        w.push_back(x);
    }
    VkDescriptorBufferInfo vtTiles{vtTileBuf.b, 0, VK_WHOLE_SIZE};
    x = ws;
    x.dstBinding = 102;
    x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    x.pBufferInfo = &vtTiles;
    w.push_back(x);
    VkDescriptorBufferInfo lb[3] = {{triLightBuf.b, 0, VK_WHOLE_SIZE}, {lightTreeBuf.b, 0, VK_WHOLE_SIZE}, {blueNoiseBuf.b, 0, VK_WHOLE_SIZE}};
    for (uint32_t i = 0; i < 3; i++) {
        x = ws;
        x.dstBinding = 80 + i;
        x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        x.pBufferInfo = &lb[i];
        w.push_back(x);
    }
    VkDescriptorBufferInfo eb{exposureBuf.b, 0, VK_WHOLE_SIZE}, ab{lightAliasBuf.b, 0, VK_WHOLE_SIZE};
    x = ws;
    x.dstBinding = 83;
    x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    x.pBufferInfo = &eb;
    w.push_back(x);
    x.dstBinding = 87;
    x.pBufferInfo = &ab;
    w.push_back(x);
    VkDescriptorImageInfo pi[2] = {{postSampler, ximgs[X_HDR].view, VK_IMAGE_LAYOUT_GENERAL}, {postSampler, bloomImg.view, VK_IMAGE_LAYOUT_GENERAL}};
    for (uint32_t i = 0; i < 2; i++) {
        x = ws;
        x.dstBinding = 84 + i;
        x.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        x.pImageInfo = &pi[i];
        w.push_back(x);
    }
    VkDescriptorImageInfo bm[8];
    for (uint32_t i = 0; i < 8; i++) bm[i] = {VK_NULL_HANDLE, bloomViews[std::min(i, BLOOM_MIPS - 1)], VK_IMAGE_LAYOUT_GENERAL};
    x = ws;
    x.dstBinding = 86;
    x.descriptorCount = 8;
    x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    x.pImageInfo = bm;
    w.push_back(x);
    VkDescriptorImageInfo xi[X_COUNT];
    for (uint32_t i = 0; i < X_COUNT; i++) {
        xi[i] = {VK_NULL_HANDLE, ximgs[i].view, VK_IMAGE_LAYOUT_GENERAL};
        x = ws;
        x.dstBinding = X_BINDING + i;
        x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        x.pImageInfo = &xi[i];
        w.push_back(x);
    }
    vkUpdateDescriptorSets(dev, (uint32_t)w.size(), w.data(), 0, nullptr);
    writePresentDescriptors();
}

static void writePresentDescriptors() {
    VkDescriptorImageInfo ui{postSampler, uiImg.view, VK_IMAGE_LAYOUT_GENERAL}, pr{VK_NULL_HANDLE, presentImg.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
    for (int i = 0; i < 2; i++) { w[i].dstSet = descSet; w[i].descriptorCount = 1; }
    w[0].dstBinding = 88; w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &ui;
    w[1].dstBinding = 89; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[1].pImageInfo = &pr;
    vkUpdateDescriptorSets(dev, 2, w, 0, nullptr);
}

static void createScreenResources() {
    // render resolution follows the upscaler mode; the output (I_OUT, RR_OUT) stays at the chosen resolution
    float sc = upscaler != dlss::OFF ? dlss::renderScale(upscaler) : 1.f;
    W = std::max(64u, (uint32_t)(OW * sc + 0.5f));
    H = std::max(64u, (uint32_t)(OH * sc + 0.5f));
    for (int i = 0; i < I_COUNT; i++) imgs[i] = i == I_OUT ? mkImage(OW, OH, imgFormats[i]) : mkImage(W, H, imgFormats[i]);
    for (int i = 0; i < RR_COUNT; i++) rrImgs[i] = i == RR_OUT ? mkImage(OW, OH, rrFormats[i]) : mkImage(W, H, rrFormats[i]);
    bloomImg = mkImageUsage(std::max(OW / 2, 1u), std::max(OH / 2, 1u), VK_FORMAT_R16G16B16A16_SFLOAT,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, BLOOM_MIPS);
    for (uint32_t m = 0; m < BLOOM_MIPS; m++) {
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = bloomImg.img;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, m, 1, 0, 1};
        VK_CHECK(vkCreateImageView(dev, &vi, nullptr, &bloomViews[m]));
    }
    for (int i = 0; i < X_COUNT; i++) {
        SizeClass c = xDefs[i].size;
        ximgs[i] = c == OUTPUT_RES ? mkImage(OW, OH, xDefs[i].fmt) : c == HALF_RES ? mkImage((W + 1) / 2, (H + 1) / 2, xDefs[i].fmt) : mkImage(W, H, xDefs[i].fmt);
    }
    VkDeviceSize rs = (VkDeviceSize)W * H * 16;
    resCur = mkBuf(rs, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    resPrev = mkBuf(rs, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    giCur = mkBuf(rs * 3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    giPrev = mkBuf(rs * 3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    depthImg = mkImage(W, H, depthFmt);
    oneShot([](VkCommandBuffer cb) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = depthImg.img;
        b.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        b.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        for (auto& im : imgs)
            imgBarrier(cb, im.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        for (auto& im : rrImgs)
            imgBarrier(cb, im.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        for (auto& im : ximgs)
            imgBarrier(cb, im.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        VkImageMemoryBarrier bb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        bb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        bb.srcQueueFamilyIndex = bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.image = bloomImg.img;
        bb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, BLOOM_MIPS, 0, 1};
        bb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &bb);
    });
    rrActive = false;
    if (upscaler != dlss::OFF)
        oneShot([&](VkCommandBuffer cb) { rrActive = dlss::create(cb, upscaler, W, H, OW, OH); });
    if (upscaler != dlss::OFF && !rrActive) fprintf(stderr, "dlss: falling back to the built-in denoiser\n");
    resetHistory = true;
}

static void destroyScreenResources() {
    dlss::release();
    for (auto& im : imgs) freeImage(im);
    for (auto& im : rrImgs) freeImage(im);
    for (auto& im : ximgs) freeImage(im);
    for (uint32_t m = 0; m < BLOOM_MIPS; m++) vkDestroyImageView(dev, bloomViews[m], nullptr);
    freeImage(bloomImg);
    freeBuf(resCur);
    freeBuf(resPrev);
    freeBuf(giCur);
    freeBuf(giPrev);
    freeImage(depthImg);
}

// ---------------------------------------------------------------- per-frame
static uint32_t flags() {
    return (S.restir ? 1u : 0u) | (S.restirTemporal ? 2u : 0u) | (S.restirSpatial ? 4u : 0u) | (S.accumulate ? 8u : 0u) |
           (S.restirGI ? 16u : 0u) | (S.giTemporal ? 32u : 0u) | (S.giSpatial ? 64u : 0u) | (S.giEnabled ? 0u : 128u) |
           (S.giCache ? 256u : 0u) | (S.giChecker ? 512u : 0u) | (S.giProbes ? 1024u : 0u);
}

static V3 sunDirection() {
    float el = S.sunElev * 3.14159265f / 180.f, az = S.sunAzim * 3.14159265f / 180.f;
    return {cosf(el) * sinf(az), sinf(el), cosf(el) * cosf(az)};
}

// Emissive triangles of the placed models (not in switched-off buildings) + the active point
// lights -> triangle light buffer and the light BVH (lighttree.h). Rebuilt every frame: the orbs move.
static void buildLightTree(const float* pointLights) {
    PROF_FUNC();
    struct Tri { float v[3][3]; float Le[3]; float area, score; };
    static std::vector<Tri> tris;
    tris.clear();
    for (uint32_t ii : emisItems) {
        const DrawItem& di = drawItems[ii];
        for (const EmisTri& e : di.m->meshes[di.mesh].emis) {
            Tri t;
            for (int v = 0; v < 3; v++)
                for (int r = 0; r < 3; r++)
                    t.v[v][r] = di.xf.m[r][0] * e.v[v][0] + di.xf.m[r][1] * e.v[v][1] + di.xf.m[r][2] * e.v[v][2] + di.xf.m[r][3];
            V3 a{t.v[0][0], t.v[0][1], t.v[0][2]}, b{t.v[1][0], t.v[1][1], t.v[1][2]}, c{t.v[2][0], t.v[2][1], t.v[2][2]};
            V3 cr = cross(b - a, c - a), ctr = (a + b + c) * (1.f / 3.f);
            t.area = 0.5f * sqrtf(dot(cr, cr));
            float cp[3] = {ctr.x, ctr.y, ctr.z};
            if (t.area < 1e-8f || lightOff(cp)) continue;  // switched-off building
            memcpy(t.Le, e.Le, 12);
            V3 dc = ctr - cam.pos;
            t.score = (0.2126f * t.Le[0] + 0.7152f * t.Le[1] + 0.0722f * t.Le[2]) * t.area / std::max(dot(dc, dc), 1.f);
            tris.push_back(t);
        }
    }
    allEmittersListed = tris.size() <= MAX_TRI_LIGHTS;
    if (!allEmittersListed) {
        static bool warned = false;
        if (!warned) fprintf(stderr, "lights: %zu emissive triangles, keeping the %u brightest near the camera\n", tris.size(), MAX_TRI_LIGHTS);
        warned = true;
        std::nth_element(tris.begin(), tris.begin() + MAX_TRI_LIGHTS, tris.end(), [](const Tri& a, const Tri& b) { return a.score > b.score; });
        tris.resize(MAX_TRI_LIGHTS);
    }
    numTriLights = (uint32_t)tris.size();
    float* T = (float*)triLightBuf.map;
    std::vector<lighttree::Prim> prims;
    prims.reserve(numLights + numTriLights);
    for (uint32_t i = 0; i < numLights; i++) {
        const float* l = pointLights + i * 8;
        float r = l[3];
        prims.push_back({{l[0] - r, l[1] - r, l[2] - r}, {l[0] + r, l[1] + r, l[2] + r}, 0.2126f * l[4] + 0.7152f * l[5] + 0.0722f * l[6]});
    }
    for (uint32_t i = 0; i < numTriLights; i++) {
        const Tri& t = tris[i];
        V3 a{t.v[0][0], t.v[0][1], t.v[0][2]}, b{t.v[1][0], t.v[1][1], t.v[1][2]}, c{t.v[2][0], t.v[2][1], t.v[2][2]};
        V3 n = normalize(cross(b - a, c - a));
        float rec[16] = {t.v[0][0], t.v[0][1], t.v[0][2], t.Le[0], t.v[1][0], t.v[1][1], t.v[1][2], t.Le[1],
                         t.v[2][0], t.v[2][1], t.v[2][2], t.Le[2], n.x, n.y, n.z, t.area};
        memcpy(T + i * 16, rec, sizeof rec);
        lighttree::Prim p;
        for (int k = 0; k < 3; k++) {
            p.mn[k] = std::min({t.v[0][k], t.v[1][k], t.v[2][k]});
            p.mx[k] = std::max({t.v[0][k], t.v[1][k], t.v[2][k]});
        }
        p.power = (0.2126f * t.Le[0] + 0.7152f * t.Le[1] + 0.0722f * t.Le[2]) * t.area;
        prims.push_back(p);
    }
    static std::vector<lighttree::Node> nodes;
    lighttree::build(prims, nodes);
    numTreeNodes = (uint32_t)nodes.size();
    memcpy(lightTreeBuf.map, nodes.data(), nodes.size() * sizeof(lighttree::Node));
    static std::vector<lighttree::AliasEntry> alias;
    lighttree::buildAlias(prims, alias);
    memcpy(lightAliasBuf.map, alias.data(), alias.size() * sizeof(lighttree::AliasEntry));
}

// CPU mirror of material.glsl vnoise / grassMask (without river, zones, craters): bare ground for clutter placement.
static float vnoiseG(float x, float y) {
    auto hu = [](uint32_t v) { v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; v ^= v >> 16; return v; };
    float ix = floorf(x), iy = floorf(y), fx = x - ix, fy = y - iy;
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy);
    auto h = [&](float a, float b) { return hu((uint32_t)(int)a * 7919u + (uint32_t)(int)b * 104729u) / 4294967295.f; };
    float a = h(ix, iy), b = h(ix + 1, iy), c = h(ix, iy + 1), d = h(ix + 1, iy + 1);
    return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
}
static float grassMaskCPU(float x, float z) {
    auto ss = [](float a, float b, float v) { float t = std::clamp((v - a) / (b - a), 0.f, 1.f); return t * t * (3 - 2 * t); };
    float n = vnoiseG(x * 0.06f, z * 0.06f) * 0.65f + vnoiseG(x * 0.21f, z * 0.21f) * 0.35f;
    float path = fabsf(z - 9.f * sinf(x * 0.045f) - 4.f * sinf(x * 0.11f + 1.f));
    return ss(0.26f, 0.42f, n) * ss(0.9f, 1.7f, path);
}

// Ground virtual texture: queue the tiles each level's window needs (coarse levels and nearby tiles first), and
// work out per level the rectangle that is fully resident for the shaders. Runs every frame before the UBO fill;
// the tiles queued here are generated by the "ground VT" pass ahead of all shading in the same frame.
static void updateGroundVT(float (*rect)[4]) {
    if (S.fractalPebbles != vtGenKey[0] || S.groundWarp != vtGenKey[1]) {
        vtGenKey[0] = S.fractalPebbles; vtGenKey[1] = S.groundWarp;
        vtVersion++;  // out of date: still shown (and resident) until regenerated
    }
    VtTileReq* out = (VtTileReq*)vtTileBuf.map;
    uint32_t n = 0, budget = (uint32_t)std::clamp(S.vtBudget, 1, 4096);
    bool generate = S.groundMode == 3;
    for (int pass = 0; pass < 2; pass++)  // missing tiles, then out-of-date ones
        for (int l = VT_L - 1; l >= 0; l--) {
            double tw = VT_TEXEL0 * (double)(1 << l) * VT_TILE;  // tile width (m)
            int cx = (int)floor(cam.pos.x / tw + 0.5) - VT_T / 2, cz = (int)floor(cam.pos.z / tw + 0.5) - VT_T / 2;
            // nearest first
            int order[VT_T * VT_T];
            for (int i = 0; i < VT_T * VT_T; i++) order[i] = i;
            auto d2 = [&](int i) { float dx = (i % VT_T) + 0.5f - VT_T / 2.f, dz = (i / VT_T) + 0.5f - VT_T / 2.f; return dx * dx + dz * dz; };
            std::sort(order, order + VT_T * VT_T, [&](int a, int b) { return d2(a) < d2(b); });
            for (int oi = 0; oi < VT_T * VT_T && generate && n < budget; oi++) {
                int tx = cx + order[oi] % VT_T, tz = cz + order[oi] / VT_T;
                uint32_t sx = (uint32_t)(tx & (VT_T - 1)), sz = (uint32_t)(tz & (VT_T - 1));
                VtSlot& sl = vtSlots[l][sz * VT_T + sx];
                bool missing = sl.tx != tx || sl.tz != tz;
                if (pass == 0 ? !missing : (missing || sl.ver == vtVersion)) continue;
                out[n++] = {(uint32_t)l, sx | sz << 8, tx, tz};
                sl = {tx, tz, vtVersion};
            }
            // resident rectangle: start with the window, cut sides until no missing tile is left inside
            int x0 = cx, x1 = cx + VT_T, z0 = cz, z1 = cz + VT_T;  // [x0, x1) tiles
            for (bool again = true; again && x0 < x1 && z0 < z1;) {
                again = false;
                for (int tz = z0; tz < z1 && !again; tz++)
                    for (int tx = x0; tx < x1 && !again; tx++) {
                        const VtSlot& sl = vtSlots[l][(tz & (VT_T - 1)) * VT_T + (tx & (VT_T - 1))];
                        if (sl.tx == tx && sl.tz == tz) continue;
                        // cut the side nearest this tile
                        int dl = tx - x0, dr = x1 - 1 - tx, db = tz - z0, dt = z1 - 1 - tz, m = std::min({dl, dr, db, dt});
                        if (m == dl) x0 = tx + 1; else if (m == dr) x1 = tx; else if (m == db) z0 = tz + 1; else z1 = tz;
                        again = true;
                    }
            }
            float tex = (float)(VT_TEXEL0 * (double)(1 << l));
            if (x0 < x1 && z0 < z1) {
                rect[l][0] = (float)(x0 * tw) + tex; rect[l][1] = (float)(z0 * tw) + tex;
                rect[l][2] = (float)(x1 * tw) - tex; rect[l][3] = (float)(z1 * tw) - tex;
            } else rect[l][0] = rect[l][1] = 1e9f, rect[l][2] = rect[l][3] = -1e9f;
        }
    vtQueued = n;
}

static void updateFrameData() {
    PROF_FUNC();
    UBO u{};
    auto camVecs = [](const Camera& c, float* pos, float* r, float* up, float* f, float time) {
        float aspect = (float)W / H, th = tanf(c.fovDeg * 0.5f * 3.14159265f / 180.f);
        set4(pos, c.pos, time);
        set4(r, c.right(), aspect);
        set4(up, c.up(), th);
        set4(f, c.fwd(), 0);
    };
    camVecs(cam, u.camPos, u.camRight, u.camUp, u.camFwd, (float)simTime);
    camVecs(prevCam, u.prevCamPos, u.prevRight, u.prevUp, u.prevFwd, 0);

    // sun and sky share one physical model: sky LUT is per unit sun irradiance, sun colour = irradiance * transmittance
    V3 sd = sunDirection();
    set4(u.sunDir, sd, cosf(S.sunRadiusDeg * 3.14159265f / 180.f));
    set4(u.sunColor, sunTransmittance(sd) * S.sunIntensity, S.skyIntensity);
    float wd = S.windDirDeg * 3.14159265f / 180.f;
    u.wind[0] = cosf(wd); u.wind[1] = sinf(wd); u.wind[2] = S.windStrength; u.wind[3] = S.windSpeed;
    u.params[0] = S.exposure; u.params[1] = (float)S.historyMax; u.params[2] = S.fireflyClamp; u.params[3] = S.spatialRadius;
    u.res[0] = W; u.res[1] = H; u.res[2] = frameIndex; u.res[3] = 0;  // light count, filled below
    u.settings[0] = (uint32_t)S.raysFirst; u.settings[1] = (uint32_t)S.bounces; u.settings[2] = flags(); u.settings[3] = (uint32_t)S.viewMode;
    u.params2[0] = S.giRadius; u.params2[1] = S.giMaxM; u.params2[2] = S.sunIntensity; u.params2[3] = S.fogScale;
    set4(u.clouds, V3{S.cloudCoverage * S.cumulus, S.cloudDensity, S.cloudSpeed}, S.clouds ? 1.f : 0.f);
    set4(u.clouds2, V3{S.cloudBase, S.cloudThickness, S.cloudShadow}, 0);
    set4(u.clouds3, V3{S.cloudCoverage * S.stratus, S.stratusLumpy, S.stratusBase}, S.stratusThick);
    set4(u.clouds4, V3{S.cloudCoverage * S.cumulonimbus, S.cbTop, S.cloudCoverage * S.cirrus}, S.cirrusAlt);
    set4(u.clouds5, V3{S.cirrusLength, S.cirrusFine, S.cirrusPuff}, 0);
    set4(u.grassC, V3{S.grassClump, S.grassClumpSize, S.grassWidth}, S.grassCurve);
    set4(u.grassD, V3{S.grassTwist, S.grassTilt, S.grassHeightVar}, S.grassDead);
    set4(u.ground2, V3{S.groundWarp, S.groundMottle, S.groundMottleScale}, S.darkSpots);
    set4(u.ground3, V3{S.darkSpotSize, S.groundGrit, S.groundCavity}, S.tileBreakup);
    set4(u.vegCol, V3{S.grassColorVar, S.groundSat, 0}, 0);
    for (int i = 0; i < 4; i++) u.groundTex[i] = groundSlots[i];
    u.groundP[0] = 4.f; u.groundP[1] = (float)groundTexSize; u.groundP[2] = groundMaxH[0]; u.groundP[3] = groundMaxH[1];
    u.groundS[0] = S.groundNormal; u.groundS[1] = S.parallax ? 1.f : 0.f; u.groundS[2] = S.parallaxDist;
    u.groundS[3] = (groundReady || S.groundMode == 3) && S.groundTextures ? 1.f : 0.f;
    updateGroundVT(u.vtRect);
    u.vtP[0] = VT_TEXEL0; u.vtP[1] = (float)VT_N; u.vtP[2] = (float)VT_L; u.vtP[3] = S.groundMode == 3 ? 1.f : 0.f;
    {   // wind map window follows the camera in whole cells; cells that scroll in are re-initialised by wind.comp
        static int prevOrigin[2] = {INT32_MIN / 2, INT32_MIN / 2};
        static double lastT = simTime;
        int ox = (int)floorf(cam.pos.x) - 128, oz = (int)floorf(cam.pos.z) - 128;
        u.windGrid[0] = ox; u.windGrid[1] = oz; u.windGrid[2] = prevOrigin[0]; u.windGrid[3] = prevOrigin[1];
        prevOrigin[0] = ox; prevOrigin[1] = oz;
        float fdt = std::clamp((float)(simTime - lastT), 0.f, 1.f / 20.f);
        lastT = simTime;
        u.windSim[0] = fdt * 0.5f; u.windSim[1] = S.gustStrength; u.windSim[2] = S.gustSpreadDeg * 3.14159265f / 180.f; u.windSim[3] = S.windMap ? 1.f : 0.f;
        u.windSim2[0] = S.windWaveSpeed * S.windWaveSpeed * 4.f; u.windSim2[1] = S.windDamping; u.windSim2[2] = S.windRelax; u.windSim2[3] = (float)simTime;
    }

    // grass grid around the camera + view cone (fov + margin) used for culling
    const float d2r = 3.14159265f / 180.f;
    float cs = 1.f / sqrtf(S.grassDensity), R = S.drawDist;
    grassGridN = (uint32_t)ceilf(2 * R / cs) + 1;
    float vfov = cam.fovDeg * d2r, hfov = 2 * atanf(tanf(vfov / 2) * W / H), m = S.fovMargin * d2r;
    u.grassA[0] = cs; u.grassA[1] = S.grassLength; u.grassA[2] = S.lodStart; u.grassA[3] = R;
    u.grassB[0] = tanf(std::min((hfov + m) / 2, 1.5f)); u.grassB[1] = tanf(std::min((vfov + m) / 2, 1.5f)); u.grassB[2] = S.keepRadius; u.grassB[3] = S.offscreenDensity;
    u.grassGrid[0] = (int)floorf((cam.pos.x - R) / cs); u.grassGrid[1] = (int)floorf((cam.pos.z - R) / cs);
    {
        V3 f = cam.fwd();
        float dx = cam.pos.x - grassBuildPos[0], dy = cam.pos.y - grassBuildPos[1], dz = cam.pos.z - grassBuildPos[2];
        bool moved = dx * dx + dy * dy + dz * dz > 1.f || f.x * grassBuildFwd[0] + f.y * grassBuildFwd[1] + f.z * grassBuildFwd[2] < 0.996f;  // 1 m, ~5 deg
        grassRefit = S.grassRefit && grassBuilt && !moved && !terrainDirty && ++grassFrames < GRASS_REBUILD;
        if (grassRefit) bladeCap = grassBuildCap;  // an update keeps the primitive count
        else {
            grassFrames = 0;
            grassBuildCap = bladeCap;
            grassBuildPos[0] = cam.pos.x; grassBuildPos[1] = cam.pos.y; grassBuildPos[2] = cam.pos.z;
            grassBuildFwd[0] = f.x; grassBuildFwd[1] = f.y; grassBuildFwd[2] = f.z;
        }
    }
    u.grassGrid[2] = (int)grassGridN; u.grassGrid[3] = (int)bladeCap;

    // terrain grid recentred in 0.5 m steps
    float cx = roundf(cam.pos.x * 2) / 2, cz = roundf(cam.pos.z * 2) / 2;
    if (cx != terrainCenter[0] || cz != terrainCenter[1]) { terrainCenter[0] = cx; terrainCenter[1] = cz; terrainDirty = true; }
    u.terrainP[0] = cx; u.terrainP[1] = cz; u.terrainP[2] = TERRAIN_INNER; u.terrainP[3] = std::max(S.viewDist - TERRAIN_INNER, 100.f);
    u.counts[0] = (uint32_t)std::min<size_t>(stream::zones().size(), MAX_ZONES); u.counts[1] = (uint32_t)S.giSpatialSamples; u.counts[2] = (uint32_t)S.candidates; u.counts[3] = (uint32_t)S.spatialSamples;
    // sub-pixel jitter for DLSS-RR: Halton(2,3), 64 phases (RR wants many), in render pixels
    prevJitter[0] = jitter[0]; prevJitter[1] = jitter[1];
    if (rrActive) {
        auto halton = [](uint32_t i, uint32_t b) { float f = 1, r = 0; for (i++; i; i /= b) { f /= b; r += f * (i % b); } return r; };
        jitter[0] = halton(frameIndex % 64, 2) - 0.5f;
        jitter[1] = halton(frameIndex % 64, 3) - 0.5f;
    } else jitter[0] = jitter[1] = 0;
    u.river[0] = gRiver.width; u.river[1] = gRiver.depth; u.river[2] = gRiver.level; u.river[3] = gRiver.slope;
    u.water[0] = S.flow; u.water[1] = S.clarity; u.water[2] = S.foam; u.water[3] = S.splash;
    u.water2[0] = S.foamTrail; u.water2[1] = std::max(S.viewDist - TERRAIN_INNER, 100.f); u.water2[2] = S.treeDensity; u.water2[3] = gRiver.bumps;
    u.waterCol[0] = S.murk[0]; u.waterCol[1] = S.murk[1]; u.waterCol[2] = S.murk[2]; u.waterCol[3] = S.absorb;
    u.dynInfo[0] = (uint32_t)thrown.size();
    for (size_t i = 0; i < thrown.size(); i++) {
        const Thrown& t = thrown[i];
        float d[4] = {t.xf.m[0][3], t.xf.m[1][3], t.xf.m[2][3], t.radius / 0.85f};
        memcpy(u.dynRocks[i], d, sizeof d);
        float v[4] = {t.v[0], t.v[1], t.v[2], 0.f};
        memcpy(u.dynRockV[i], v, sizeof v);
    }
    u.dynInfo[1] = (uint32_t)std::min(splashEvents.size(), (size_t)MAX_SPLASH_EV);
    for (uint32_t i = 0; i < u.dynInfo[1]; i++) {
        const SplashEvent& e = splashEvents[i];
        float d[8] = {e.x, e.z, e.t, e.impact, e.sc, e.speed, 0, 0};
        memcpy(u.splashEv[2 * i], d, sizeof d);
    }
    u.dynInfo[2] = (uint32_t)std::min(gCraterCount.load(), MAX_CRATERS);
    memcpy(u.craters, gCraters, sizeof(Crater) * u.dynInfo[2]);
    for (uint32_t i = 0; i < u.dynInfo[2]; i++) {
        const CraterWater& w = craterWater[i];
        u.craterG[i][0] = w.ground[0]; u.craterG[i][1] = w.ground[1]; u.craterG[i][2] = w.ground[2]; u.craterG[i][3] = 0.f;
        u.craterW[i][0] = w.pond ? w.level : -1e9f; u.craterW[i][1] = w.fill; u.craterW[i][2] = w.divert; u.craterW[i][3] = w.inflowSpeed;
    }
    u.jitter[0] = jitter[0]; u.jitter[1] = jitter[1]; u.jitter[2] = prevJitter[0]; u.jitter[3] = prevJitter[1];
    {   // fog, weather, decals, dust
        // the fog pass also runs (without fog) while there's dust to show
        set4(u.fog0, V3{S.fog ? S.fogDensity : 0.f, S.fogFalloff, S.fogAniso}, S.fog || !dustList.empty() ? 1.f : 0.f);
        set4(u.fog1, V3{S.fogLights ? 1.f : 0.f, S.fogDist, S.fogNoise}, S.fogBlend);
        static double lastWetT = simTime;
        float wdt = std::clamp((float)(simTime - lastWetT), 0.f, 0.25f);
        lastWetT = simTime;
        if (S.rainWets) {  // soaks in within ~half a minute of rain, dries over a few minutes
            float target = std::min(S.rain * 1.5f, 1.f);
            S.wetness += (target - S.wetness) * (1.f - expf(-wdt * (target > S.wetness ? 0.08f : 0.01f)));
        }
        u.weather[0] = S.wetness; u.weather[1] = S.rain; u.weather[2] = S.puddles; u.weather[3] = S.caustics;
        // world map + simulation
        for (int i = 0; i < 8; i++) {
            const std::string& nm = worldmap::paletteName(i);
            u.mapPal[i] = nm.empty() ? 0u : zoneTexSlot(nm);
            u.mapPalTile[i] = S.mapTexTile;
        }
        u.mapP[0] = worldmap::ORIGIN; u.mapP[1] = worldmap::CELL; u.mapP[2] = (float)worldmap::N; u.mapP[3] = 0.f;
        if (S.snowFollowsClouds) S.snowLine = S.cloudBase;
        u.sim0[0] = S.snowLine; u.sim0[1] = S.rain; u.sim0[2] = S.meltRate / 3600.f; u.sim0[3] = S.riverRange;
        u.sim1[0] = tanf(S.reposeDeg * 3.14159265f / 180.f); u.sim1[1] = S.avalanche; u.sim1[2] = std::clamp(frameDt * S.simSpeed, 0.001f, 0.5f);
        u.sim1[3] = S.worldSim ? 1.f : 0.f;
        u.sim2[0] = S.snowfall / 3600.f; u.sim2[1] = S.dryRate; u.sim2[2] = S.waterMobility; u.sim2[3] = riverSwell;
        memcpy(u.simBrush, gSimBrush, 16);
        buildCraterBins(cam.pos.x, cam.pos.z);
        int fd = std::clamp(S.fogDiv, 1, 4);
        u.fogRes[0] = fd; u.fogRes[1] = ((int)W + fd - 1) / fd; u.fogRes[2] = ((int)H + fd - 1) / fd; u.fogRes[3] = 0;
        u.craterGrid[0] = cbOrigin[0]; u.craterGrid[1] = cbOrigin[1]; u.craterGrid[2] = CB_N; u.craterGrid[3] = cbValid ? 1 : 0;
        u.fractal[0] = (float)S.groundMode; u.fractal[1] = S.fractalRelief; u.fractal[2] = S.fractalPebbles; u.fractal[3] = S.fractalFinest;
        uint32_t nd = 0;
        for (const Decal& d : decalList) {
            float age = std::min((float)(simTime - d.t0), 999.f);
            if (age > 300.f) continue;
            float a[8] = {d.p[0], d.p[1], d.p[2], d.r, d.n[0], d.n[1], d.n[2], (float)d.kind + age / 1000.f};
            memcpy(u.decals[2 * nd], a, sizeof a);
            nd++;
        }
        u.lightInfo[3] = nd;
        dustList.erase(std::remove_if(dustList.begin(), dustList.end(), [](const Dust& d) { return simTime - d.t0 > 15.0; }), dustList.end());
        uint32_t nu = 0;
        for (const Dust& d : dustList) {
            float a[8] = {d.p[0], d.p[1], d.p[2], d.r, d.density, (float)(simTime - d.t0), d.rise, 0};
            memcpy(u.dust[2 * nu], a, sizeof a);
            nu++;
        }
        u.fxInfo[0] = nu;
    }
    {   // post
        static double lastPostT = simTime;
        float pdt = std::clamp((float)(simTime - lastPostT), 0.f, 0.25f);
        lastPostT = simTime;
        u.post0[0] = S.evComp; u.post0[1] = S.autoExposure ? 1.f : 0.f; u.post0[2] = S.adaptUp; u.post0[3] = S.adaptDown;
        float ap = S.physCam ? std::clamp(4.f / S.fStop, 0.3f, 1.5f) : 1.f;  // wide open vignettes more
        u.post1[0] = S.bloom; u.post1[1] = 0; u.post1[2] = S.vignette * ap; u.post1[3] = S.chromAb;
        u.post2[0] = S.grain * (S.physCam ? sqrtf(S.iso / 100.f) : 1.f); u.post2[1] = S.sharpen; u.post2[2] = (float)S.tonemapper; u.post2[3] = pdt;
        float focal = 0.012f / tanf(cam.fovDeg * 0.5f * 3.14159265f / 180.f);  // 24 mm tall sensor
        u.post3[0] = S.focusDist; u.post3[1] = focal / std::max(S.fStop, 0.5f);
        u.post3[2] = !S.motionBlur ? 0.f : S.physCam ? std::clamp((1.f / S.shutterDen) / std::max(frameDt, 1e-3f), 0.f, 2.f) : S.mbShutter;
        u.post3[3] = S.dof ? 1.f : 0.f;
        if (S.physCam) {  // EV100 = log2(N^2 / t) - log2(ISO / 100); exposure = 1 / (1.2 * 2^EV100) in cd/m^2
            float ev100 = log2f(S.fStop * S.fStop * S.shutterDen) - log2f(S.iso / 100.f);
            u.params[0] = S.sceneLum / (1.2f * exp2f(ev100));
            u.post0[1] = 0.f;
        }
        u.post4[0] = (float)hdrActive; u.post4[1] = S.paperWhite; u.post4[2] = S.peakNits; u.post4[3] = (float)OW / OH;
    }
    memcpy(uboBuf.map, &u, sizeof u);

    float* L = (float*)lightBuf.map;
    float t = (float)simTime;
    int orbs = std::min(S.lightCount, MAX_ORBS);
    for (int i = 0; i < orbs; i++) {
        auto& b = lightBase[i];
        const float R = S.orbRange;
        auto wrap = [R](float v) { return v - R * floorf(v / R + 0.5f); };
        float x = cam.pos.x + wrap(b.p.x * R + sinf(t * 0.31f + b.phase) * 1.6f - cam.pos.x);
        float z = cam.pos.z + wrap(b.p.z * R + cosf(t * 0.27f + b.phase * 1.3f) * 1.6f - cam.pos.z);
        float y = heightFn(x, z) + b.p.y + sinf(t * 0.83f + b.phase * 1.7f) * 0.35f;
        // fade out (light and orb size) toward the tile edge, where an orb wraps to the far side: no pop
        float e = std::max(fabsf(x - cam.pos.x), fabsf(z - cam.pos.z)) / (0.5f * R);
        float fade = 1.f - std::clamp((e - 0.75f) / 0.25f, 0.f, 1.f);
        fade = fade * fade * (3.f - 2.f * fade);
        orbFade[i] = fade;
        V3 c = b.col * (b.inten * S.lightIntensity * fade);
        float v[8] = {x, y, z, 0.06f * fade, c.x, c.y, c.z, 0};
        memcpy(L + i * 8, v, sizeof v);
    }
    // streamed lights: keep the strongest at the camera (radius 0 = not drawn as an orb)
    size_t nw = wlx.size(), room = MAX_LIGHTS - orbs;
    std::vector<uint32_t> pick;
    if (nw) {
        lightScores(wlx.data(), wly.data(), wlz.data(), wlLum.data(), nw, cam.pos.x, cam.pos.y, cam.pos.z, wlScore.data());
        pick.resize(nw);
        for (uint32_t i = 0; i < nw; i++) pick[i] = i;
        if (nw > room) {
            std::nth_element(pick.begin(), pick.begin() + room, pick.end(), [](uint32_t a, uint32_t b) { return wlScore[a] > wlScore[b]; });
            pick.resize(room);
        }
    }
    auto& wl = stream::lights();
    // switched-off lights go after the active ones: only the probe bake (lights-on state) samples them
    std::stable_partition(pick.begin(), pick.end(), [&](uint32_t i) { return !lightOff(wl[i].pos); });
    uint32_t active = 0;
    for (size_t k = 0; k < pick.size(); k++) {
        const WorldLight& w = wl[pick[k]];
        active += !lightOff(w.pos);
        // radius 4 cm: drawn as a small glowing bulb (hidden inside lamp models that have their own glowing mesh)
        float v[8] = {w.pos[0], w.pos[1], w.pos[2], 0.04f, w.color[0] * lightScale, w.color[1] * lightScale, w.color[2] * lightScale, 0};
        memcpy(L + (orbs + k) * 8, v, sizeof v);
    }
    numLights = orbs + active;
    numWorldLights = active;
    bakeOffLights = (uint32_t)pick.size() - active;
    buildLightTree(L);
    UBO* ub = (UBO*)uboBuf.map;
    ub->res[3] = numLights;
    ub->bake[0] = (uint32_t)bakeVols.size();
    ub->bake[1] = (uint32_t)orbs;
    ub->bake[2] = bakedGI ? 1u : 0u;
    ub->bake[3] = numLights + bakeOffLights;
    ub->lightInfo[0] = numTriLights;
    ub->lightInfo[1] = numTreeNodes;
    ub->lightInfo[2] = allEmittersListed ? 1u : 0u;
    writeBakeVolumes();

    // rocks: hashed 7 m grid out to the terrain's edge, TLAS instances + raster transforms. No fixed radius: each
    // rock is kept while it covers >= ~1.5 output pixels (screen-size LOD), so big boulders stay visible across the
    // field and only rocks too small to resolve drop out, where the pop is invisible.
    const float RC = 7.f, RR = std::max(S.viewDist - TERRAIN_INNER, 100.f);
    const float pixAngle = 2.f * tanf(cam.fovDeg * 0.5f * 3.14159265f / 180.f) / (float)OH, keepPx = 1.5f;
    // Placement is ~70k grid cells of noise + height lookups (several ms): redo it only after moving 2 m or a
    // world / setting change. The TLAS instances and raster transforms stay valid in their buffers meanwhile.
    static V3 placedAt{1e9f, 0, 0};
    static uint32_t placedModels = UINT32_MAX;
    static float placedDensity = -1.f, placedPix = 0.f, placedView = 0.f;
    V3 mv = cam.pos - placedAt;
    float step = std::max(2.f, S.viewDist / 300.f);  // a far view costs more to place: redo it less often (it's far away)
    if (!placeDirty && dot(mv, mv) < step * step && placedModels == numModelInst && placedDensity == S.treeDensity && placedPix == pixAngle &&
        placedView == S.viewDist)
        return;
    placedView = S.viewDist;
    placeDirty = false;
    placedAt = cam.pos;
    placedModels = numModelInst;
    placedDensity = S.treeDensity;
    placedPix = pixAngle;
    auto* inst = (VkAccelerationStructureInstanceKHR*)instBuf.map;
    float* xf = (float*)rockXf.map;
    const auto& zs = stream::zones();
    // grid rows run on the job workers, each into its own list; the lists are then appended in row order
    struct Placed { float m[3][4]; TreeInst t; };
    auto placeRows = [&](float cell, float range, const std::function<bool(int, int, Placed&)>& fn) {
        int z0 = (int)floorf((cam.pos.z - range) / cell), z1 = (int)floorf((cam.pos.z + range) / cell);
        int x0 = (int)floorf((cam.pos.x - range) / cell), x1 = (int)floorf((cam.pos.x + range) / cell);
        std::vector<std::vector<Placed>> rows(z1 - z0 + 1);
        jobs::parallelFor((uint32_t)rows.size(), 4, [&](uint32_t b, uint32_t e) {
            for (uint32_t r = b; r < e; r++)
                for (int ix = x0; ix <= x1; ix++) {
                    Placed pl;
                    if (fn(ix, z0 + (int)r, pl)) rows[r].push_back(pl);
                }
        });
        return rows;
    };

    numRocks = 0;
    auto rocks = placeRows(RC, RR, [&](int ix, int iz, Placed& pl) {
        float h = hash2(ix * 31 + 7, iz * 17 + 3);
        if (h > 0.16f) return false;
        float x = (ix + hash2(ix + 911, iz)) * RC, z = (iz + hash2(ix, iz + 577)) * RC;
        float sr = hash2(ix + 3, iz + 5);
        float sc = 0.25f + 0.9f * sr * sr + (h < 0.015f ? 1.8f : 0.f);
        float far = std::min(RR, 2.f * sc / (pixAngle * keepPx));  // rock is ~2*sc across
        if ((x - cam.pos.x) * (x - cam.pos.x) + (z - cam.pos.z) * (z - cam.pos.z) > far * far) return false;
        if (zoneCover(x, z, zs.data(), (int)zs.size()) > 0.f || inCrater(x, z, 0.f)) return false;  // no rocks on city ground / in craters
        if (!brokenRocks.empty() && brokenRocks.count(rockKey(x, z))) return false;  // knocked loose
        float yaw = hash2(ix + 13, iz + 29) * 6.2831853f, c = cosf(yaw) * sc, sn = sinf(yaw) * sc;
        float m[3][4] = {{c, 0, sn, x}, {0, sc, 0, heightFn(x, z) - 0.3f * sc}, {-sn, 0, c, z}};
        memcpy(pl.m, m, sizeof m);
        return true;
    });
    for (auto& row : rocks)
        for (auto& pl : row) {
            if (numRocks >= (uint32_t)MAX_ROCKS) break;
            uint32_t var = rockVarOf(pl.m[0][3], pl.m[2][3]);
            setRockInstance(inst[INST_MODELS + numModelInst + numRocks], var, pl.m);
            writeRockXf(xf, numRocks, pl.m, pl.m, var);
            numRocks++;
        }
    // buried rocks: the ground holds stones everywhere (a hashed 1.5 m grid, each at its own depth below the natural
    // surface), seen only where a crater has dug down to them. Deterministic, so a crater uncovers the same stones
    // wherever it is dug, and a deeper crater uncovers more.
    {
        const float BC = 1.5f;
        int nc = std::min(gCraterCount.load(std::memory_order_relaxed), MAX_CRATERS);
        for (int ci = 0; ci < nc && numRocks < (uint32_t)MAX_ROCKS; ci++) {
            const Crater& cr = gCraters[ci];
            float dx = cr.x - cam.pos.x, dz = cr.z - cam.pos.z;
            if (dx * dx + dz * dz > RR * RR) continue;
            for (int iz = (int)floorf((cr.z - cr.r) / BC); iz <= (int)floorf((cr.z + cr.r) / BC); iz++)
                for (int ix = (int)floorf((cr.x - cr.r) / BC); ix <= (int)floorf((cr.x + cr.r) / BC); ix++) {
                    float h = hash2(ix * 41 + 9, iz * 23 + 17);
                    if (h > 0.6f) continue;
                    float x = (ix + hash2(ix + 101, iz + 7)) * BC, z = (iz + hash2(ix + 13, iz + 211)) * BC;
                    float cx = x - cr.x, cz = z - cr.z;
                    if (cx * cx + cz * cz > cr.r * cr.r) continue;
                    // owned by the first crater that covers it (overlapping craters don't double it)
                    bool earlier = false;
                    for (int k = 0; k < ci && !earlier; k++) {
                        float ex = x - gCraters[k].x, ez = z - gCraters[k].z;
                        earlier = ex * ex + ez * ez < gCraters[k].r * gCraters[k].r;
                    }
                    if (earlier) continue;
                    float sr = hash2(ix + 3, iz + 61);
                    float sc = 0.15f + 0.6f * sr * sr;
                    float burial = 0.25f + 2.2f * hash2(ix + 77, iz + 5);  // top of the stone below the natural ground
                    float ground = heightFn(x, z), natural = ground - craterCarve(x, z, 0.f);
                    float dug = natural - ground;
                    if (dug < burial - 0.2f * sc) continue;  // still buried
                    if (!brokenRocks.empty() && brokenRocks.count(rockKey(x, z))) continue;
                    // bedded in the crater floor where the digging stopped
                    float yaw = hash2(ix + 19, iz + 3) * 6.2831853f, c = cosf(yaw) * sc, sn = sinf(yaw) * sc;
                    float m[3][4] = {{c, 0, sn, x}, {0, sc, 0, ground - 0.35f * sc}, {-sn, 0, c, z}};
                    uint32_t var = rockVarOf(x, z);
                    setRockInstance(inst[INST_MODELS + numModelInst + numRocks], var, m);
                    writeRockXf(xf, numRocks, m, m, var);
                    numRocks++;
                }
        }
    }
    for (auto& pr : stream::props()) {  // hand-placed rocks
        if (pr.kind != PROP_ROCK || numRocks >= (uint32_t)MAX_ROCKS) continue;
        uint32_t var = rockVarOf(pr.xf.m[0][3], pr.xf.m[2][3]);
        setRockInstance(inst[INST_MODELS + numModelInst + numRocks], var, pr.xf.m);
        writeRockXf(xf, numRocks, pr.xf.m, pr.xf.m, var);
        numRocks++;
    }
    numBreakableRocks = numRocks;
    // river rocks (terrain.h riverRock): fixed layout the water shader also evaluates
    for (int i = (int)floorf((cam.pos.x - RR) / RIVER_ROCK_L); i <= (int)floorf((cam.pos.x + RR) / RIVER_ROCK_L) && numRocks < (uint32_t)MAX_ROCKS; i++) {
        float x, y, z, sc;
        if (!riverRock(i, x, y, z, sc)) continue;
        float far = std::min(RR, 2.f * sc / (pixAngle * keepPx));
        if ((x - cam.pos.x) * (x - cam.pos.x) + (z - cam.pos.z) * (z - cam.pos.z) > far * far) continue;
        float yaw = hash2(i, 863) * 6.2831853f, c = cosf(yaw) * sc, sn = sinf(yaw) * sc;
        float m[3][4] = {{c, 0, sn, x}, {0, sc, 0, y}, {-sn, 0, c, z}};
        uint32_t var = rockVarOf(x, z);
        setRockInstance(inst[INST_MODELS + numModelInst + numRocks], var, m);
        writeRockXf(xf, numRocks, m, m, var);
        numRocks++;
    }
    // placed rocks don't move: writeRockXf already set previous = current

    // trees: hashed 9 m grid; forests where a low-frequency noise is high, the odd lone tree elsewhere.
    // Same screen-size rule as rocks; the BVH gets the real mesh at every distance.
    numTrees = 0;
    treeList.clear();
    if (treeModel.gpu) {
        auto* g = (GpuModel*)treeModel.gpu;
        const float TC = 9.f;
        auto sstep = [](float a, float b, float x) { float t = std::clamp((x - a) / (b - a), 0.f, 1.f); return t * t * (3 - 2 * t); };
        // the tree a grid cell would hold, before camera / town / water rules: a pure function of the cell so
        // neighbours can be evaluated for the footprint test
        struct Cand { float x, z, sc; uint32_t v; };
        auto candidate = [&](int ix, int iz, Cand& cd) {
            float x = (ix + 0.15f + 0.7f * hash2(ix + 71, iz + 3)) * TC, z = (iz + 0.15f + 0.7f * hash2(ix + 5, iz + 191)) * TC;
            float forest = vnoise(x * 0.0045f + 31.f, z * 0.0045f - 7.f) * 0.7f + vnoise(x * 0.017f, z * 0.017f + 11.f) * 0.3f;
            float painted = worldmap::treeMul(x, z);  // world map tree brush: 0 = cleared, 2 = double, also outside forests
            if (hash2(ix * 13 + 5, iz * 7 + 1) > ((sstep(0.5f, 0.66f, forest) * 0.8f + 0.025f) * painted + std::max(painted - 1.f, 0.f) * 0.35f) * S.treeDensity)
                return false;
            float sc = 0.75f + 0.5f * hash2(ix + 19, iz + 41);
            uint32_t v;
            if (numTreeVariants == trees::VARIANTS)
{  // species stands: two low-frequency noises pick one of the four, the seed varies tree to tree
                uint32_t sp = (vnoise(x * 0.01f + 50.f, z * 0.01f) > 0.5f ? 2u : 0u) + (vnoise(x * 0.013f - 20.f, z * 0.013f + 9.f) > 0.5f ? 1u : 0u);
                v = sp * 2u + (hash2(ix + 7, iz + 77) > 0.5f ? 1u : 0u);
            }
            else {  // weighted pick from the forest mix
                float h = hash2(ix + 7, iz + 77) * treeWeightSum;
                v = (uint32_t)numTreeVariants - 1;
                for (int k = 0; k < numTreeVariants; k++) if ((h -= treeWeight[k]) < 0.f && treeWeight[k] > 0.f) { v = (uint32_t)k; break; }
                if (treeWeight[v] <= 0.f) return false;
            }
            cd = {x, z, sc, v};
            return true;
        };
        float maxFoot = 0.f;
        for (int k = 0; k < numTreeVariants; k++) maxFoot = std::max(maxFoot, treeFoot[k] * 1.25f * forestSpacing);
        const int reach = (int)ceilf(2.f * maxFoot / TC);
        auto treesP = placeRows(TC, RR, [&](int ix, int iz, Placed& pl) {
            Cand cd;
            if (!candidate(ix, iz, cd)) return false;
            if (!brokenTrees.empty() && brokenTrees.count(treeKey(ix, iz))) return false;  // knocked down
            float x = cd.x, z = cd.z, sc = cd.sc;
            uint32_t v = cd.v;
            // bounding footprints (x spacing) may not overlap: of two overlapping candidates the higher priority stays
            if (reach > 0) {
                float ra = treeFoot[v] * sc * forestSpacing, pa = hash2(ix + 911, iz + 37);
                for (int dz = -reach; dz <= reach; dz++)
                    for (int dx = -reach; dx <= reach; dx++) {
                        Cand o;
                        if ((!dx && !dz) || !candidate(ix + dx, iz + dz, o)) continue;
                        float rb = treeFoot[o.v] * o.sc * forestSpacing, rr = ra + rb;
                        if (rr <= 0.f || (o.x - x) * (o.x - x) + (o.z - z) * (o.z - z) >= rr * rr) continue;
                        float pb = hash2(ix + dx + 911, iz + dz + 37);
                        if (pb > pa || (pb == pa && (dx > 0 || (dx == 0 && dz > 0)))) return false;
                    }
            }
            float far = std::min(RR, 2.f * treeR[v] * sc / (pixAngle * 2.f));
            if ((x - cam.pos.x) * (x - cam.pos.x) + (z - cam.pos.z) * (z - cam.pos.z) > far * far) return false;
            if (zoneCover(x, z, zs.data(), (int)zs.size()) > 0.f || riverDry(x, z, 2.5f) == 0.f || inCrater(x, z, 0.f)) return false;  // not in town / water / craters
            float yaw = hash2(ix + 3, iz + 9) * 6.2831853f, c = cosf(yaw) * sc, sn = sinf(yaw) * sc, y = heightFn(x, z) - 0.05f;
            float m[3][4] = {{c, 0, sn, x}, {0, sc, 0, y}, {-sn, 0, c, z}};
            memcpy(pl.m, m, sizeof m);
            pl.t = {{x, y, z}, yaw, sc, v, ix, iz};
            return true;
        });
        for (auto& row : treesP)
            for (auto& pl : row) {
                if (numTrees >= (uint32_t)MAX_TREES) break;
                setInstance(inst[INST_MODELS + numModelInst + numRocks + numTrees], 3 + g->meshes[pl.t.v].slot, g->meshes[pl.t.v].blasAddr, pl.m);
                treeList.push_back(pl.t);
                numTrees++;
            }
        // hand-placed (editor / Blender: extras tree = <variant>); keyed outside the grid so breaking works the same
        for (auto& pr : stream::props()) {
            if (pr.kind != PROP_TREE || numTrees >= (uint32_t)MAX_TREES) continue;
            const Xf& x = pr.xf;
            float sc = sqrtf(x.m[0][0] * x.m[0][0] + x.m[1][0] * x.m[1][0] + x.m[2][0] * x.m[2][0]);
            float yaw = atan2f(x.m[0][2], x.m[2][2]);
            int ix = 0x40000000 | ((int)lroundf(x.m[0][3] * 10.f) & 0x3FFFFFFF), iz = (int)lroundf(x.m[2][3] * 10.f);
            if (!brokenTrees.empty() && brokenTrees.count(treeKey(ix, iz))) continue;
            uint32_t v = (uint32_t)std::clamp(pr.variant, 0, numTreeVariants - 1);
            float c = cosf(yaw) * sc, sn = sinf(yaw) * sc;
            float m[3][4] = {{c, 0, sn, x.m[0][3]}, {0, sc, 0, x.m[1][3]}, {-sn, 0, c, x.m[2][3]}};
            setInstance(inst[INST_MODELS + numModelInst + numRocks + numTrees], 3 + g->meshes[v].slot, g->meshes[v].blasAddr, m);
            treeList.push_back({{x.m[0][3], x.m[1][3], x.m[2][3]}, yaw, sc, v, ix, iz});
            numTrees++;
        }
    }

    // plants: hashed 2.5 m grid within PLANT_RANGE. Cattail beds in damp patches, dandelion meadows, roses and
    // shrubs scattered everywhere (thicker along forest edges); dropped once under ~3 px like rocks.
    numPlants = 0;
    plantList.clear();
    if (plantModel.gpu) {
        auto* g = (GpuModel*)plantModel.gpu;
        const float PC = 2.5f;
        auto sstepP = [](float a, float b, float x) { float t = std::clamp((x - a) / (b - a), 0.f, 1.f); return t * t * (3 - 2 * t); };
        uint32_t base = INST_MODELS + numModelInst + numRocks + numTrees;
        auto plantsP = placeRows(PC, PLANT_RANGE, [&](int ix, int iz, Placed& pl) {
            float x = (ix + hash2(ix + 401, iz + 13)) * PC, z = (iz + hash2(ix + 17, iz + 409)) * PC;
            float h = hash2(ix * 29 + 11, iz * 23 + 5) / std::max(S.treeDensity, 0.01f);
            float wet = vnoise(x * 0.02f + 90.f, z * 0.02f - 40.f), meadow = vnoise(x * 0.035f - 13.f, z * 0.035f + 77.f);
            float edge = vnoise(x * 0.0045f + 31.f, z * 0.0045f - 7.f);  // same low-frequency forest noise as the trees
            uint32_t v;
            float bank = 1e9f;
            if (riverDry(x, z, 3.5f) == 0.f) {  // riverside: reed beds right at the waterline, nothing in deeper water
                RiverInfo ri = riverInfo(x, z);
                bank = fabsf(ri.d) - ri.w;
                if (bank < -0.8f) return false;
            }
            if (bank < 3.5f && h < 0.6f * sstepP(0.35f, 0.6f, vnoise(x * 0.06f + 7.f, z * 0.06f))) v = trees::PLANT_CATTAIL;  // reed beds in stretches
            else if (bank < 0.5f) return false;
            else if (wet > 0.75f && h < 0.35f) v = trees::PLANT_CATTAIL;
            else if (meadow > 0.58f && h < 0.4f) v = trees::PLANT_DANDELION;
            else if (h < (0.025f + 0.08f * std::clamp((edge - 0.42f) * 6.f, 0.f, 1.f)) * S.bushAmount) {
                float k = hash2(ix + 3, iz + 61);
                v = k < S.berryShare ? trees::PLANT_BERRYBUSH : (k < S.berryShare + (1.f - S.berryShare) * 0.5f ? trees::PLANT_BUSH : trees::PLANT_FLOWERBUSH);
            }
            else if (h < 0.035f + 0.03f * meadow) v = trees::PLANT_ROSE;
            else return false;
            float sc = 0.75f + 0.5f * hash2(ix + 23, iz + 7);
            float far = std::min(PLANT_RANGE, 2.f * plantR[v] * sc / (pixAngle * 3.f));
            if ((x - cam.pos.x) * (x - cam.pos.x) + (z - cam.pos.z) * (z - cam.pos.z) > far * far) return false;
            if (zoneCover(x, z, zs.data(), (int)zs.size()) > 0.f || inCrater(x, z, 0.f)) return false;
            float yaw = hash2(ix + 5, iz + 37) * 6.2831853f, c = cosf(yaw) * sc, sn = sinf(yaw) * sc;
            float m[3][4] = {{c, 0, sn, x}, {0, sc, 0, heightFn(x, z) - 0.02f}, {-sn, 0, c, z}};
            memcpy(pl.m, m, sizeof m);
            pl.t.v = v;
            return true;
        });
        // ground clutter: denser 1.2 m grid close by. Ferns, twigs and leaf litter under the forest (same forest noise as
        // the trees), pebbles on bare ground and along the river banks.
        const float CC = 1.2f;
        auto clutterP = placeRows(CC, S.clutterRange, [&](int ix, int iz, Placed& pl) {
            if (S.clutter <= 0.f) return false;
            float x = (ix + hash2(ix + 911, iz + 37)) * CC, z = (iz + hash2(ix + 41, iz + 977)) * CC;
            float h = hash2(ix * 31 + 7, iz * 17 + 3) / S.clutter;
            float forest = vnoise(x * 0.0045f + 31.f, z * 0.0045f - 7.f);
            float bare = 1.f - grassMaskCPU(x, z);
            uint32_t v;
            float bank = 1e9f;
            if (riverDry(x, z, 2.f) == 0.f) {
                RiverInfo ri = riverInfo(x, z);
                bank = fabsf(ri.d) - ri.w;
                if (bank < -0.3f) return false;
            }
            if (bank < 2.5f && h < 0.5f) v = trees::PLANT_PEBBLES;
            else if (forest > 0.45f && h < 0.22f * (forest - 0.3f) * 4.f) {
                float k = hash2(ix + 71, iz + 5);
                v = k < 0.45f ? trees::PLANT_LITTER : (k < 0.75f ? trees::PLANT_TWIGS : trees::PLANT_FERN);
            } else if (h < 0.06f + 0.25f * bare) v = trees::PLANT_PEBBLES;
            else if (h < 0.09f + 0.25f * bare) v = trees::PLANT_TWIGS;
            else return false;
            float sc = 0.7f + 0.6f * hash2(ix + 29, iz + 11);
            float far = std::min(S.clutterRange, 2.f * plantR[v] * sc / (pixAngle * 2.f));
            if ((x - cam.pos.x) * (x - cam.pos.x) + (z - cam.pos.z) * (z - cam.pos.z) > far * far) return false;
            if (zoneCover(x, z, zs.data(), (int)zs.size()) > 0.f) return false;
            float yaw = hash2(ix + 3, iz + 53) * 6.2831853f, c = cosf(yaw) * sc, sn = sinf(yaw) * sc;
            float m[3][4] = {{c, 0, sn, x}, {0, sc, 0, heightFn(x, z) - 0.01f}, {-sn, 0, c, z}};
            memcpy(pl.m, m, sizeof m);
            pl.t.v = v;
            return true;
        });
        for (auto* rows : {&plantsP, &clutterP})
            for (auto& row : *rows)
                for (auto& pl : row) {
                    if (numPlants >= (uint32_t)MAX_PLANTS) break;
                    setInstance(inst[base + numPlants], 3 + g->meshes[pl.t.v].slot, g->meshes[pl.t.v].blasAddr, pl.m);
                    PlantInst pi;
                    memcpy(pi.m, pl.m, 48);
                    pi.v = pl.t.v;
                    plantList.push_back(pi);
                    numPlants++;
                }
    }
}

static void dispatch(VkCommandBuffer cb, int p, uint32_t gx, uint32_t gy, std::array<uint32_t, 4> pc = {0, 0, 0, 0}) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[p]);
    vkCmdPushConstants(cb, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc.data());
    vkCmdDispatch(cb, gx, gy, 1);
}

static void copyImg(VkCommandBuffer cb, int src, int dst) {
    VkImageCopy r{};
    r.srcSubresource = r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.extent = {W, H, 1};
    vkCmdCopyImage(cb, imgs[src].img, VK_IMAGE_LAYOUT_GENERAL, imgs[dst].img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
}

static void rasterGBuffer(VkCommandBuffer cb) {
    VkRenderingAttachmentInfo ca[GBUF_TARGETS];
    for (uint32_t i = 0; i < GBUF_TARGETS; i++) {
        ca[i] = {VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        ca[i].imageView = gbufImage(i).view;
        ca[i].imageLayout = VK_IMAGE_LAYOUT_GENERAL;
        ca[i].loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
        ca[i].storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        ca[i].clearValue.color = {{i == 0 ? -1.f : 0.f, 0, 0, 0}}; // depth -1 marks sky
    }
    VkRenderingAttachmentInfo da{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
    da.imageView = depthImg.view;
    da.imageLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
    da.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
    da.storeOp = VK_ATTACHMENT_STORE_OP_DONT_CARE;
    da.clearValue.depthStencil = {0.f, 0};
    VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
    ri.renderArea = {{0, 0}, {W, H}};
    ri.layerCount = 1;
    ri.colorAttachmentCount = GBUF_TARGETS;
    ri.pColorAttachments = ca;
    ri.pDepthAttachment = &da;
    vkCmdBeginRendering(cb, &ri);
    VkViewport vp{0, 0, (float)W, (float)H, 0, 1};
    VkRect2D sc{{0, 0}, {W, H}};
    vkCmdSetViewport(cb, 0, 1, &vp);
    vkCmdSetScissor(cb, 0, 1, &sc);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, pipeLayout, 0, 1, &descSet, 0, nullptr);
    VkDeviceSize off = 0;
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, rasterTerrain);
    vkCmdBindVertexBuffers(cb, 0, 1, &terrainV.b, &off);
    vkCmdBindIndexBuffer(cb, terrainI.b, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexed(cb, terrainTris * 3, 1, 0, 0, 0);
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, rasterGrass);
    vkCmdBindVertexBuffers(cb, 0, 1, &grassV.b, &off);
    vkCmdBindIndexBuffer(cb, grassI.b, 0, VK_INDEX_TYPE_UINT32);
    vkCmdDrawIndexedIndirect(cb, grassCounter.b, 4, 1, 0); // blade count from grass_tail.comp
    if (numRocks + numThrownRock) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, rasterRock);
        vkCmdBindVertexBuffers(cb, 0, 1, &rockV.b, &off);
        vkCmdBindIndexBuffer(cb, rockI.b, 0, VK_INDEX_TYPE_UINT32);
        vkCmdDrawIndexed(cb, rockTris * 3, numRocks + numThrownRock, 0, 0, 0);
    }
    if (!drawRuns.empty()) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, rasterModel);
        GpuModel* bound = nullptr;
        for (auto& r : drawRuns) {
            if (r.m != bound) { vkCmdBindIndexBuffer(cb, r.m->geom.b, 0, VK_INDEX_TYPE_UINT32); bound = r.m; }
            const GpuMesh& gm = r.m->meshes[r.mesh];
            vkCmdDrawIndexed(cb, gm.indexCount, r.count, gm.firstIndex, 0, r.first);
        }
    }
    if (numImpostors) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_GRAPHICS, rasterImpostor);
        vkCmdDraw(cb, 6, numImpostors, 0, 0);
    }
    vkCmdEndRendering(cb);
}

// The frame as a render graph (rendergraph.h): passes declare what they read / write; barriers, queue
// batches and timeline waits are derived. The BVH build runs on the async compute queue and overlaps the
// raster G-buffer, which only needs the grass vertices, not the acceleration structures.
static void buildFrameGraph(uint32_t si, ImDrawData* dd) {
    PROF_FUNC();
    bool grabUI = false;
    if (shotUIReq && swapCanCopy && !shotUIPending) {
        size_t need = (size_t)swapExtent.width * swapExtent.height * 4;
        if (!uiShotBuf.b || uiShotBuf.size < need) {
            if (uiShotBuf.b) freeBuf(uiShotBuf);
            uiShotBuf = mkBuf(need, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        }
        uiShotW = swapExtent.width; uiShotH = swapExtent.height;
        grabUI = shotUIPending = true;
        shotUIReq = false;
    }
    using namespace rg;
    constexpr VkPipelineStageFlags2 CS = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, XFER = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                                    ASB = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                    RASTER = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                             VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                    COLOR = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
    constexpr VkAccessFlags2 R = VK_ACCESS_2_SHADER_READ_BIT, WR = VK_ACCESS_2_SHADER_WRITE_BIT, RW = R | WR,
                             XR = VK_ACCESS_2_TRANSFER_READ_BIT, XW = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                             ASR = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR, ASW = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                             VTX = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | R,
                             CA = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
    Graph& g = graph;
    const uint32_t models = g.resource("streamed models"), grassCount = g.resource("grass counter"), grass = g.resource("grass geometry"),
                   terrain = g.resource("terrain"), skyR = g.resource("sky LUT"), accel = g.resource("BVH"), gbuf = g.resource("G-buffer"),
                   diRes = g.resource("DI reservoirs"), giRes = g.resource("GI reservoirs"), light = g.resource("lighting"),
                   denoise = g.resource("denoise"), history = g.resource("history"), out = g.resource("output"),
                   swap = g.resource("swapchain"), probesR = g.resource("probes"), windR = g.resource("wind map"),
                   hdr = g.resource("HDR scene"), expoR = g.resource("exposure"), bloomR = g.resource("bloom");
    const Queue bvhQ = asyncCompute ? COMPUTE : GFX, fogQ = asyncCompute && asyncFog ? COMPUTE : GFX;
    uint32_t gx = (W + 7) / 8, gy = (H + 7) / 8;
    auto bind = [](VkCommandBuffer cb) { vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &descSet, 0, nullptr); };

    // the uploads command buffer (stream.cpp models + their BLASes) is submitted first on the graphics queue
    g.add("uploads", GFX, [](VkCommandBuffer) {}).write(models, ASB | XFER, ASW | XW);

    if (resetHistory) {
        resetRR = true;
        g.add("reset history", GFX, [](VkCommandBuffer cb) {
            VkClearColorValue invalid{{-1, 0, 0, 0}}, zero{{0, 0, 0, 0}};
            VkImageSubresourceRange rr{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
            vkCmdClearColorImage(cb, imgs[I_PREVPOS].img, VK_IMAGE_LAYOUT_GENERAL, &invalid, 1, &rr);
            for (int i : {I_PREVNORM, I_HISTCOLOR, I_HISTMOM, I_ACCUM, I_ACCUMMOM, I_PINGA, I_PINGB})
                vkCmdClearColorImage(cb, imgs[i].img, VK_IMAGE_LAYOUT_GENERAL, &zero, 1, &rr);
            for (Buffer* b : {&resCur, &resPrev, &giCur, &giPrev, &radCacheBuf}) vkCmdFillBuffer(cb, b->b, 0, VK_WHOLE_SIZE, 0);
        }).write(history, XFER, XW).write(diRes, XFER, XW).write(giRes, XFER, XW).write(denoise, XFER, XW);
        resetHistory = false;
    }

    g.add("sky LUT", GFX, [=](VkCommandBuffer cb) { bind(cb); dispatch(cb, P_SKY, SKY_W / 8, SKY_H / 8); }).write(skyR, CS, WR);
    if (S.worldSim) {
        int N = worldmap::N, win = 1024;
        int ox = std::clamp((int)floorf((cam.pos.x - worldmap::ORIGIN) / worldmap::CELL) - win / 2, 0, N - win);
        int oz = std::clamp((int)floorf((cam.pos.z - worldmap::ORIGIN) / worldmap::CELL) - win / 2, 0, N - win);
        bool clear = simClear;
        simClear = false;
        int rbx = std::clamp((int)floorf((cam.pos.x - worldmap::ORIGIN) / worldmap::CELL) - SIM_RB / 2, 0, N - SIM_RB);
        int rbz = std::clamp((int)floorf((cam.pos.z - worldmap::ORIGIN) / worldmap::CELL) - SIM_RB / 2, 0, N - SIM_RB);
        simRBx = rbx; simRBz = rbz;
        g.add("world sim", GFX, [=](VkCommandBuffer cb) {
            if (clear) {
                VkClearColorValue z{{0, 0, 0, 0}};
                VkImageSubresourceRange rr{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
                vkCmdClearColorImage(cb, simImg.img, VK_IMAGE_LAYOUT_GENERAL, &z, 1, &rr);
                memBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            }
            bind(cb);
            for (uint32_t pass = 0; pass < 3; pass++) {
                dispatch(cb, P_WORLDSIM, win / 8, win / 8, {pass, (uint32_t)ox, (uint32_t)oz, 0});
                computeBarrier(cb);
            }
            {   // the physics' patch (read on the CPU after this frame's fence)
                VkBufferImageCopy c{};
                c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
                c.imageOffset = {rbx, rbz, 0};
                c.imageExtent = {SIM_RB, SIM_RB, 1};
                vkCmdCopyImageToBuffer(cb, simImg.img, VK_IMAGE_LAYOUT_GENERAL, simRB.b, 1, &c);
            }
            // every later pass may read the snow / wetness (terrain, shading, grass)
            memBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_ACCESS_SHADER_READ_BIT);
        }).write(terrain, CS, RW);
    }
    if (vtQueued) {
        uint32_t nt = vtQueued;
        g.add("ground VT", GFX, [=](VkCommandBuffer cb) {
            bind(cb);
            dispatch(cb, P_VTGEN, (VT_TILE / 16) * (VT_TILE / 16), nt);
            memBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,
                       VK_ACCESS_SHADER_READ_BIT);
        }).write(terrain, CS, RW);
    }
    if (terrainDirty)
        g.add("terrain", GFX, [=](VkCommandBuffer cb) { bind(cb); dispatch(cb, P_TERRAIN, (TNV * TNV + 63) / 64, 1); }).write(terrain, CS, WR);
    if (S.windMap)
        g.add("wind map", GFX, [=](VkCommandBuffer cb) {
            bind(cb);
            dispatch(cb, P_WIND, 256 / 8, 256 / 8, {0, 0, 0, 0});  // A -> B
            computeBarrier(cb);
            dispatch(cb, P_WIND, 256 / 8, 256 / 8, {1, 0, 0, 0});  // B -> A: readers always see A
        }).write(windR, CS, RW);
    bool refit = grassRefit;
    g.add("grass", GFX, [=](VkCommandBuffer cb) {
        if (refit) {  // same blades, same slots: re-animate in place
            bind(cb);
            dispatch(cb, P_GRASS, (bladeCap + 63) / 64, 1, {1, 0, 0, 0});
        } else {
            vkCmdFillBuffer(cb, grassCounter.b, 0, 4, 0);
            memBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            bind(cb);
            dispatch(cb, P_GRASS, (grassGridN + 7) / 8, (grassGridN + 7) / 8);
        }
        computeBarrier(cb);
        dispatch(cb, P_GRASS_TAIL, (bladeCap + 63) / 64, 1);
    }).read(terrain, CS, R).read(windR, CS, R).write(grassCount, CS | XFER, RW | XW).write(grass, CS, WR);
    g.add("blade count readback", GFX, [](VkCommandBuffer cb) {
        VkBufferCopy cc{0, 0, 4};
        vkCmdCopyBuffer(cb, grassCounter.b, countReadback.b, 1, &cc);
    }).read(grassCount, XFER, XR);

    if (numCharInst)  // graphics queue: the async BVH build waits on it anyway, the raster G-buffer needn't wait on the BVH
        g.add("skinning", GFX, [=](VkCommandBuffer cb) { bind(cb); recordSkinning(cb); }).write(models, CS | ASB, RW | ASW | ASR);
    bool rebuildTerrain = terrainDirty;
    g.add("BVH build", bvhQ, [rebuildTerrain](VkCommandBuffer cb) {
        bool saved = terrainDirty;
        terrainDirty = rebuildTerrain;
        cmdBuildScene(cb);
        terrainDirty = saved;
    }).read(grass, ASB, R).read(terrain, ASB, R).read(models, ASB, ASR).write(accel, ASB, ASW | ASR);
    terrainDirty = false;

    if (S.rasterPrimary)
        g.add("raster G-buffer", GFX, [](VkCommandBuffer cb) { rasterGBuffer(cb); })
            .read(grass, RASTER, VTX).read(grassCount, RASTER, VTX).read(terrain, RASTER, VTX).read(models, RASTER, VTX)
            .write(gbuf, COLOR, CA);
    g.add("primary", GFX, [=](VkCommandBuffer cb) { bind(cb); dispatch(cb, P_PRIMARY, gx, gy, {S.rasterPrimary ? 1u : 0u, 0, 0, 0}); })
        .read(accel, CS, ASR | R).read(skyR, CS, R).read(models, CS, R).read(grass, CS, R).read(terrain, CS, R).write(gbuf, CS, RW);
    const uint32_t fogR = g.resource("fog"), fogHistR = g.resource("fog history");
    g.add("surface fx", GFX, [=](VkCommandBuffer cb) { bind(cb); dispatch(cb, P_SURFACE_FX, gx, gy); })
        .read(accel, CS, ASR | R).write(gbuf, CS, RW);
    g.add("volumetric fog", fogQ, [=](VkCommandBuffer cb) { bind(cb); uint32_t fd = (uint32_t)std::clamp(S.fogDiv, 1, 4);
        dispatch(cb, P_VOLFOG, ((W + fd - 1) / fd + 7) / 8, ((H + fd - 1) / fd + 7) / 8); })
        .read(accel, CS, ASR | R).read(gbuf, CS, R).read(skyR, CS, R).read(fogHistR, CS, R).write(fogR, CS, WR);
    g.add("ReSTIR DI", GFX, [=](VkCommandBuffer cb) {
        bind(cb);
        dispatch(cb, P_RESTIR_INIT, gx, gy);
        computeBarrier(cb);
        dispatch(cb, P_RESTIR_SPATIAL, gx, gy);
    }).read(accel, CS, ASR | R).read(gbuf, CS, R).read(history, CS, R).read(skyR, CS, R).write(diRes, CS, RW).write(light, CS, WR);
    BakeJob job = nextBakeJob();
    if (job.vol >= 0)
        g.add("probe bake", GFX, [=](VkCommandBuffer cb) {
            bind(cb);
            if (!job.filter) {
                dispatch(cb, P_PROBE_BAKE, (job.n + 63) / 64, 1, {(uint32_t)job.vol, (uint32_t)job.st, job.first, job.pass});
                return;
            }
            dispatch(cb, P_PROBE_FILTER, (job.count + 63) / 64, 1, {(uint32_t)job.vol, (uint32_t)job.st, 0, 0});
            memBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            VkBufferCopy c{0, (VkDeviceSize)job.base * 64, (VkDeviceSize)job.count * 64};
            vkCmdCopyBuffer(cb, probeTmp.b, probeBuf.b, 1, &c);
        }).read(accel, CS, ASR | R).read(skyR, CS, R).read(models, CS, R).write(probesR, CS | XFER, RW | XW);
    g.add("GI", GFX, [=](VkCommandBuffer cb) {
        bind(cb);
        dispatch(cb, P_GI, gx, gy);
        if (S.restirGI) {
            computeBarrier(cb);
            dispatch(cb, P_GI_SPATIAL, gx, gy);
        }
        if (S.giCache) {
            computeBarrier(cb);
            dispatch(cb, P_RADCACHE, RC_ENTRIES / 64, 1);
        }
    }).read(accel, CS, ASR | R).read(gbuf, CS, R).read(history, CS, R).read(skyR, CS, R).read(probesR, CS, R).write(giRes, CS, RW).write(light, CS, RW);

    int iters = S.denoise ? S.atrousIters : 0;
    uint32_t finalImg = iters == 0 ? 0u : (iters % 2 ? 1u : 2u);
    if (rrActive) {
        // DLSS Ray Reconstruction replaces temporal accumulation + a-trous + composite
        const uint32_t rrIn = g.resource("RR inputs"), rrOut = g.resource("RR output");
        g.add("RR inputs", GFX, [=](VkCommandBuffer cb) { bind(cb); dispatch(cb, P_DLSS_PREP, gx, gy); })
            .read(fogR, CS, R).read(gbuf, CS, R).read(light, CS, R).read(skyR, CS, R).read(windR, CS, R).write(rrIn, CS, WR);
        bool reset = resetRR;
        resetRR = false;
        g.add("DLSS-RR", GFX, [=](VkCommandBuffer cb) {
            auto toRead = [&](VkImageLayout from, VkImageLayout to) {
                for (int i = 0; i < RR_OUT; i++)
                    imgBarrier(cb, rrImgs[i].img, from, to, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_SHADER_READ_BIT,
                               VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
            };
            toRead(VK_IMAGE_LAYOUT_GENERAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
            dlss::Inputs in{};
            auto img = [](const Image& im, int fmtIdx, uint32_t w, uint32_t h) { return dlss::Img{im.img, im.view, rrFormats[fmtIdx], w, h}; };
            in.color = img(rrImgs[RR_COLOR], RR_COLOR, W, H);
            in.diffuseAlbedo = img(rrImgs[RR_DIFF], RR_DIFF, W, H);
            in.specularAlbedo = img(rrImgs[RR_SPEC], RR_SPEC, W, H);
            in.normalRoughness = img(rrImgs[RR_NRM], RR_NRM, W, H);
            in.motion = img(rrImgs[RR_MV], RR_MV, W, H);
            in.viewZ = img(rrImgs[RR_VIEWZ], RR_VIEWZ, W, H);
            in.output = img(rrImgs[RR_OUT], RR_OUT, OW, OH);
            // DLSS wants the negated sample offset of cameraRay() on both axes. With either axis unflipped RR can't
            // cancel the jitter and the whole image shakes (measured on 8 consecutive still frames: 0.1-0.3 px
            // shifts per frame with Y unflipped, 0.00 px with both flipped).
            in.jitterX = -jitter[0];
            in.jitterY = -jitter[1];
            in.frameMs = cpuFrameMs;
            in.reset = reset;
            // row-vector matrices: view = [x y z 1] * worldToView (left-handed, +z forward), reverse-Z infinite projection
            V3 r = cam.right(), u = cam.up(), f = cam.fwd();
            float m[16] = {r.x, u.x, f.x, 0, r.y, u.y, f.y, 0, r.z, u.z, f.z, 0, -dot(r, cam.pos), -dot(u, cam.pos), -dot(f, cam.pos), 1};
            memcpy(in.worldToView, m, sizeof m);
            float th = tanf(cam.fovDeg * 0.5f * 3.14159265f / 180.f), aspect = (float)W / H;
            float pm[16] = {1.f / (th * aspect), 0, 0, 0, 0, 1.f / th, 0, 0, 0, 0, 0, 1, 0, 0, 0.05f, 0};
            memcpy(in.viewToClip, pm, sizeof pm);
            dlss::evaluate(cb, in);
            toRead(VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_IMAGE_LAYOUT_GENERAL);
        }).read(rrIn, CS, R).write(rrOut, CS, RW);
        g.add("RR output", GFX, [=](VkCommandBuffer cb) { bind(cb); dispatch(cb, P_DLSS_TONEMAP, (OW + 7) / 8, (OH + 7) / 8); })
            .read(rrOut, CS, R).write(hdr, CS, WR);
        g.add("history copy", GFX, [=](VkCommandBuffer cb) {
            copyImg(cb, I_GPOS, I_PREVPOS);
            copyImg(cb, I_GNORM, I_PREVNORM);
        }).read(gbuf, XFER, XR).write(history, XFER, XW);
    } else {
    g.add("denoise", GFX, [=](VkCommandBuffer cb) {
        bind(cb);
        dispatch(cb, P_TEMPORAL, gx, gy);
        uint32_t src = 0;
        for (int i = 0; i < iters; i++) {
            computeBarrier(cb);
            uint32_t dst = (i % 2 == 0) ? 1u : 2u;
            dispatch(cb, P_ATROUS, gx, gy, {1u << i, src, dst, i == 0 ? 1u : 0u});
            src = dst;
        }
    }).read(gbuf, CS, R).read(light, CS, R).read(history, CS, R).write(denoise, CS, RW);
    g.add("composite", GFX, [=](VkCommandBuffer cb) { bind(cb); dispatch(cb, P_COMPOSITE, gx, gy, {finalImg, 0, 0, 0}); })
        .read(fogR, CS, R).read(gbuf, CS, R).read(light, CS, R).read(denoise, CS, R).read(skyR, CS, R).read(windR, CS, R).write(hdr, CS, WR);
    g.add("history copy", GFX, [=](VkCommandBuffer cb) {
        copyImg(cb, I_GPOS, I_PREVPOS);
        copyImg(cb, I_GNORM, I_PREVNORM);
        copyImg(cb, I_ACCUMMOM, I_HISTMOM);
        if (iters == 0) copyImg(cb, I_ACCUM, I_HISTCOLOR);
    }).read(gbuf, XFER, XR).read(denoise, XFER, XR).write(history, XFER, XW);
    }

    g.add("fog history", GFX, [=](VkCommandBuffer cb) {
        VkImageCopy r{};
        r.srcSubresource = r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.extent = {W, H, 1};
        vkCmdCopyImage(cb, ximgs[X_FOG].img, VK_IMAGE_LAYOUT_GENERAL, ximgs[X_FOGHIST].img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    }).read(fogR, XFER, XR).read(hdr, XFER, XR).write(fogHistR, XFER, XW);
    // post: auto exposure (histogram -> adapted exposure), bloom mip chain, final tonemap / sharpen / display encode
    g.add("exposure", GFX, [=](VkCommandBuffer cb) {
        vkCmdFillBuffer(cb, exposureBuf.b, 0, 128 * 4, 0);
        memBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                   VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT);
        bind(cb);
        dispatch(cb, P_EXPOSURE, (OW / 4 + 15) / 16, (OH / 4 + 15) / 16, {0, 0, 0, 0});
        computeBarrier(cb);
        dispatch(cb, P_EXPOSURE, 1, 1, {1, 0, 0, 0});
    }).read(hdr, CS, R).write(expoR, CS | XFER, RW | XW);
    g.add("bloom", GFX, [=](VkCommandBuffer cb) {
        bind(cb);
        uint32_t bw = std::max(OW / 2, 1u), bh = std::max(OH / 2, 1u);
        auto mw = [&](uint32_t m) { return std::max(bw >> m, 1u); };
        auto mh = [&](uint32_t m) { return std::max(bh >> m, 1u); };
        for (uint32_t m = 0; m < BLOOM_MIPS; m++) {
            if (m) computeBarrier(cb);
            dispatch(cb, P_BLOOM, (mw(m) + 7) / 8, (mh(m) + 7) / 8, {0, m, 0, 0});
        }
        for (int m = BLOOM_MIPS - 2; m >= 0; m--) {
            computeBarrier(cb);
            dispatch(cb, P_BLOOM, (mw(m) + 7) / 8, (mh(m) + 7) / 8, {1, (uint32_t)m, 0, 0});
        }
    }).read(hdr, CS, R).write(bloomR, CS, RW);
    g.add("post", GFX, [=](VkCommandBuffer cb) { bind(cb); dispatch(cb, P_POST, (OW + 7) / 8, (OH + 7) / 8); })
        .read(hdr, CS, R).read(bloomR, CS, R).read(expoR, CS, R).write(out, CS, WR);

    if (hdrActive) {
        // HDR: UI into its own SDR image, composed over the HDR frame at paper white, then copied to the swapchain
        g.add("present", GFX, [=](VkCommandBuffer cb) {
            VkImage sw = swapImages[si];
            VkRenderingAttachmentInfo att{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
            att.imageView = uiImg.view;
            att.imageLayout = VK_IMAGE_LAYOUT_GENERAL;
            att.loadOp = VK_ATTACHMENT_LOAD_OP_CLEAR;
            att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
            att.clearValue.color = {{0, 0, 0, 0}};
            VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
            ri.renderArea = {{0, 0}, swapExtent};
            ri.layerCount = 1;
            ri.colorAttachmentCount = 1;
            ri.pColorAttachments = &att;
            vkCmdBeginRendering(cb, &ri);
            ImGui_ImplVulkan_RenderDrawData(dd, cb);
            vkCmdEndRendering(cb);
            memBarrier(cb, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT, VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                       VK_ACCESS_SHADER_READ_BIT);
            float sa = (float)OW / OH, da = (float)swapExtent.width / swapExtent.height;
            int dw = swapExtent.width, dh = swapExtent.height;
            if (da > sa) dw = (int)(dh * sa); else dh = (int)(dw / sa);
            int ox = ((int)swapExtent.width - dw) / 2, oy = ((int)swapExtent.height - dh) / 2;
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &descSet, 0, nullptr);
            dispatch(cb, P_PRESENT_HDR, (swapExtent.width + 7) / 8, (swapExtent.height + 7) / 8, {(uint32_t)ox, (uint32_t)oy, (uint32_t)dw, (uint32_t)dh});
            memBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            imgBarrier(cb, sw, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                       VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
            VkImageBlit blit{};
            blit.srcSubresource = blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            blit.srcOffsets[1] = blit.dstOffsets[1] = {(int)swapExtent.width, (int)swapExtent.height, 1};
            vkCmdBlitImage(cb, presentImg.img, VK_IMAGE_LAYOUT_GENERAL, sw, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_NEAREST);
            imgBarrier(cb, sw, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
        }).read(out, CS, R).write(swap, XFER | COLOR | CS, XW | CA | WR);
        return;
    }

    g.add("present", GFX, [=](VkCommandBuffer cb) {
        // blit to swapchain, letterboxed, then ImGui on top
        VkImage sw = swapImages[si];
        imgBarrier(cb, sw, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT, 0,
                   VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        VkClearColorValue black{{0, 0, 0, 1}};
        VkImageSubresourceRange rr{VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        vkCmdClearColorImage(cb, sw, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, &black, 1, &rr);
        imgBarrier(cb, sw, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
        float sa = (float)OW / OH, da = (float)swapExtent.width / swapExtent.height;
        int dw = swapExtent.width, dh = swapExtent.height;
        if (da > sa) dw = (int)(dh * sa); else dh = (int)(dw / sa);
        int ox = ((int)swapExtent.width - dw) / 2, oy = ((int)swapExtent.height - dh) / 2;
        VkImageBlit blit{};
        blit.srcSubresource = blit.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        blit.srcOffsets[1] = {(int)OW, (int)OH, 1};
        blit.dstOffsets[0] = {ox, oy, 0};
        blit.dstOffsets[1] = {ox + dw, oy + dh, 1};
        vkCmdBlitImage(cb, imgs[I_OUT].img, VK_IMAGE_LAYOUT_GENERAL, sw, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &blit, VK_FILTER_LINEAR);
        imgBarrier(cb, sw, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_PIPELINE_STAGE_TRANSFER_BIT,
                   VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                   VK_ACCESS_COLOR_ATTACHMENT_READ_BIT | VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT);
        VkRenderingAttachmentInfo att{VK_STRUCTURE_TYPE_RENDERING_ATTACHMENT_INFO};
        att.imageView = swapViews[si];
        att.imageLayout = VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL;
        att.loadOp = VK_ATTACHMENT_LOAD_OP_LOAD;
        att.storeOp = VK_ATTACHMENT_STORE_OP_STORE;
        VkRenderingInfo ri{VK_STRUCTURE_TYPE_RENDERING_INFO};
        ri.renderArea = {{0, 0}, swapExtent};
        ri.layerCount = 1;
        ri.colorAttachmentCount = 1;
        ri.pColorAttachments = &att;
        vkCmdBeginRendering(cb, &ri);
        ImGui_ImplVulkan_RenderDrawData(dd, cb);
        vkCmdEndRendering(cb);
        if (grabUI) {  // screenshot with the UI: the finished swapchain image, read back after this frame's fence
            imgBarrier(cb, sw, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_READ_BIT);
            VkBufferImageCopy c{};
            c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
            c.imageExtent = {swapExtent.width, swapExtent.height, 1};
            vkCmdCopyImageToBuffer(cb, sw, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, uiShotBuf.b, 1, &c);
            imgBarrier(cb, sw, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_TRANSFER_BIT,
                       VK_ACCESS_TRANSFER_READ_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
        } else
            imgBarrier(cb, sw, VK_IMAGE_LAYOUT_COLOR_ATTACHMENT_OPTIMAL, VK_IMAGE_LAYOUT_PRESENT_SRC_KHR, VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT,
                       VK_ACCESS_COLOR_ATTACHMENT_WRITE_BIT, VK_PIPELINE_STAGE_BOTTOM_OF_PIPE_BIT, 0);
    }).read(out, XFER, XR).write(swap, XFER | COLOR, XW | CA);
}

// ---------------------------------------------------------------- stats / benchmark / scaler
static int raysPerPixel(int n, int b) {
    // [primary] + ReSTIR DI visibility + final light shadow + sun shadow, then per path vertex: bounce + sun + light shadow rays,
    // + [ReSTIR GI reconnection visibility]
    return (S.rasterPrimary ? 0 : 1) + 3 + n * b * 3 + (S.restirGI && S.giSpatial ? 1 : 0);
}

static void readTimestamps() {
    gpuTotal = graph.frameMs();
    if (gpuTotal <= 0) return;
    float gi = graph.passMs("GI");
    gpuEmaTotal = gpuEmaTotal == 0 ? gpuTotal : gpuEmaTotal * 0.92f + gpuTotal * 0.08f;
    gpuEmaGI = gpuEmaGI == 0 ? gi : gpuEmaGI * 0.92f + gi * 0.08f;
    if (submittedTier > 0 && submittedFrame >= Bench::WARM) bench.samples[submittedTier].push_back(gpuTotal);
}

static void startBenchmark() {
    bench.running = true;
    bench.done = false;
    bench.tier = 1;
    bench.frame = 0;
    bench.vt = 0;
    for (auto& s : bench.samples) s.clear();
    bench.saved = S;
    bench.savedCam = cam;
    resetHistory = true;
}

static void benchStep() {
    if (!bench.running) { submittedTier = 0; return; }
    if (bench.frame >= Bench::WARM + Bench::MEAS) {
        auto& v = bench.samples[bench.tier];
        if (!v.empty()) {
            float sum = 0;
            for (float x : v) sum += x;
            bench.avg[bench.tier] = sum / v.size();
            std::sort(v.begin(), v.end());
            bench.p95[bench.tier] = v[std::min(v.size() - 1, (size_t)(v.size() * 0.95f))];
        }
        bench.tier++;
        bench.frame = 0;
        resetHistory = true;
        if (bench.tier > 4) {
            bench.running = false;
            bench.done = true;
            S = bench.saved;
            cam = prevCam = bench.savedCam;
            submittedTier = 0;
            return;
        }
    }
    // fixed, repeatable workload: default quality settings, scripted orbit, simulated 60 Hz clock
    Settings keep = S;
    S = Settings{};
    S.exposure = keep.exposure;
    S.rasterPrimary = keep.rasterPrimary;
    S.restirGI = keep.restirGI;
    S.raysFirst = bench.tier;
    double a = bench.vt * 0.25;
    cam.pos = {(float)sin(a) * 16.f, 2.0f + (float)sin(a * 0.7) * 0.8f, (float)cos(a) * 16.f};
    V3 d = normalize(V3{0, 0.5f, 0} - cam.pos);
    cam.yaw = atan2f(d.x, d.z);
    cam.pitch = asinf(d.y);
    simTime = bench.vt;
    bench.vt += 1.0 / 60.0;
    submittedTier = bench.tier;
    submittedFrame = bench.frame++;
}

// best tier whose p95 GPU time fits the frame budget
static int benchBestTier(int fps) {
    float budget = 1000.f / fps;
    int best = 0;
    for (int t = 1; t <= 4; t++)
        if (bench.p95[t] > 0 && bench.p95[t] <= budget) best = t;
    return best;
}

static void autoScale() {
    if (targetFps <= 0 || bench.running || gpuEmaTotal <= 0) return;
    float budget = 1000.f / targetFps * 0.92f;
    float perRay = gpuEmaGI / std::max(1, S.raysFirst);
    float fixed = gpuEmaTotal - gpuEmaGI;
    int fit = perRay > 0 ? std::clamp((int)floorf((budget - fixed) / perRay), 1, 4) : S.raysFirst;
    if (fit < S.raysFirst) { if (++scaleDown > 10) { S.raysFirst = fit; scaleDown = scaleUp = 0; } } else scaleDown = 0;
    if (fit > S.raysFirst) { if (++scaleUp > 90) { S.raysFirst++; scaleDown = scaleUp = 0; } } else scaleUp = 0;
}

// ---------------------------------------------------------------- input / UI
static uint32_t debrisSeed = 1;
static float rnd01() {
    debrisSeed = debrisSeed * 747796405u + 2891336453u;
    return (float)((debrisSeed >> 8) & 0xFFFFFF) / 16777215.f;
}
static V3 rndDir() {
    V3 d{rnd01() * 2.f - 1.f, rnd01() * 2.f - 1.f, rnd01() * 2.f - 1.f};
    return dot(d, d) > 1e-4f ? normalize(d) : V3{0, 1, 0};
}
static void addDebris(uint8_t kind, V3 p, V3 v, float size, float delay = 0.f) {
    if (spray.size() >= (size_t)MAX_SPRAY) return;
    Droplet d;
    d.p = p; d.v = v; d.size = size; d.life = 0.f; d.delay = delay; d.kind = kind; d.spin = rndDir();
    spray.push_back(d);
}

// Something of size sc (m) hit the water at velocity v: a splash event for the surface (water.glsl) plus spray,
// a crown sheet breaking into droplets that fly up and out, then a delayed Worthington jet from the cavity.
// Impact weighs the vertical speed most (a flat skip throws a small splash, a plunge a big one).
static void waterImpact(V3 at, float sc, const float v[3], float scale = 1.f) {
    float vh = sqrtf(v[0] * v[0] + v[2] * v[2]), vy = fabsf(v[1]);
    float e = std::min(sqrtf(sc) * (0.3f + (0.8f * vy + 0.25f * vh) / 8.f), 3.f) * scale;
    // full: replace the event with the least left of it (impact x foam remaining), never a big fresh splash for
    // the plop of a clod -- FIFO eviction let a burst of debris re-entries wipe the rock's own splash at once
    if (splashEvents.size() >= (size_t)MAX_SPLASH_EV) {
        size_t w = 0;
        float wv = 1e30f;
        for (size_t i = 0; i < splashEvents.size(); i++) {
            float left = splashEvents[i].impact * expf(-(float)(simTime - splashEvents[i].t) / SPLASH_FOAM_TAU);
            if (left < wv) { wv = left; w = i; }
        }
        if (wv > e) return;
        splashEvents.erase(splashEvents.begin() + w);
    }
    splashEvents.push_back({at.x, at.z, (float)simTime, e, sc, sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])});
    int n = std::min((int)(60.f * e * std::max(sc, 0.5f)), 500);
    V3 carry = V3{v[0], 0, v[2]} * 0.25f;  // a skipping stone throws its spray forward
    for (int i = 0; i < n; i++) {
        bool jet = i < n / 5 && scale >= 1.f;
        float a = rnd01() * 6.2831853f, se = sqrtf(e);
        float out = jet ? rnd01() * 0.4f : (1.2f + 2.5f * rnd01()) * se, up = jet ? (3.f + 3.f * rnd01()) * se : (1.5f + 3.5f * rnd01()) * se;
        float r0 = jet ? 0.1f * rnd01() * sc : (0.5f + 0.4f * rnd01()) * sc;
        float size = (0.015f + 0.05f * rnd01() * rnd01()) * (0.6f + 0.9f * std::min(sc, 2.f)) * (jet ? 1.6f : 1.f);
        addDebris(DEB_WATER, at + V3{cosf(a) * r0, 0.02f, sinf(a) * r0}, V3{cosf(a) * out, up, sinf(a) * out} + (jet ? V3{} : carry), size,
                  jet ? 0.18f + 0.1f * rnd01() : 0.03f * rnd01());
    }
}

// Ground impact releasing energy E (J): crater volume ~ E / soil strength, a bowl a third as deep as it is wide
// (simple-crater proportions) with a rim of thrown-out soil, and ejecta clods flung out at ~45 degrees.
// Trees close to big impacts get blown over by the blast.
static void knockTree(const TreeInst& t, V3 impulse, V3 at, float severity);
static bool strikeTree(const TreeInst& t, V3 at, V3 vel, float mass, V3* took = nullptr);
static void addDecal(V3 p, V3 n, float r, int kind) {
    if (!S.decals) return;
    n = normalize(n);
    Decal d{{p.x, p.y, p.z}, r, {n.x, n.y, n.z}, kind, simTime};
    if (decalList.size() >= MAX_DECALS) decalList.erase(decalList.begin());
    decalList.push_back(d);
}
static void addDust(V3 p, float r, float density, float rise = 0.4f) {
    if (dustList.size() >= MAX_DUST) dustList.erase(dustList.begin());
    dustList.push_back({{p.x, p.y, p.z}, r, density, rise, simTime});
}
// Craters next to the river fill from it instead of appearing full: the water level rises as river water pours in
// (a share of the river's discharge, falling off with the remaining head), and while it does, that share is
// missing downstream, so the river there slows and drops a little (water.glsl). A fresh crater, or one that grows,
// starts (part) empty again.
// Ground under a terrain contact: friction / restitution by what's there. Bare soil and grass give little bounce
// (soft, absorbs), hard surfaces (stone, paving, painted rock / concrete textures) more; loose crater soil less;
// water films make it slicker, snow swallows the bounce and is slippery.
static void surfaceOfMaterial(const std::string& n, float& f, float& e) {
    auto has = [&](const char* k) { return n.find(k) != std::string::npos; };
    if (has("ice")) { f = 0.05f; e = 0.1f; }
    else if (has("metal") || has("steel")) { f = 0.45f; e = 0.4f; }
    else if (has("stone") || has("rock") || has("concrete") || has("asphalt") || has("paving") || has("brick") || has("tile") || has("cobble")) { f = 0.62f; e = 0.35f; }
    else if (has("wood") || has("plank")) { f = 0.55f; e = 0.25f; }
    else if (has("sand") || has("gravel")) { f = 0.9f; e = 0.03f; }
    else if (has("mud")) { f = 0.95f; e = 0.0f; }
    else { f = 0.75f; e = 0.12f; }
}
static void groundSurfaceAt(const float p[3], float& f, float& e) {
    float x = p[0], z = p[2];
    f = S.groundFriction; e = S.groundBounce;  // natural soil / grass
    const auto& zs = stream::zones();
    float zc = zoneCover(x, z, zs.data(), (int)zs.size());
    if (zc > 0.5f) {  // flattened (city) ground: paved, or its own texture
        float zf = 0.62f, ze = 0.3f;
        for (auto& zn : zs) if (zoneWeight(zn, x, z) > 0.5f && !zn.mat.empty()) surfaceOfMaterial(zn.mat, zf, ze);
        f = zf; e = ze;
    }
    // painted ground texture (world map)
    if (!std::isfinite(x) || !std::isfinite(z)) return;
    int cx = (int)floorf((x - worldmap::ORIGIN) / worldmap::CELL), cz = (int)floorf((z - worldmap::ORIGIN) / worldmap::CELL);
    if (cx >= 0 && cz >= 0 && cx < worldmap::N && cz < worldmap::N) {
        const worldmap::Paint& pt = worldmap::paintData()[(size_t)cz * worldmap::N + cx];
        if (pt.mat && pt.matW > 64 && !worldmap::paletteName(pt.mat).empty()) {
            float pf, pe;
            surfaceOfMaterial(worldmap::paletteName(pt.mat), pf, pe);
            float w = pt.matW / 255.f;
            f += (pf - f) * w; e += (pe - e) * w;
        }
    }
    if (inCrater(x, z, 0.f)) { f = std::max(f, 0.85f); e = std::min(e, 0.04f); }  // loose, churned soil
    if (simSnapValid) {
        int i = cx - simSnapX, k = cz - simSnapZ;
        if (i >= 0 && k >= 0 && i < SIM_RB && k < SIM_RB) {
            float snow = simSnap[k * SIM_RB + i][0], wet = simSnap[k * SIM_RB + i][1];
            f *= 1.f - 0.35f * std::clamp(wet, 0.f, 1.f);
            e *= 1.f - 0.3f * std::clamp(wet, 0.f, 1.f);
            float sk = std::clamp((snow - 0.02f) / 0.2f, 0.f, 1.f);
            f += (0.28f - f) * sk;
            e *= 1.f - sk;
        }
    }
}

static void updateRiverSwell(float dt) {
    float target = S.riverRainRise * S.rain;
    float tau = target > riverSwell ? 60.f : 180.f;
    riverSwell += (target - riverSwell) * (1.f - expf(-dt * S.simSpeed / tau));
}

// CPU mirror of material.glsl waterLevelAt: inside a crater only its pond counts; -1e9 = dry
static float waterLevelAt(float x, float z) {
    int n = std::min(gCraterCount.load(std::memory_order_relaxed), MAX_CRATERS);
    for (int i = 0; i < n; i++) {
        float dx = x - gCraters[i].x, dz = z - gCraters[i].z, r = gCraters[i].r * 1.2f;
        if (dx * dx + dz * dz < r * r) return craterWater[i].pond ? craterWater[i].level : -1e9f;
    }
    if (fabsf(z - riverCenter(x)) > 3.f * gRiver.width + 1.f) return -1e9f;
    RiverInfo ri = riverInfo(x, z);
    return fabsf(ri.d) < ri.w + 1.f + 2.f * riverSwell ? ri.level + riverSwell : -1e9f;
}

static void updateCraterWater(float dt) {
    int n = std::min(gCraterCount.load(std::memory_order_relaxed), MAX_CRATERS);
    float U = 1.3f * S.flow, Qr = std::max(U * gRiver.width * gRiver.depth, 0.1f);  // river discharge, m^3/s
    for (int i = 0; i < n; i++) {
        const Crater& c = gCraters[i];
        CraterWater& w = craterWater[i];
        if (w.lastR != c.r) {  // new or grown: the old water spreads over the bigger bowl
            w.fill = w.lastR > 0.f ? w.fill * powf(w.lastR / c.r, 3.f) : 0.f;
            w.lastR = c.r;
            // natural ground around it (craters removed): a plane the shaders build the local surface on
            auto nat = [](float x, float z) { return heightFn(x, z) - craterCarve(x, z, 0.f); };
            float e = std::max(1.f, c.r), h0 = nat(c.x, c.z);
            w.ground[0] = h0;
            w.ground[1] = (nat(c.x + e, c.z) - nat(c.x - e, c.z)) / (2.f * e);
            w.ground[2] = (nat(c.x, c.z + e) - nat(c.x, c.z - e)) / (2.f * e);
        }
        RiverInfo ri = riverInfo(c.x, c.z);
        float bottom = heightFn(c.x, c.z);
        bool touches = fabsf(ri.d) - ri.w < c.r * 1.1f;  // the bowl breaks into the channel
        w.pond = touches && bottom < ri.level - 0.02f;
        if (!w.pond) { w.divert = w.inflowSpeed = 0.f; w.fill = 1.f; continue; }
        float depth = ri.level - bottom, V = 0.5f * 3.14159f * c.r * c.r * depth;
        float head = std::max(1.f - w.fill, 0.f);
        float Q = std::min(S.craterInflow * Qr * sqrtf(head), Qr) * (1.f + riverSwell / std::max(gRiver.depth, 0.3f));
        w.fill = std::min(w.fill + Q * dt / std::max(V, 0.01f), 1.f);
        w.level = bottom + (ri.level + riverSwell - bottom) * w.fill;
        w.divert = w.fill < 1.f ? Q / Qr : 0.f;
        w.inflowSpeed = w.fill < 1.f ? std::min(sqrtf(2.f * 9.81f * depth * head) * 0.6f, 4.f) : 0.f;
    }
}

static void groundImpact(V3 at, float E) {
    const float soilStrength = 1.2e5f;  // J per m^3 of loosened soil
    float V = E / soilStrength, R = std::min(cbrtf(V / 0.55f), 25.f);
    if (R < 0.3f) return;
    int n = std::min(gCraterCount.load(std::memory_order_relaxed), MAX_CRATERS);
    // landing inside an existing crater deepens and widens it (volumes add) rather than stacking a new bowl
    int into = -1;
    for (int i = 0; i < n; i++) {
        float dx = gCraters[i].x - at.x, dz = gCraters[i].z - at.z;
        if (dx * dx + dz * dz < 0.36f * gCraters[i].r * gCraters[i].r && (into < 0 || gCraters[i].r > gCraters[into].r)) into = i;
    }
    if (into >= 0) {
        Crater& o = gCraters[into];
        float Vo = 0.55f * o.r * o.r * o.r, Vt = Vo + V, w = V / Vt;
        float Rn = std::min(cbrtf(Vt / 0.55f), 25.f);
        o = Crater{o.x + (at.x - o.x) * w, o.z + (at.z - o.z) * w, Rn, 0.33f * Rn};
    } else {
        Crater c{at.x, at.z, R, 0.33f * R};
        if (n < MAX_CRATERS) {
            gCraters[n] = c;
            gCraterCount.store(n + 1, std::memory_order_release);
        } else {  // full: the smallest crater goes (the least visible loss)
            int k = 0;
            for (int i = 1; i < n; i++) if (gCraters[i].r < gCraters[k].r) k = i;
            if (gCraters[k].r < R) gCraters[k] = c;
        }
    }
    terrainDirty = placeDirty = true;
    stream::terrainChanged();
    addDecal(at, {0, 1, 0}, R * 2.2f, 1);                  // thrown-out soil around the crater
    addDust(at + V3{0, 0.3f * R, 0}, R * 1.1f, 0.35f * std::min(R, 3.f), 0.5f);
    int clods = std::min((int)(25.f * R * R), 300);
    for (int i = 0; i < clods; i++) {
        float a = rnd01() * 6.2831853f, r0 = R * sqrtf(rnd01()) * 0.8f, sp = (2.f + 5.f * rnd01()) * sqrtf(R);
        float elev = 0.6f + 0.6f * rnd01();
        addDebris(DEB_DIRT, at + V3{cosf(a) * r0, 0.1f, sinf(a) * r0}, V3{cosf(a) * cosf(elev), sinf(elev), sinf(a) * cosf(elev)} * sp,
                  (0.015f + 0.07f * rnd01() * rnd01() * rnd01()) * std::min(sqrtf(R), 2.f), 0.05f * rnd01());
    }
    // blast: trees within ~2 crater radii go over if the shock carries enough energy to them
    if (E > 2e5f) {
        int knocked = 0;
        std::vector<TreeInst> hit;
        for (const TreeInst& t : treeList) {
            float dx = t.p.x - at.x, dz = t.p.z - at.z, d2 = dx * dx + dz * dz;
            if (d2 < 4.f * R * R && knocked < 24) { hit.push_back(t); knocked++; }
        }
        for (const TreeInst& t : hit) {
            V3 d = t.p - at;
            d.y = 0;
            float dist = std::max(sqrtf(dot(d, d)), 0.5f);
            V3 dir = d * (1.f / dist);
            float J = std::min(sqrtf(E) * 8.f / dist, 4e4f);
            // the blast snaps trunks low down; closer in it shreds them
            V3 at2 = t.p + V3{0, (0.5f + 0.9f * rnd01()) * t.scale, 0}, Jv = dir * J + V3{0, J * 0.15f, 0};
            if (t.v < (uint32_t)trees::VARIANTS) strikeTree(t, at2, Jv * (1.f / 60.f), 60.f);  // the shock front as a 60 kg body
            else knockTree(t, Jv, at2, 1.f + J / 6e3f);
        }
    }
}

static void dropThrown(size_t i);
static void dropStump(size_t i);

// Break a placed tree where it was hit (at): trees::breakApart cuts it into a stump, which stays as a static body,
// and logs / branches / crowns, which become wood pieces on the thrown list (they float, roll, knock other trees
// and break walls). severity >= 1 grows with the blow: more trunk breaks, more limbs and crowns torn loose, more
// splinters and leaves. The impulse goes into the piece nearest the hit, the rest get the tree's share of it.
static void knockTree(const TreeInst& t, V3 impulse, V3 at, float severity) {
    if (brokenTrees.count(treeKey(t.ix, t.iz))) return;
    for (auto& q : pendingTreeBreaks) if (q.t.ix == t.ix && q.t.iz == t.iz) return;
    pendingTreeBreaks.push_back({t, impulse, at, severity});
}
// Imported tree (variant v >= trees::VARIANTS) broken at local height cutY: triangles are sorted by centroid into
// the stump (below the cut) and the trunk + crown above, which splits once more at mid height on a hard blow.
// Mass ~ a real hardwood of that height (~1 t for 10 m), shared by height.
static void assetBreak(int v, float cutY, float severity, std::vector<trees::Piece>& out) {
    const MeshData& m = treeModel.data.meshes[v];
    float y0 = m.bmin[1], H = std::max(m.bmax[1] - m.bmin[1], 0.5f);
    cutY = std::clamp(cutY, y0 + 0.25f, y0 + 0.5f * H);
    float cut2 = severity > 2.f ? cutY + 0.5f * (m.bmax[1] - cutY) : 1e30f;
    float total = 1000.f * powf(H / 10.f, 2.5f);
    auto piece = [&](float lo, float hi, int kind) {
        trees::Piece pc;
        pc.kind = kind;
        std::vector<uint32_t> remap(m.verts.size() / 8, UINT32_MAX);
        for (size_t t = 0; t * 3 + 2 < m.idx.size(); t++) {
            const uint32_t* tri = &m.idx[t * 3];
            float cy = (m.verts[tri[0] * 8 + 1] + m.verts[tri[1] * 8 + 1] + m.verts[tri[2] * 8 + 1]) / 3.f;
            if (cy < lo || cy >= hi) continue;
            for (int k = 0; k < 3; k++) {
                uint32_t& r = remap[tri[k]];
                if (r == UINT32_MAX) {
                    r = (uint32_t)pc.mesh.verts.size() / 8;
                    pc.mesh.verts.insert(pc.mesh.verts.end(), &m.verts[tri[k] * 8], &m.verts[tri[k] * 8 + 8]);
                }
                pc.mesh.idx.push_back(r);
            }
            pc.mesh.triMat.push_back(t < m.triMat.size() ? m.triMat[t] : 0);
        }
        pc.mass = total * (std::min(hi, m.bmax[1]) - std::max(lo, y0)) / H;
        if (!pc.mesh.idx.empty()) out.push_back(std::move(pc));
    };
    piece(-1e30f, cutY, trees::PIECE_STUMP);
    piece(cutY, cut2, trees::PIECE_LOG);
    if (cut2 < 1e30f) piece(cut2, 1e30f, trees::PIECE_LEAVES);
}

// Only the materials the pieces use, and their textures (an imported tree's are large), remapped into md.
static void takeTreeMaterials(Model& md, std::vector<trees::Piece>& pcs) {
    std::map<uint32_t, uint32_t> mats;
    std::map<int, int> texs;
    for (auto& pc : pcs)
        for (uint32_t& tm : pc.mesh.triMat) {
            auto [it, fresh] = mats.try_emplace(tm, (uint32_t)md.data.materials.size());
            if (fresh) {
                MaterialData mt = treeModel.data.materials[tm];
                for (int* tx : {&mt.tex, &mt.normalTex, &mt.ormTex, &mt.occTex, &mt.emisTex}) {
                    if (*tx < 0) continue;
                    auto [jt, nt] = texs.try_emplace(*tx, (int)md.data.textures.size());
                    if (nt) md.data.textures.push_back(treeModel.data.textures[*tx]);
                    *tx = jt->second;
                }
                md.data.materials.push_back(mt);
            }
            tm = it->second;
        }
}

static void breakTreeNow(const TreeInst& t, V3 impulse, V3 at, float severity) {
    addDecal(at, impulse * -1.f, 0.3f, 2);  // splintered wood where it was hit
    addDust(at, 0.8f, 0.25f, 0.2f);         // a snapping trunk throws bark dust
    uint64_t key = treeKey(t.ix, t.iz);
    if (!brokenTrees.insert(key).second) return;
    placeDirty = true;
    auto it = treeBodies.find(key);
    if (it != treeBodies.end()) { physics::removeBodies(&it->second, 1); treeBodies.erase(it); }
    std::vector<trees::Piece> pcs;
    std::vector<float> shed;  // leaves shaken loose (tree-local positions)
    assetBreak((int)t.v, (at.y - t.p.y) / t.scale, severity, pcs);

    auto fs = std::make_unique<FragSet>();
    Model& md = fs->model;
    md.path = "<tree pieces>";
    md.state = 1;
    takeTreeMaterials(md, pcs);
    float c = cosf(t.yaw), sn = sinf(t.yaw), s3 = t.scale * t.scale * t.scale;
    struct Spawn { size_t mesh; Xf xf; float mass, r; int kind; std::vector<float> col; std::vector<uint32_t> colIdx; };
    std::vector<Spawn> spawns;
    float total = 0.f;
    for (auto& pc : pcs) {
        MeshData& m = pc.mesh;
        if (m.idx.empty()) continue;
        // into metres, centred on its bounding box
        V3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
        for (size_t i = 0; i < m.verts.size(); i += 8) {
            for (int k = 0; k < 3; k++) m.verts[i + k] *= t.scale;
            V3 q{m.verts[i], m.verts[i + 1], m.verts[i + 2]};
            lo = {std::min(lo.x, q.x), std::min(lo.y, q.y), std::min(lo.z, q.z)};
            hi = {std::max(hi.x, q.x), std::max(hi.y, q.y), std::max(hi.z, q.z)};
        }
        V3 ctr = (lo + hi) * 0.5f;
        float r = 0.f;
        for (size_t i = 0; i < m.verts.size(); i += 8) {
            m.verts[i] -= ctr.x; m.verts[i + 1] -= ctr.y; m.verts[i + 2] -= ctr.z;
            r = std::max(r, m.verts[i] * m.verts[i] + m.verts[i + 1] * m.verts[i + 1] + m.verts[i + 2] * m.verts[i + 2]);
        }
        for (int k = 0; k < 3; k++) { m.bmin[k] = (&lo.x)[k] - (&ctr.x)[k]; m.bmax[k] = (&hi.x)[k] - (&ctr.x)[k]; }
        // same yaw as the placed instance ({c, 0, sn}, {0, 1, 0}, {-sn, 0, c})
        V3 cw = t.p + V3{c * ctr.x + sn * ctr.z, ctr.y, -sn * ctr.x + c * ctr.z};
        Xf px = {{{c, 0, sn, cw.x}, {0, 1, 0, cw.y}, {-sn, 0, c, cw.z}}};
        md.data.meshes.push_back(std::move(m));
        for (size_t i = 0; i < pc.colVerts.size(); i += 3)  // the standing tree's own collider, same frame as its mesh
            for (int k = 0; k < 3; k++) pc.colVerts[i + k] = pc.colVerts[i + k] * t.scale - (&ctr.x)[k];
        spawns.push_back({md.data.meshes.size() - 1, px, pc.mass * s3, sqrtf(r), pc.kind, std::move(pc.colVerts), std::move(pc.colIdx)});
        if (pc.kind != trees::PIECE_STUMP) total += pc.mass * s3;
    }
    if (spawns.empty()) return;
    uploadModel(md);
    FragSet* set = fs.get();
    fragSets.push_back(std::move(fs));
    V3 vBase = impulse * (0.5f / std::max(total, 1.f));
    float vb = sqrtf(dot(vBase, vBase));
    if (vb > 15.f) vBase = vBase * (15.f / vb);
    uint32_t hitBody = UINT32_MAX;
    float hitD = 1e30f;
    for (auto& sp : spawns) {
        const MeshData& mesh = set->model.data.meshes[sp.mesh];
        uint32_t nv = (uint32_t)mesh.verts.size() / 8;
        // hulls from at most ~4k points (an imported tree piece can have millions)
        std::vector<float> sub;
        const float* hp = mesh.verts.data();
        uint32_t hs = 8, hn = nv;
        if (nv > 4096) {
            for (uint32_t i = 0; i < nv; i += nv / 4096) sub.insert(sub.end(), &mesh.verts[(size_t)i * 8], &mesh.verts[(size_t)i * 8 + 3]);
            hp = sub.data(); hs = 3; hn = (uint32_t)sub.size() / 3;
        }
        void* shape = !sp.colIdx.empty() ? physics::meshShape(sp.col.data(), 3, (uint32_t)sp.col.size() / 3, sp.colIdx.data(), (uint32_t)sp.colIdx.size() / 3)
                                         : physics::convexShape(hp, hs, hn);
        if (!shape) continue;
        if (sp.kind == trees::PIECE_STUMP) {
            if (stumps.size() >= (size_t)MAX_FALLEN) dropStump(0);
            Stump st{set, (uint32_t)sp.mesh, 0, shape, sp.xf};
            physics::addStatic(&st.shape, &st.xf, 1, &st.body);
            set->live++;
            stumps.push_back(st);
            continue;
        }
        // the hull is mostly air around a crown: pick the density that gives the piece its real mass
        fracture::Poly hull = fracture::fromPoints(hp, hs, hn, 26);
        float dens = std::clamp(sp.mass / std::max(fracture::volume(hull), 1e-3f), 5.f, 900.f);
        V3 v = vBase + rndDir() * (0.8f * (std::min(severity, 6.f) - 1.f) * rnd01());
        V3 w = rndDir() * (0.4f * std::min(severity, 5.f) * rnd01());
        if (thrown.size() >= (size_t)MAX_THROWN) dropThrown(0);
        Thrown f;
        f.sc = 1.f;
        f.radius = sp.r;
        f.frag = set;
        f.fragMesh = (uint32_t)sp.mesh;
        f.shape = shape;
        f.body = physics::addDynamicPose(shape, sp.xf, &v.x, &w.x, dens);
        f.xf = sp.xf;
        f.cooldown = 0.4f;  // pieces settling off the stump aren't impacts
        f.density = dens;
        f.wood = true;
        memcpy(f.v, &v.x, 12);
        set->live++;
        V3 d = V3{sp.xf.m[0][3], sp.xf.m[1][3], sp.xf.m[2][3]} - at;
        if (dot(d, d) < hitD) { hitD = dot(d, d); hitBody = f.body; }
        thrown.push_back(std::move(f));
    }
    if (hitBody != UINT32_MAX) {
        V3 J = impulse * 0.5f;
        physics::addImpulse(hitBody, &J.x, &at.x);
    }
    if (set->live == 0) {
        auto keep = std::shared_ptr<FragSet>(fragSets.back().release());
        fragSets.pop_back();
        retireNext.push_back([keep]() { if (keep->model.gpu) releaseModel(keep->model); });
    }
    float sev = std::min(severity, 4.f);
    V3 cut{t.p.x, at.y, t.p.z};
    for (int i = 0; i < (int)(20.f * sev); i++)  // splinters from the snap
        addDebris(DEB_WOOD, cut + rndDir() * (0.3f * t.scale), rndDir() * (2.f + 4.f * rnd01()) + V3{0, 2.f, 0}, 0.04f + 0.08f * rnd01());
    V3 dir = normalize(impulse + V3{0, 1e-3f, 0});
    // leaves shaken out of the crown, from where they actually hung (at most ~600, picked evenly)
    size_t nShed = shed.size() / 3, stepL = std::max<size_t>(1, nShed / 600);
    for (size_t i = 0; i < nShed; i += stepL) {
        V3 l{shed[i * 3] * t.scale, shed[i * 3 + 1] * t.scale, shed[i * 3 + 2] * t.scale};
        V3 q = t.p + V3{c * l.x + sn * l.z, l.y, -sn * l.x + c * l.z};
        addDebris(DEB_LEAF, q, rndDir() * (0.5f + 1.5f * rnd01()) + dir * (1.5f * rnd01() * sev), 0.035f + 0.03f * rnd01(), 0.6f * rnd01());
    }
    if (t.v >= (uint32_t)trees::VARIANTS)
        for (int i = 0; i < (int)(60.f * sev); i++) {  // imported trees: no leaf positions, shake from the crown's bounds
            V3 q = t.p + V3{0, (4.f + 4.f * rnd01()) * t.scale, 0} + rndDir() * (2.5f * t.scale * sqrtf(rnd01()));
            addDebris(DEB_LEAF, q, rndDir() * (1.f + 2.f * rnd01()) + dir * (2.f * rnd01() * sev), 0.035f + 0.03f * rnd01(), 0.15f * rnd01());
        }
}

// The current slapping the upstream face of boulders that break the surface: now and then a small burst of
// droplets thrown up and back (only near the camera, where you can see them).
static void boulderSpray(float dt) {
    if (S.splash <= 0.f || S.flow <= 0.f) return;
    for (int i = (int)floorf((cam.pos.x - 50.f) / RIVER_ROCK_L); i <= (int)floorf((cam.pos.x + 50.f) / RIVER_ROCK_L); i++) {
        float x, y, z, sc;
        if (!riverRock(i, x, y, z, sc)) continue;
        RiverInfo ri = riverInfo(x, z);
        float top = y + 0.72f * sc;
        if (top < ri.level + 0.05f || (x - cam.pos.x) * (x - cam.pos.x) + (z - cam.pos.z) * (z - cam.pos.z) > 2500.f) continue;
        float cur[2];
        riverVelocity(x - 0.9f * sc, z, S.flow, cur);
        float U = sqrtf(cur[0] * cur[0] + cur[1] * cur[1]);
        if (rnd01() > dt * 1.5f * S.splash * U) continue;  // ~1.5 bursts/s per m/s of current
        V3 dir = V3{cur[0], 0, cur[1]} * (1.f / std::max(U, 1e-3f));
        float hh = (ri.level - y) / (0.72f * sc), hR = 0.95f * sc * sqrtf(std::clamp(1.f - hh * hh, 0.05f, 1.f));
        V3 face = V3{x, ri.level + 0.03f, z} - dir * hR;
        int n = 6 + (int)(rnd01() * 10.f * std::min(U, 2.f));
        for (int k = 0; k < n; k++) {
            V3 side = V3{-dir.z, 0, dir.x} * ((rnd01() - 0.5f) * 1.6f * hR);
            addDebris(DEB_WATER, face + side,
                      dir * (-0.6f * U * rnd01()) + side * (1.5f / std::max(hR, 0.3f)) + V3{0, (1.f + 1.5f * rnd01()) * std::min(U, 2.f), 0},
                      0.012f + 0.03f * rnd01() * rnd01());
        }
    }
}

// Ballistic debris with a little air drag. Water is gone when it falls back into the river or lands; clods and
// splinters bounce once off the ground, then settle and vanish.
static void updateSpray(float dt) {
    boulderSpray(dt);
    for (size_t i = 0; i < spray.size();) {
        Droplet& d = spray[i];
        if (d.delay > 0.f) { d.delay -= dt; i++; continue; }
        d.life += dt;
        d.v.y -= 9.8f * dt;
        d.v = d.v * (1.f - std::min((d.kind == DEB_WATER ? 0.6f : d.kind == DEB_LEAF ? 3.f : 0.15f) * dt, 0.5f));
        d.p = d.p + d.v * dt;
        float ground = heightFn(d.p.x, d.p.z);
        RiverInfo ri = riverInfo(d.p.x, d.p.z);
        float wl = waterLevelAt(d.p.x, d.p.z);
        bool inWater = d.v.y < 0.f && d.p.y < wl;
        bool dead = d.life > (d.kind == DEB_WATER ? 4.f : d.kind == DEB_LEAF ? 10.f : 6.f) || ((d.kind == DEB_WATER || d.kind == DEB_LEAF) && inWater);
        if (!dead && d.p.y < ground) {
            if (d.kind == DEB_WATER || dot(d.v, d.v) < 1.f) dead = true;
            else { d.p.y = ground; d.v = V3{d.v.x * 0.4f, -d.v.y * 0.25f, d.v.z * 0.4f}; }
        }
        if (!dead && d.kind != DEB_WATER && inWater && d.size > 0.05f && d.v.y < -2.f) {  // clods plopping in
            float v[3] = {d.v.x, d.v.y, d.v.z};
            waterImpact({d.p.x, wl, d.p.z}, d.size * 2.f, v, 0.5f);
            dead = true;
        }
        if (dead) { spray[i] = spray.back(); spray.pop_back(); continue; }
        i++;
    }
    while (!splashEvents.empty() && simTime - splashEvents.front().t > SPLASH_LIFE) splashEvents.erase(splashEvents.begin());
}

// One user of a fragment model gone; the model is freed with the last (after the GPU is done with it).
static void releaseFrag(FragSet* fs) {
    if (--fs->live > 0) return;
    auto it = std::find_if(fragSets.begin(), fragSets.end(), [&](auto& p) { return p.get() == fs; });
    if (it == fragSets.end()) return;
    std::shared_ptr<FragSet> keep(it->release());
    fragSets.erase(it);
    // instances went away this frame; the last frame using them finishes before retireNext runs
    retireNext.push_back([keep]() { if (keep->model.gpu) releaseModel(keep->model); });
}

// Remove thrown[i]: body, own hull, and its fragment model once no piece uses it (freed after the GPU is done).
static void dropThrown(size_t i) {
    Thrown& t = thrown[i];
    physics::removeBodies(&t.body, 1);
    if (t.shape) physics::releaseShape(t.shape);
    if (t.frag) releaseFrag(t.frag);
    thrown.erase(thrown.begin() + i);
}

static void dropStump(size_t i) {
    Stump& s = stumps[i];
    if (s.key) damagedTrees.erase(s.key);
    physics::removeBodies(&s.body, 1);
    physics::releaseShape(s.shape);
    releaseFrag(s.frag);
    stumps.erase(stumps.begin() + i);
}

// Rock solid (body space at scale 1) and its surface textures for fragments.
static fracture::Poly rockPoly[16];
static void ensureRockShape();
static TextureData rockTexture() {
    TextureData t;
    t.w = t.h = 64;
    t.rgba.resize(64 * 64 * 4);
    for (uint32_t y = 0; y < 64; y++)
        for (uint32_t x = 0; x < 64; x++) {
            float k = vnoise(x * 0.25f, y * 0.25f) * 0.6f + vnoise(x * 0.9f + 7.f, y * 0.9f) * 0.4f;
            uint8_t v = (uint8_t)(150.f + 90.f * k), *p = &t.rgba[(y * 64 + x) * 4];
            p[0] = v; p[1] = (uint8_t)(v * 0.97f); p[2] = (uint8_t)(v * 0.92f); p[3] = 255;
        }
    return t;
}

// Throw a rock from the camera (G / left click / right trigger), size from the UI slider.
static void moveLogEvent(const char* kind, const std::string& arg);
static void throwRockImpl();
static void throwRock() {
    moveLogEvent("throw", std::to_string(S.throwSize) + " " + std::to_string(S.throwSpeed));
    throwRockImpl();
}
static void throwRockImpl() {
    ensureRockShape();
    uint32_t var = (uint32_t)(hash2((int)frameIndex, 7) * ROCK_VARIANTS) % ROCK_VARIANTS;
    if (!rockHull[var]) return;
    if (thrown.size() >= (size_t)MAX_THROWN) dropThrown(0);  // recycle the oldest
    float sc = S.throwSize;
    V3 f = cam.fwd(), p = cam.pos + f * (0.6f + sc * 1.2f) - V3{0, 0.2f, 0};
    V3 v = f * S.throwSpeed;
    float w[3] = {hash2((int)frameIndex, 1) * 8.f - 4.f, hash2((int)frameIndex, 2) * 8.f - 4.f, hash2((int)frameIndex, 3) * 8.f - 4.f};
    Thrown t;
    t.sc = sc;
    t.radius = 0.85f * sc;
    t.rockVar = var;
    t.body = physics::addDynamic(rockHull[var], sc, &p.x, &v.x, w, 2600.f);
    t.xf = {{{sc, 0, 0, p.x}, {0, sc, 0, p.y}, {0, 0, sc, p.z}}};
    t.poly = std::make_shared<fracture::Poly>(rockPoly[var]);
    for (auto& fc : t.poly->faces) for (auto& q : fc.pts) q = {q.x * sc, q.y * sc, q.z * sc};
    memcpy(t.v, &v.x, 12);
    thrown.push_back(t);
}

// Fracture: a blow carrying normal energy En into a solid of cross-section A cracks it once En exceeds Gc * A
// (Gc = effective fracture energy per m^2 of crack, incl. what goes into heat and sound; tuned so a fist-sized
// stone survives a throw and a boulder breaks on a hard landing, big things being weaker for their size).
// Energy beyond that opens more crack surface: more, smaller pieces, the pattern spreading further from the
// contact as the ratio grows. Pieces under ~4 cm go to gravel particles.
static constexpr float FRACTURE_GC = 3e4f;
static float fractureThreshold(float radius) { return FRACTURE_GC * 3.14159f * radius * radius; }

static bool breakStatic(V3 p, V3 dir, float En, float reach);

// Procedural trees break by beam mechanics (trees::woodStrike / woodLanding): whatever hits a tree loads the
// limbs from the contact down to the roots, the weakest overloaded joint snaps, leaves near the blow fall. What
// still stands is a static Stump (its Wood in damagedTrees, hit again like a whole tree); what broke off is a
// wood piece on the thrown list carrying its Wood, so it snaps again where its landing loads it hardest.
// A wood's frame: world = R (s x - ctr) + T, x tree-local (unscaled).
struct WoodFrame { float R[3][3]; V3 T; float s; V3 ctr; };
static V3 woodRot(const WoodFrame& F, V3 l) {
    return {F.R[0][0] * l.x + F.R[0][1] * l.y + F.R[0][2] * l.z, F.R[1][0] * l.x + F.R[1][1] * l.y + F.R[1][2] * l.z,
            F.R[2][0] * l.x + F.R[2][1] * l.y + F.R[2][2] * l.z};
}
static V3 woodRotT(const WoodFrame& F, V3 d) {
    return {F.R[0][0] * d.x + F.R[1][0] * d.y + F.R[2][0] * d.z, F.R[0][1] * d.x + F.R[1][1] * d.y + F.R[2][1] * d.z,
            F.R[0][2] * d.x + F.R[1][2] * d.y + F.R[2][2] * d.z};
}
static V3 woodToWorld(const WoodFrame& F, V3 x) { return F.T + woodRot(F, x * F.s - F.ctr); }
static V3 worldToWood(const WoodFrame& F, V3 p) { return (woodRotT(F, p - F.T) + F.ctr) * (1.f / F.s); }
static WoodFrame treeFrame(const TreeInst& t) {
    float c = cosf(t.yaw), sn = sinf(t.yaw);
    return {{{c, 0, sn}, {0, 1, 0}, {-sn, 0, c}}, t.p, t.scale, {0, 0, 0}};
}
static WoodFrame thrownFrame(const Thrown& t) {
    WoodFrame F;
    for (int r = 0; r < 3; r++) for (int k = 0; k < 3; k++) F.R[r][k] = t.xf.m[r][k];
    F.T = {t.xf.m[0][3], t.xf.m[1][3], t.xf.m[2][3]};
    F.s = t.woodScale;
    F.ctr = t.woodCtr;
    return F;
}
static uint32_t woodSeed(const TreeInst& t) { return (uint32_t)(t.ix * 73856093) ^ (uint32_t)(t.iz * 19349663); }

// Hit of a procedural tree (standing or a loose piece: body != UINT32_MAX), carried out in the upload window.
struct WoodReq { TreeInst t; uint32_t body; trees::WoodHit hit; V3 at, J; float vmax = 1e30f; };  // vmax: impactor speed
static std::vector<WoodReq> pendingWood;

// A rock / wood piece / blast of mass at world point at with velocity vel against tree t. True if it touched the
// wood; then a break or leaf fall is queued and an impactor that broke through keeps the rest of its momentum.
static bool strikeTree(const TreeInst& t, V3 at, V3 vel, float mass, V3* took) {
    uint64_t key = treeKey(t.ix, t.iz);
    auto dt = damagedTrees.find(key);
    trees::Wood whole;
    const trees::Wood* w = nullptr;
    if (dt != damagedTrees.end()) w = &dt->second.second;
    else if (!brokenTrees.count(key)) { trees::woodWhole((int)t.v, whole); w = &whole; }
    if (!w) return false;
    WoodFrame F = treeFrame(t);
    V3 lp = worldToWood(F, at), lv = woodRotT(F, vel);
    trees::WoodHit h = trees::woodStrike(*w, t.scale, &lp.x, &lv.x, mass);
    if (treeLog) fprintf(stderr, "tree hit: %.0f kg at %.1f m/s, h %.2f m: stress %.2f, %zu cut, %d leaves\n", mass, sqrtf(dot(vel, vel)), at.y - t.p.y, h.ratio, h.cuts.size(), h.lost);
    V3 J = woodRot(F, V3{h.impulse[0], h.impulse[1], h.impulse[2]});
    if (took) *took = J;
    if (h.cuts.empty() && h.lost < 8) return true;
    for (auto& q : pendingWood) if (q.body == UINT32_MAX && q.t.ix == t.ix && q.t.iz == t.iz) return true;
    pendingWood.push_back({t, UINT32_MAX, std::move(h), at, J, sqrtf(dot(vel, vel))});
    return true;
}

// Rock (or wood piece) at contact point pc: the nearest procedural tree surface within reach takes the blow.
static bool strikeTrees(V3 pc, V3 vel, float mass, float reach) {
    const TreeInst* best = nullptr;
    float bd = reach;
    auto test = [&](const TreeInst& tr, const trees::Wood* w) {
        float dx = pc.x - tr.p.x, dz = pc.z - tr.p.z, R = treeR[tr.v] * 2.f * tr.scale + reach;
        if (dx * dx + dz * dz > R * R) return;
        trees::Wood whole;
        if (!w) { trees::woodWhole((int)tr.v, whole); w = &whole; }
        V3 lp = worldToWood(treeFrame(tr), pc);
        float d;
        if (trees::woodNearest(*w, &lp.x, d) >= 0 && d * tr.scale < bd) { bd = d * tr.scale; best = &tr; }
    };
    for (const TreeInst& tr : treeList) if (tr.v < (uint32_t)trees::VARIANTS) test(tr, nullptr);
    for (auto& [k, dw] : damagedTrees) test(dw.first, &dw.second);
    if (!best) return false;
    TreeInst t = *best;
    return strikeTree(t, pc, vel, mass);
}

// Thin limbs have no collider (a rock would fly through them): sweep what moved this step against them. A limb
// in the way takes the blow like any other hit, and the rock loses the momentum it gave the wood.
static void sweepThinLimbs(Thrown& t, V3 p0, V3 p1, V3 vel, float mass) {
    if (treeList.empty() && damagedTrees.empty()) return;
    V3 mid = (p0 + p1) * 0.5f;
    float half = sqrtf(dot(p1 - p0, p1 - p0)) * 0.5f;
    auto test = [&](const TreeInst& tr, const trees::Wood* w) {
        float dx = mid.x - tr.p.x, dz = mid.z - tr.p.z, R = treeR[tr.v] * 2.f * tr.scale + half + t.radius;
        if (dx * dx + dz * dz > R * R) return false;
        trees::Wood whole;
        if (!w) { trees::woodWhole((int)tr.v, whole); w = &whole; }
        WoodFrame F = treeFrame(tr);
        V3 a = worldToWood(F, p0), b = worldToWood(F, p1), at;
        if (trees::woodSweep(*w, &a.x, &b.x, t.radius * 0.8f / tr.scale, &at.x) < 0) return false;
        TreeInst ti = tr;
        V3 J{0, 0, 0}, pc = woodToWorld(F, at);
        strikeTree(ti, pc, vel, mass, &J);
        V3 nv = vel - J * (1.f / mass);
        physics::setVelocity(t.body, &nv.x);
        memcpy(t.v, &nv.x, 12);
        return true;
    };
    for (const TreeInst& tr : treeList) if (tr.v < (uint32_t)trees::VARIANTS && test(tr, nullptr)) return;
    for (auto& [k, dw] : damagedTrees) if (test(dw.first, &dw.second)) return;
}

// Meshes + bodies for woods in frame F (the pose they had as one): anchored ones become the standing Stump of
// tree cell key, loose ones thrown wood pieces moving with v + av x (centre - pivot). Pieces too small to be
// worth a body become splinters. Returns the bodies of the loose pieces (UINT32_MAX for those not spawned).
struct WoodSpawn { trees::Wood w; V3 v, av, pivot; };
static std::vector<uint32_t> spawnWoods(std::vector<WoodSpawn>& ws, const WoodFrame& F, uint64_t key, uint32_t seed) {
    std::vector<uint32_t> bodies(ws.size(), UINT32_MAX);
    std::vector<trees::Piece> pcs(ws.size());
    for (size_t i = 0; i < ws.size(); i++) trees::woodMesh(ws[i].w, seed, pcs[i]);
    auto fs = std::make_unique<FragSet>();
    Model& md = fs->model;
    md.path = "<tree pieces>";
    md.state = 1;
    takeTreeMaterials(md, pcs);
    float s3 = F.s * F.s * F.s;
    struct Out { size_t piece, mesh; Xf xf; float r; V3 ctr; };
    std::vector<Out> outs;
    for (size_t i = 0; i < pcs.size(); i++) {
        MeshData& m = pcs[i].mesh;
        if (m.idx.empty()) continue;
        if (!ws[i].w.anchored && pcs[i].mass * s3 < 0.3f) {  // a twig: splinters and its leaves
            V3 q = woodToWorld(F, V3{0.5f * (m.bmin[0] + m.bmax[0]), 0.5f * (m.bmin[1] + m.bmax[1]), 0.5f * (m.bmin[2] + m.bmax[2])});
            for (int k = 0; k < 3; k++) addDebris(DEB_WOOD, q + rndDir() * 0.1f, ws[i].v + rndDir() * 1.5f, 0.03f + 0.04f * rnd01());
            for (int k = 0; k < 6; k++) addDebris(DEB_LEAF, q + rndDir() * 0.2f, ws[i].v + rndDir() * 0.8f, 0.035f + 0.03f * rnd01());
            continue;
        }
        V3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
        for (size_t k = 0; k < m.verts.size(); k += 8) {
            for (int j = 0; j < 3; j++) m.verts[k + j] *= F.s;
            V3 q{m.verts[k], m.verts[k + 1], m.verts[k + 2]};
            lo = {std::min(lo.x, q.x), std::min(lo.y, q.y), std::min(lo.z, q.z)};
            hi = {std::max(hi.x, q.x), std::max(hi.y, q.y), std::max(hi.z, q.z)};
        }
        V3 ctr = (lo + hi) * 0.5f;
        float r = 0.f;
        for (size_t k = 0; k < m.verts.size(); k += 8) {
            m.verts[k] -= ctr.x; m.verts[k + 1] -= ctr.y; m.verts[k + 2] -= ctr.z;
            r = std::max(r, m.verts[k] * m.verts[k] + m.verts[k + 1] * m.verts[k + 1] + m.verts[k + 2] * m.verts[k + 2]);
        }
        for (int k = 0; k < 3; k++) { m.bmin[k] = (&lo.x)[k] - (&ctr.x)[k]; m.bmax[k] = (&hi.x)[k] - (&ctr.x)[k]; }
        for (auto* col : {&pcs[i].colVerts, &pcs[i].hullPts})
            for (size_t k = 0; k < col->size(); k += 3)
                for (int j = 0; j < 3; j++) (*col)[k + j] = (*col)[k + j] * F.s - (&ctr.x)[j];
        V3 pos = F.T + woodRot(F, ctr - F.ctr);
        Xf xf = {{{F.R[0][0], F.R[0][1], F.R[0][2], pos.x}, {F.R[1][0], F.R[1][1], F.R[1][2], pos.y}, {F.R[2][0], F.R[2][1], F.R[2][2], pos.z}}};
        md.data.meshes.push_back(std::move(m));
        outs.push_back({i, md.data.meshes.size() - 1, xf, sqrtf(r), ctr});
    }
    if (outs.empty()) return bodies;
    uploadModel(md);
    FragSet* set = fs.get();
    fragSets.push_back(std::move(fs));
    for (const Out& o : outs) {
        trees::Piece& pc = pcs[o.piece];
        const MeshData& mesh = set->model.data.meshes[o.mesh];
        if (ws[o.piece].w.anchored) {
            if (pc.colIdx.empty()) continue;
            void* shape = physics::meshShape(pc.colVerts.data(), 3, (uint32_t)pc.colVerts.size() / 3, pc.colIdx.data(), (uint32_t)pc.colIdx.size() / 3);
            if (!shape) continue;
            if (stumps.size() >= (size_t)MAX_FALLEN) dropStump(0);
            Stump st{set, (uint32_t)o.mesh, 0, shape, o.xf, key};
            physics::addStatic(&st.shape, &st.xf, 1, &st.body);
            set->live++;
            stumps.push_back(st);
            continue;
        }
        void* shape = pc.hullCount.empty() ? nullptr : physics::compoundShape(pc.hullPts.data(), pc.hullCount.data(), (uint32_t)pc.hullCount.size());
        if (!shape) {  // only thin wood: the hull of the mesh (at most ~4k points)
            uint32_t nv = (uint32_t)mesh.verts.size() / 8;
            std::vector<float> sub;
            for (uint32_t k = 0; k < nv; k += std::max(1u, nv / 4096)) sub.insert(sub.end(), &mesh.verts[(size_t)k * 8], &mesh.verts[(size_t)k * 8 + 3]);
            shape = physics::convexShape(sub.data(), 3, (uint32_t)sub.size() / 3);
        }
        if (!shape) continue;
        float dens = std::clamp(pc.mass * s3 / std::max(physics::shapeVolume(shape), 1e-5f), 20.f, 2000.f);
        const WoodSpawn& sp = ws[o.piece];
        V3 c{o.xf.m[0][3], o.xf.m[1][3], o.xf.m[2][3]};
        V3 v = sp.v + cross(sp.av, c - sp.pivot);
        if (thrown.size() >= (size_t)MAX_THROWN) dropThrown(0);
        Thrown f;
        f.sc = 1.f;
        f.radius = o.r;
        f.frag = set;
        f.fragMesh = (uint32_t)o.mesh;
        f.shape = shape;
        {   // the real mass distribution: leaves spread over the crown make it hard to spin
            float wm, com[3], I[9];
            trees::woodMass(sp.w, F.s, wm, com, I);
            for (int k = 0; k < 3; k++) com[k] -= (&o.ctr.x)[k];
            f.body = wm > 0.1f ? physics::addDynamicPoseInertia(shape, o.xf, &v.x, &sp.av.x, wm, com, I)
                               : physics::addDynamicPose(shape, o.xf, &v.x, &sp.av.x, dens);
        }
        {   // air drag: a leafy crown is a sail, it tumbles slowly and settles; bare wood keeps a little spin
            int nl = 0;
            for (uint16_t n : sp.w.leaves) nl += n;
            float leafy = std::min((float)nl / std::max(pc.mass * s3, 1.f) / 40.f, 1.f);  // leaves per kg of wood
            physics::setDamping(f.body, 0.05f + 0.5f * leafy, 0.8f + 3.f * leafy);
        }
        f.xf = o.xf;
        f.cooldown = 0.25f;  // pieces settling off what they broke from aren't impacts
        f.density = dens;
        f.wood = true;
        f.woodPart = std::make_shared<trees::Wood>(sp.w);
        f.woodScale = F.s;
        f.woodCtr = o.ctr;
        f.woodSeed = seed;
        memcpy(f.v, &v.x, 12);
        set->live++;
        bodies[o.piece] = f.body;
        thrown.push_back(std::move(f));
    }
    if (set->live == 0) {
        auto keep = std::shared_ptr<FragSet>(fragSets.back().release());
        fragSets.pop_back();
        retireNext.push_back([keep]() { if (keep->model.gpu) releaseModel(keep->model); });
    }
    return bodies;
}

static void processWood() {
    std::vector<WoodReq> wq;
    wq.swap(pendingWood);
    for (WoodReq& q : wq) {
        std::vector<trees::Wood> off;
        std::vector<float> shed;
        std::vector<WoodSpawn> ws;
        WoodFrame F;
        V3 v{0, 0, 0}, av{0, 0, 0}, pivot{0, 0, 0};
        uint64_t key = 0;
        uint32_t seed;
        if (q.body == UINT32_MAX) {  // standing tree
            key = treeKey(q.t.ix, q.t.iz);
            trees::Wood w;
            auto dt = damagedTrees.find(key);
            if (dt != damagedTrees.end()) w = dt->second.second;
            else if (!brokenTrees.count(key)) trees::woodWhole((int)q.t.v, w);
            else continue;
            trees::woodApply(w, q.hit, off, &shed);
            for (size_t i = 0; i < stumps.size(); i++)
                if (stumps[i].key == key) { stumps[i].key = 0; dropStump(i); break; }
            if (brokenTrees.insert(key).second) {
                placeDirty = true;
                auto it = treeBodies.find(key);
                if (it != treeBodies.end()) { physics::removeBodies(&it->second, 1); treeBodies.erase(it); }
            }
            F = treeFrame(q.t);
            seed = woodSeed(q.t);
            damagedTrees[key] = {q.t, w};
            ws.push_back({std::move(w), v, av, pivot});
        } else {  // a loose piece landed
            size_t i = 0;
            while (i < thrown.size() && thrown[i].body != q.body) i++;
            if (i == thrown.size() || !thrown[i].woodPart) continue;
            float mass;
            if (!physics::bodyState(thrown[i].body, 1.f, thrown[i].xf, &v.x, &av.x, mass)) continue;
            F = thrownFrame(thrown[i]);
            pivot = F.T;
            seed = thrown[i].woodSeed;
            trees::Wood w = *thrown[i].woodPart;
            trees::woodApply(w, q.hit, off, &shed);
            dropThrown(i);
            ws.push_back({std::move(w), v, av, pivot});
        }
        for (trees::Wood& o : off) ws.push_back({std::move(o), v, av, pivot});
        std::vector<uint32_t> bodies = spawnWoods(ws, F, key, seed);
        if (q.body == UINT32_MAX && !q.hit.cuts.empty()) {
            // the snapped part takes the blow at the contact (it falls, twisting, the way it was pushed)
            // (no faster than what hit it: they go on together)
            for (size_t i = 1; i < bodies.size(); i++) {
                float vb[3], wb[3], mb;
                Xf xb;
                if (bodies[i] == UINT32_MAX || !physics::bodyState(bodies[i], 1.f, xb, vb, wb, mb)) continue;
                float jl = sqrtf(dot(q.J, q.J)), cap = 0.8f * q.vmax * mb;
                V3 J = jl > cap ? q.J * (cap / jl) : q.J;
                physics::addImpulse(bodies[i], &J.x, &q.at.x);
                // a snapping limb doesn't fly off like a struck propeller: the rock stays in contact while the
                // joint tears, so much of the push goes in at its centre. Spin no more than ~half a turn per
                // metre of the rock's travel over the piece's length.
                float vn[3], wn[3], mn;
                Xf xn;
                if (physics::bodyState(bodies[i], 1.f, xn, vn, wn, mn)) {
                    float wl = sqrtf(wn[0] * wn[0] + wn[1] * wn[1] + wn[2] * wn[2]), r = 0.5f;
                    for (const Thrown& th : thrown) if (th.body == bodies[i]) r = std::max(th.radius, 0.3f);
                    float cap = std::max(1.5f, 0.35f * q.vmax / r);
                    if (wl > cap) { for (float& c : wn) c *= cap / wl; physics::setAngularVelocity(bodies[i], wn); }
                }
                break;
            }
            addDecal(q.at, q.J * -1.f, 0.12f, 2);
        }
        // splinters from every snap, leaves where they hung
        float sev = std::min(q.hit.ratio, 4.f);
        if (!q.hit.cuts.empty()) {
            for (size_t c = 0; c < q.hit.cuts.size(); c++)
                for (int i = 0; i < (int)(6.f * sev); i++)
                    addDebris(DEB_WOOD, q.at + rndDir() * (0.2f * F.s), v + rndDir() * (1.5f + 3.f * rnd01()) + V3{0, 1.5f, 0}, 0.03f + 0.06f * rnd01());
        }
        V3 dir = dot(q.J, q.J) > 1e-6f ? normalize(q.J) : V3{0, 0, 0};
        size_t nShed = shed.size() / 3, stepL = std::max<size_t>(1, nShed / 600);
        for (size_t i = 0; i < nShed; i += stepL) {
            V3 lq = woodToWorld(F, V3{shed[i * 3], shed[i * 3 + 1], shed[i * 3 + 2]});
            V3 lv = v + cross(av, lq - pivot);
            addDebris(DEB_LEAF, lq, lv + rndDir() * (0.4f + 1.2f * rnd01()) + dir * (1.2f * rnd01()), 0.035f + 0.03f * rnd01(), 0.3f * rnd01());
        }
    }
}

static void processFractures() {
    std::vector<TreeBreakReq> tq;
    tq.swap(pendingTreeBreaks);
    for (auto& q : tq) breakTreeNow(q.t, q.impulse, q.at, q.severity);
    processWood();
    std::vector<WallBreakReq> wq;
    wq.swap(pendingWallBreaks);
    for (auto& q : wq) {
        bool broke = breakStatic(q.p, q.dir, q.En, q.reach);
        if (!broke) {  // too strong to break: rubble and dust
            int nd = std::min((int)(q.En / 4e3f), 120);
            for (int i = 0; i < nd; i++)
                addDebris(DEB_DIRT, q.p + rndDir() * (q.reach - 0.4f), rndDir() * (1.f + 4.f * rnd01()) + V3{0, 1.5f, 0}, 0.03f + 0.08f * rnd01());
            continue;
        }
        for (auto& t : thrown)  // the rock carries on through the breach with part of its speed
            if (t.body == q.body) { V3 nv = q.v * 0.6f; physics::setVelocity(t.body, &nv.x); break; }
    }
    if (pendingFractures.empty()) return;
    static TextureData tex = rockTexture();
    std::vector<FractureReq> reqs;
    reqs.swap(pendingFractures);
    for (const FractureReq& rq : reqs) {
        size_t i = 0;
        while (i < thrown.size() && thrown[i].body != rq.body) i++;
        if (i == thrown.size() || !thrown[i].poly) continue;
        Thrown& t = thrown[i];
        float v[3], w[3], mass;
        Xf xf;
        if (!physics::bodyState(t.body, 1.f, xf, v, w, mass)) continue;
        // body frame: rotation only (bodyState with scale 1), the poly is already at the body's size
        auto toLocal = [&](V3 p) { V3 d = p - V3{xf.m[0][3], xf.m[1][3], xf.m[2][3]};
                                   return fracture::Vec{xf.m[0][0] * d.x + xf.m[1][0] * d.y + xf.m[2][0] * d.z,
                                                        xf.m[0][1] * d.x + xf.m[1][1] * d.y + xf.m[2][1] * d.z,
                                                        xf.m[0][2] * d.x + xf.m[1][2] * d.y + xf.m[2][2] * d.z}; };
        auto rotW = [&](fracture::Vec a) { return V3{xf.m[0][0] * a.x + xf.m[0][1] * a.y + xf.m[0][2] * a.z,
                                                     xf.m[1][0] * a.x + xf.m[1][1] * a.y + xf.m[1][2] * a.z,
                                                     xf.m[2][0] * a.x + xf.m[2][1] * a.y + xf.m[2][2] * a.z}; };
        fracture::Vec ip = toLocal(rq.contact), idir = toLocal(V3{xf.m[0][3], xf.m[1][3], xf.m[2][3]} + rq.dir);
        float Ec = fractureThreshold(t.radius), ratio = rq.En / Ec;
        int count = std::clamp((int)(2.f + 5.f * log2f(ratio)), 2, 64);
        float spread = std::clamp(log10f(ratio) / 3.f, 0.f, 1.f);
        std::vector<fracture::Poly> pieces;
        fracture::shatter(*t.poly, ip, idir, count, spread, frameIndex * 31u + (uint32_t)i, pieces);
        if (pieces.size() < 2) continue;

        auto fs = std::make_unique<FragSet>();
        Model& md = fs->model;
        md.path = "<fragments>";
        md.state = 1;
        md.data.textures = {tex};
        MaterialData outer, inner;  // weathered outside, fresh lighter rock on the new faces
        outer.color[0] = 0.30f; outer.color[1] = 0.29f; outer.color[2] = 0.26f; outer.tex = 0;
        inner.color[0] = 0.50f; inner.color[1] = 0.48f; inner.color[2] = 0.44f; inner.tex = 0;
        md.data.materials = {outer, inner};
        float vKick = sqrtf(2.f * 0.15f * std::max(rq.En - Ec, 0.f) / std::max(mass, 1e-3f));
        V3 P0{xf.m[0][3], xf.m[1][3], xf.m[2][3]}, W{w[0], w[1], w[2]};
        struct Spawn { size_t mesh; Xf xf; V3 v, w; fracture::Poly poly; float r; };
        std::vector<Spawn> spawns;
        for (auto& pc : pieces) {
            fracture::Vec c;
            float vol = fracture::volume(pc, &c);
            float r = fracture::radius(pc, c);
            V3 cw = P0 + rotW(c), out = normalize(rotW({c.x - ip.x, c.y - ip.y, c.z - ip.z}) + rq.dir * 0.5f);
            V3 vel = V3{v[0], v[1], v[2]} + cross(W, cw - P0) + out * (vKick * (0.5f + rnd01()));
            if (r < 0.04f || vol < 2e-5f) {  // gravel
                addDebris(DEB_DIRT, cw, vel, std::max(r, 0.01f));
                continue;
            }
            fracture::translate(pc, {-c.x, -c.y, -c.z});
            MeshData mesh;
            fracture::toMesh(pc, mesh, 0, 1);
            md.data.meshes.push_back(std::move(mesh));
            Xf px = xf;
            px.m[0][3] = cw.x; px.m[1][3] = cw.y; px.m[2][3] = cw.z;
            spawns.push_back({md.data.meshes.size() - 1, px, vel, W + rndDir() * (2.f * rnd01()), pc, r});
        }
        dropThrown(i);  // parent goes; its references are gone after this
        if (spawns.empty()) continue;
        uploadModel(md);
        FragSet* set = fs.get();
        fragSets.push_back(std::move(fs));
        for (auto& sp : spawns) {
            if (thrown.size() >= (size_t)MAX_THROWN) dropThrown(0);
            const MeshData& mesh = set->model.data.meshes[sp.mesh];
            Thrown f;
            f.sc = 1.f;
            f.radius = sp.r;
            f.frag = set;
            f.fragMesh = (uint32_t)sp.mesh;
            f.shape = physics::convexShape(mesh.verts.data(), 8, (uint32_t)mesh.verts.size() / 8);
            if (!f.shape) continue;
            f.body = physics::addDynamicPose(f.shape, sp.xf, &sp.v.x, &sp.w.x, 2600.f);
            f.xf = sp.xf;
            f.poly = std::make_shared<fracture::Poly>(std::move(sp.poly));
            f.cooldown = 0.1f;  // pieces separating aren't new impacts
            memcpy(f.v, &sp.v.x, 12);
            set->live++;
            thrown.push_back(std::move(f));
        }
        if (set->live == 0) {  // nothing spawned after all
            auto keep = std::shared_ptr<FragSet>(fragSets.back().release());
            fragSets.pop_back();
            retireNext.push_back([keep]() { if (keep->model.gpu) releaseModel(keep->model); });
        }
    }
}

// ---------------------------------------------------------------- breakable buildings
// Static meshes (walls, floors, crates, ...) are unions of convex pieces, mostly boxes: a mesh's connected
// components. A hard enough blow breaches the components it reaches: each is shattered around the contact
// (fracture::shatter); cells inside the breach radius fly off as fragments (thrown, and can break again), cells
// that lost every hold on the rest of the structure fall too, and the remaining cells stay as static geometry in a
// damaged copy of the mesh that replaces it in the world (stream::overridePlaced, survives cell reloads).
// Breach area = En / WALL_GC.
static constexpr float WALL_GC = 2e5f;  // J per m^2 of breach (brick / plaster wall incl. what goes into dust)

static const MeshParts& meshParts(const MeshData& m) {
    auto it = meshPartsCache.find(&m);
    if (it != meshPartsCache.end()) return it->second;
    uint32_t nv = (uint32_t)m.verts.size() / 8;
    // weld by position (1 mm), then union-find over triangles
    std::vector<uint32_t> par(nv);
    for (uint32_t i = 0; i < nv; i++) par[i] = i;
    auto find = [&](uint32_t x) { while (par[x] != x) x = par[x] = par[par[x]]; return x; };
    auto unite = [&](uint32_t a, uint32_t b) { a = find(a); b = find(b); if (a != b) par[a] = b; };
    std::unordered_map<uint64_t, uint32_t> weld;
    for (uint32_t i = 0; i < nv; i++) {
        auto q = [&](int k) { return (uint64_t)(uint32_t)(int32_t)lroundf(m.verts[i * 8 + k] * 1000.f) & 0x1FFFFF; };
        auto [w, fresh] = weld.emplace(q(0) | q(1) << 21 | q(2) << 42, i);
        if (!fresh) unite(i, w->second);
    }
    for (size_t t = 0; t < m.idx.size(); t += 3) { unite(m.idx[t], m.idx[t + 1]); unite(m.idx[t], m.idx[t + 2]); }
    MeshParts mp;
    std::unordered_map<uint32_t, uint32_t> comp;
    for (size_t t = 0; t < m.idx.size(); t += 3) {
        uint32_t root = find(m.idx[t]);
        auto [c, fresh] = comp.emplace(root, (uint32_t)mp.tris.size());
        if (fresh) { mp.tris.emplace_back(); mp.box.push_back({1e30f, 1e30f, 1e30f, -1e30f, -1e30f, -1e30f}); }
        mp.tris[c->second].push_back((uint32_t)(t / 3));
        auto& b = mp.box[c->second];
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++) {
                float x = m.verts[m.idx[t + j] * 8 + k];
                b[k] = std::min(b[k], x); b[3 + k] = std::max(b[3 + k], x);
            }
    }
    return meshPartsCache.emplace(&m, std::move(mp)).first->second;
}

static Xf xfInverse(const Xf& a) {
    const float(*m)[4] = a.m;
    float det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    float id = 1.f / det;
    Xf r;
    r.m[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * id; r.m[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id; r.m[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id;
    r.m[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * id; r.m[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id; r.m[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * id;
    r.m[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * id; r.m[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * id; r.m[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id;
    for (int i = 0; i < 3; i++) r.m[i][3] = -(r.m[i][0] * m[0][3] + r.m[i][1] * m[1][3] + r.m[i][2] * m[2][3]);
    return r;
}

// Masonry: the wall part (a box in the mesh's local space, vertical, thin along one horizontal axis) comes apart
// along its mortar. Courses of 75 mm, bricks 225 mm in running bond; bricks inside the breach (and some at its rim)
// fly out whole, the rest of each touched course stays as convex runs between the gaps, untouched courses as one
// block. Local cells go into dm, loose bricks to flyCell (world space). False if the part isn't a wall.
template <class Fly>
static bool brickBreach(const Placed& hit, const std::array<float, 6>& box, V3 p, V3 dir, float rb, uint32_t outer, uint32_t inner, MeshData& dm, Fly&& flyCell) {
    const float* mn = &box[0];
    const float* mx = &box[3];
    int thin = mx[0] - mn[0] < mx[2] - mn[2] ? 0 : 2, along = 2 - thin;
    if (mx[1] - mn[1] < 0.3f || mx[thin] - mn[thin] > 0.6f) return false;  // a slab / pillar: shatter instead
    const float BL = 0.225f, BH = 0.075f;
    Xf inv = xfInverse(hit.xf);
    float pw[3] = {p.x, p.y, p.z}, pl[3];
    xfPoint(inv, pw, pl);
    // local box poly
    std::vector<float> corners;
    for (int i = 0; i < 8; i++) corners.insert(corners.end(), {i & 1 ? mx[0] : mn[0], i & 2 ? mx[1] : mn[1], i & 4 ? mx[2] : mn[2]});
    fracture::Poly whole = fracture::fromPoints(corners.data(), 3, 8, 26);
    auto slab = [&](const fracture::Poly& q, int axis, float lo, float hi) {
        fracture::Vec n{0, 0, 0};
        (&n.x)[axis] = 1.f;
        fracture::Poly r = fracture::clip(q, n, hi);
        (&n.x)[axis] = -1.f;
        return fracture::clip(r, n, -lo);
    };
    auto stay = [&](const fracture::Poly& q) {
        if (q.faces.empty()) return;
        MeshData pm;
        fracture::toMesh(q, pm, outer, inner, 1.f);
        uint32_t base = (uint32_t)(dm.verts.size() / 8);
        dm.verts.insert(dm.verts.end(), pm.verts.begin(), pm.verts.end());
        for (uint32_t i : pm.idx) dm.idx.push_back(base + i);
        dm.triMat.insert(dm.triMat.end(), pm.triMat.begin(), pm.triMat.end());
    };
    auto world = [&](fracture::Poly q) {
        for (auto& f : q.faces)
            for (auto& v : f.pts) { float a[3] = {v.x, v.y, v.z}, o[3]; xfPoint(hit.xf, a, o); v = {o[0], o[1], o[2]}; }
        return q;
    };
    float reach = rb * 1.3f;
    int rows = std::max(1, (int)ceilf((mx[1] - mn[1]) / BH));
    float untouched0 = mn[1];  // bottom of the current run of untouched courses
    bool inRun = false;
    for (int r = 0; r <= rows; r++) {
        float y0 = mn[1] + r * BH, y1 = std::min(mx[1], y0 + BH);
        bool touched = r < rows && fabsf((y0 + y1) * 0.5f - pl[1]) < reach;
        if (!touched || r == rows) {
            if (r < rows) { if (!inRun) { untouched0 = y0; inRun = true; } continue; }
        }
        if (inRun) { stay(slab(whole, 1, untouched0, y0)); inRun = false; }
        if (r == rows) break;
        fracture::Poly course = slab(whole, 1, y0, y1);
        float off = (r & 1) ? BL * 0.5f : 0.f;
        float runStart = mn[along];
        for (float a0 = mn[along] - off; a0 < mx[along]; a0 += BL) {
            float lo = std::max(a0, mn[along]), hi = std::min(a0 + BL, mx[along]);
            if (hi - lo < 1e-3f) continue;
            float c[3];
            c[thin] = pl[thin]; c[1] = (y0 + y1) * 0.5f; c[along] = (lo + hi) * 0.5f;
            float d = sqrtf((c[1] - pl[1]) * (c[1] - pl[1]) + (c[along] - pl[along]) * (c[along] - pl[along]));
            uint32_t h = (uint32_t)(r * 7919 + (int)(a0 * 1000.f)) * 2654435761u;
            bool loose = d < rb || (d < reach && (h >> 16) % 100 < 45);
            if (!loose) continue;
            if (lo > runStart + 1e-3f) stay(slab(course, along, runStart, lo));
            flyCell(world(slab(course, along, lo, hi)), outer, inner, d < rb);
            runStart = hi;
        }
        if (runStart < mx[along] - 1e-3f) stay(slab(course, along, runStart, mx[along]));
    }
    for (int i = 0; i < (int)std::min(30.f * rb, 100.f); i++)  // mortar dust
        addDebris(DEB_DIRT, p + rndDir() * (rb * 0.6f), (dir * 2.f + rndDir()) * (1.f + 3.f * rnd01()), 0.01f + 0.02f * rnd01());
    return true;
}

static float vlen(V3 a) { return sqrtf(dot(a, a)); }
// Metal: a blow below tearing energy dents it. The triangles near the contact are subdivided down to ~4 cm and
// pushed in along the blow with a smooth falloff (the whole thickness moves, like a plate bending); the dented
// copy replaces the mesh like a breach does, so dents add up.
static bool dentMetal(const Placed& hit, V3 p, V3 dir, float En) {
    const MeshData& src = hit.model->data.meshes[hit.mesh];
    float s = std::max(src.strength, 0.1f);
    float depth = std::clamp(0.015f + 0.3f * sqrtf(En / (s * 1e6f)), 0.01f, 0.5f);
    float R = 0.2f + 1.6f * depth;
    Xf inv = xfInverse(hit.xf);
    float pw[3] = {p.x, p.y, p.z}, pl[3];
    xfPoint(inv, pw, pl);
    float dw[3] = {dir.x, dir.y, dir.z}, dl[3];
    for (int i = 0; i < 3; i++) dl[i] = inv.m[i][0] * dw[0] + inv.m[i][1] * dw[1] + inv.m[i][2] * dw[2];
    float dn = sqrtf(dl[0] * dl[0] + dl[1] * dl[1] + dl[2] * dl[2]);
    for (float& x : dl) x /= std::max(dn, 1e-6f);
    // subdivide near the contact (verts are pos3 nrm3 uv2)
    std::vector<float> V = src.verts;
    std::vector<uint32_t> I = src.idx, M = src.triMat;
    if (M.empty()) M.assign(I.size() / 3, 0u);
    auto P = [&](uint32_t i) { return V3{V[i * 8], V[i * 8 + 1], V[i * 8 + 2]}; };
    V3 c{pl[0], pl[1], pl[2]};
    const float target = 0.04f;
    for (int pass = 0; pass < 10; pass++) {
        std::vector<uint32_t> nI, nM;
        bool any = false;
        std::unordered_map<uint64_t, uint32_t> mids;
        auto mid = [&](uint32_t a, uint32_t b) {
            uint64_t k = ((uint64_t)std::min(a, b) << 32) | std::max(a, b);
            auto it = mids.find(k);
            if (it != mids.end()) return it->second;
            uint32_t n = (uint32_t)(V.size() / 8);
            for (int j = 0; j < 8; j++) V.push_back(0.5f * (V[a * 8 + j] + V[b * 8 + j]));
            mids[k] = n;
            return n;
        };
        for (size_t t = 0; t < I.size(); t += 3) {
            uint32_t a = I[t], b = I[t + 1], d = I[t + 2];
            V3 pa = P(a), pb = P(b), pd = P(d);
            float e = std::max({vlen(pb - pa), vlen(pd - pb), vlen(pa - pd)});
            V3 ctr = (pa + pb + pd) * (1.f / 3.f);
            if (e > target && vlen(ctr - c) < R + e && nI.size() < 600000) {
                uint32_t ab = mid(a, b), bd = mid(b, d), da = mid(d, a);
                nI.insert(nI.end(), {a, ab, da, ab, b, bd, da, bd, d, ab, bd, da});
                nM.insert(nM.end(), {M[t / 3], M[t / 3], M[t / 3], M[t / 3]});
                any = true;
            } else { nI.insert(nI.end(), {a, b, d}); nM.push_back(M[t / 3]); }
        }
        I.swap(nI); M.swap(nM);
        if (!any) break;
    }
    // edges split on one side only leave T-junctions; with every vertex displaced by a smooth field they stay tiny
    std::vector<bool> moved(V.size() / 8, false);
    for (size_t i = 0; i < V.size() / 8; i++) {
        V3 q = P((uint32_t)i), d = q - c;
        float along = d.x * dl[0] + d.y * dl[1] + d.z * dl[2];
        V3 lat = d - V3{dl[0], dl[1], dl[2]} * along;
        float x = vlen(lat) / R;
        if (x >= 1.f) continue;
        float f = (1.f - x * x) * (1.f - x * x);
        for (int k = 0; k < 3; k++) V[i * 8 + k] += dl[k] * depth * f;
        moved[i] = true;
    }
    // flat normals for the dented triangles (shared verts would smear across the plate's corners)
    std::vector<float> outV;
    std::vector<uint32_t> outI;
    outV.reserve(V.size());
    std::vector<uint32_t> remap(V.size() / 8, UINT32_MAX);
    for (size_t t = 0; t < I.size(); t += 3) {
        bool dented = moved[I[t]] || moved[I[t + 1]] || moved[I[t + 2]];
        V3 a = P(I[t]), b = P(I[t + 1]), d = P(I[t + 2]);
        V3 n = normalize(cross(b - a, d - a));
        for (int j = 0; j < 3; j++) {
            uint32_t vi = I[t + j];
            if (!dented && remap[vi] != UINT32_MAX) { outI.push_back(remap[vi]); continue; }
            uint32_t ni = (uint32_t)(outV.size() / 8);
            outV.insert(outV.end(), &V[vi * 8], &V[vi * 8 + 8]);
            if (dented) { outV[ni * 8 + 3] = n.x; outV[ni * 8 + 4] = n.y; outV[ni * 8 + 5] = n.z; }
            else remap[vi] = ni;
            outI.push_back(ni);
        }
    }
    auto copy = std::make_shared<Model>();
    copy->state = 1;
    ModelData& cd = copy->data;
    const ModelData& sd = hit.model->data;
    cd.materials = sd.materials;
    std::unordered_map<int, int> texMap;
    for (auto& mt : cd.materials)
        forEachTex(mt, [&](int t) {
            if (t < 0) return t;
            auto [e, fresh] = texMap.emplace(t, (int)cd.textures.size());
            if (fresh) cd.textures.push_back(sd.textures[t]);
            return e->second;
        });
    MeshData dm;
    dm.verts = std::move(outV);
    dm.idx = std::move(outI);
    dm.triMat = std::move(M);
    dm.strength = src.strength;
    dm.breakKind = src.breakKind;
    for (int k = 0; k < 3; k++) { dm.bmin[k] = 1e30f; dm.bmax[k] = -1e30f; }
    for (size_t i = 0; i < dm.verts.size(); i += 8)
        for (int k = 0; k < 3; k++) { dm.bmin[k] = std::min(dm.bmin[k], dm.verts[i + k]); dm.bmax[k] = std::max(dm.bmax[k], dm.verts[i + k]); }
    dm.shape = physics::meshShape(dm.verts.data(), 8, (uint32_t)(dm.verts.size() / 8), dm.idx.data(), (uint32_t)(dm.idx.size() / 3));
    cd.meshes.push_back(std::move(dm));
    if (!uploadModel(*copy)) {
        if (cd.meshes[0].shape) physics::releaseShape(cd.meshes[0].shape);
        return false;
    }
    std::shared_ptr<Model> old = stream::overridePlaced(stream::placedKey(hit), copy);
    damagedModels.push_back(copy);
    if (old) {
        damagedModels.erase(std::remove(damagedModels.begin(), damagedModels.end(), old), damagedModels.end());
        retireNext.push_back([old]() {
            if (old->gpu) releaseModel(*old);
            if (old->data.meshes[0].shape) physics::releaseShape(old->data.meshes[0].shape);
        });
    }
    onStreamChanged();
    for (int i = 0; i < 12; i++)  // a few sparks / flakes of paint
        addDebris(DEB_DIRT, p - dir * 0.05f, (rndDir() - dir) * (1.f + 2.f * rnd01()), 0.006f + 0.01f * rnd01());
    return true;
}

static bool breakStaticImpl(V3 p, V3 dir, float En, float reach);
static bool breakStatic(V3 p, V3 dir, float En, float reach) {
    // the impact marks the wall and kicks up plaster / brick dust whether or not it breaks through
    addDecal(p, dir * -1.f, std::clamp(0.25f + 0.4f * reach, 0.25f, 1.5f), 0);
    addDust(p - dir * 0.3f, std::clamp(0.4f + 0.5f * reach, 0.4f, 2.5f), 0.6f, 0.25f);
    return breakStaticImpl(p, dir, En, reach);
}
static bool breakStaticImpl(V3 p, V3 dir, float En, float reach) {
    // the placed mesh whose nearest component is closest to the contact
    Placed hit{};
    int hitPart = -1;
    float best = reach;
    stream::forEachPlaced([&](const Placed& pl) {
        const MeshData& m = pl.model->data.meshes[pl.mesh];
        if (!m.shape || m.idx.empty()) return;
        float mn[3], mx[3];
        xfAabb(pl.xf, m.bmin, m.bmax, mn, mx);
        if (p.x < mn[0] - reach || p.y < mn[1] - reach || p.z < mn[2] - reach || p.x > mx[0] + reach || p.y > mx[1] + reach || p.z > mx[2] + reach) return;
        const MeshParts& mp = meshParts(m);
        for (size_t i = 0; i < mp.box.size(); i++) {
            float bmn[3], bmx[3];
            xfAabb(pl.xf, &mp.box[i][0], &mp.box[i][3], bmn, bmx);
            float d2 = 0.f;
            for (int k = 0; k < 3; k++) { float e = std::max({bmn[k] - (&p.x)[k], (&p.x)[k] - bmx[k], 0.f}); d2 += e * e; }
            if (sqrtf(d2) < best) { best = sqrtf(d2); hit = pl; hitPart = (int)i; }
        }
    });
    if (hitPart < 0) return false;
    const MeshData& src = hit.model->data.meshes[hit.mesh];
    float rb = sqrtf(En / (WALL_GC * std::max(src.strength, 0.01f) * 3.14159f));  // breach radius
    int kind = src.breakKind;
    if (kind == BREAK_SHATTER && !src.triMat.empty() && hit.model->data.materials[src.triMat[0]].metal > 0.5f) kind = BREAK_METAL;
    if (kind == BREAK_METAL && rb < 0.7f) return dentMetal(hit, p, dir, En);  // tears only when hit very hard
    if (rb < 0.15f) return false;
    const MeshParts& mp = meshParts(src);
    Xf inv = xfInverse(hit.xf);
    auto toLocal = [&](fracture::Vec v) { float a[3] = {v.x, v.y, v.z}, o[3]; xfPoint(inv, a, o); return fracture::Vec{o[0], o[1], o[2]}; };
    // components the breach reaches (always the one hit)
    std::vector<int> parts;
    for (size_t i = 0; i < mp.box.size(); i++) {
        float bmn[3], bmx[3];
        xfAabb(hit.xf, &mp.box[i][0], &mp.box[i][3], bmn, bmx);
        float d2 = 0.f;
        for (int k = 0; k < 3; k++) { float e = std::max({bmn[k] - (&p.x)[k], (&p.x)[k] - bmx[k], 0.f}); d2 += e * e; }
        if ((int)i == hitPart || sqrtf(d2) < 0.5f * rb) parts.push_back((int)i);
    }

    // damaged copy: the untouched triangles, then the static remains of each breached part
    auto copy = std::make_shared<Model>();
    copy->state = 1;
    ModelData& cd = copy->data;
    const ModelData& sd = hit.model->data;
    cd.materials = sd.materials;
    std::unordered_map<int, int> texMap;  // only the textures this mesh's materials use
    for (auto& mt : cd.materials)
        forEachTex(mt, [&](int t) {
            if (t < 0) return t;
            auto [e, fresh] = texMap.emplace(t, (int)cd.textures.size());
            if (fresh) cd.textures.push_back(sd.textures[t]);
            return e->second;
        });
    std::vector<bool> gone(src.idx.size() / 3, false);
    for (int pi : parts) for (uint32_t t : mp.tris[pi]) gone[t] = true;
    MeshData dm;
    dm.verts = src.verts;
    for (size_t t = 0; t < gone.size(); t++)
        if (!gone[t]) {
            dm.idx.insert(dm.idx.end(), {src.idx[t * 3], src.idx[t * 3 + 1], src.idx[t * 3 + 2]});
            dm.triMat.push_back(src.triMat.empty() ? 0u : src.triMat[t]);
        }
    auto append = [&](MeshData& a, const MeshData& b) {
        uint32_t base = (uint32_t)(a.verts.size() / 8);
        a.verts.insert(a.verts.end(), b.verts.begin(), b.verts.end());
        for (uint32_t i : b.idx) a.idx.push_back(base + i);
        a.triMat.insert(a.triMat.end(), b.triMat.begin(), b.triMat.end());
    };

    // fragments: own model (the wall's materials + a fresh broken-surface one)
    auto fs = std::make_unique<FragSet>();
    Model& fm = fs->model;
    fm.path = "<rubble>";
    fm.state = 1;
    fm.data.materials = cd.materials;
    fm.data.textures = cd.textures;
    struct Spawn { size_t mesh; Xf xf; V3 v, w; fracture::Poly poly; float r; };
    std::vector<Spawn> spawns;
    // a world-space cell coming loose: grit if tiny, else a rigid fragment; inBreach = blown out by the blow
    auto flyCell = [&](fracture::Poly cell, uint32_t outer, uint32_t inner, bool inBreach) {
        fracture::Vec c;
        float vol = fracture::volume(cell, &c);
        float r = fracture::radius(cell, c);
        V3 cw{c.x, c.y, c.z};
        V3 d = cw - p;
        // ~15% of the blow goes into throwing the breach's debris (about rb^2 * pi * 25 cm of wall)
        float vKick = inBreach ? std::min(sqrtf(0.3f * En / (2000.f * 3.14159f * rb * rb * 0.25f)), 25.f) : 0.f;
        V3 out = normalize(dir + d * (1.f / std::max(sqrtf(dot(d, d)), 0.1f)) * 0.6f);
        V3 vel = out * (vKick * (0.4f + rnd01()));
        if (r < 0.03f || vol < 1e-5f) {  // grit
            addDebris(DEB_DIRT, cw, vel, std::max(r, 0.015f));
            return;
        }
        fracture::translate(cell, {-c.x, -c.y, -c.z});
        MeshData mesh;
        fracture::toMesh(cell, mesh, outer, inner);
        fm.data.meshes.push_back(std::move(mesh));
        Xf px = xfIdentity();
        px.m[0][3] = cw.x; px.m[1][3] = cw.y; px.m[2][3] = cw.z;
        spawns.push_back({fm.data.meshes.size() - 1, px, vel, rndDir() * (3.f * rnd01()), std::move(cell), r});
    };
    for (int pi : parts) {
        uint32_t outer = src.triMat.empty() ? 0u : src.triMat[mp.tris[pi][0]];
        MaterialData in = sd.materials[outer];  // broken faces: the same stuff, paler and untextured
        for (int k = 0; k < 3; k++) in.color[k] = std::min(in.color[k] * 1.25f + 0.08f, 1.f);
        forEachTex(in, [](int) { return -1; });
        in.heightScale = 0.f;
        uint32_t inner = (uint32_t)cd.materials.size();
        cd.materials.push_back(in);
        fm.data.materials.push_back(in);
        std::vector<float> pts;
        for (uint32_t t : mp.tris[pi])
            for (int j = 0; j < 3; j++) {
                const float* v = &src.verts[src.idx[t * 3 + j] * 8];
                float o[3];
                xfPoint(hit.xf, v, o);
                pts.insert(pts.end(), {o[0], o[1], o[2]});
            }
        if (kind == BREAK_BRICK && brickBreach(hit, mp.box[pi], p, dir, rb, outer, inner, dm, flyCell)) continue;
        fracture::Poly whole = fracture::fromPoints(pts.data(), 3, (uint32_t)pts.size() / 3, 64);
        // world box of the part: cells touching its faces across the thin axis (floor, ceiling, wall ends) hold on
        float bmn[3], bmx[3];
        xfAabb(hit.xf, &mp.box[pi][0], &mp.box[pi][3], bmn, bmx);
        int thin = 0;
        for (int k = 1; k < 3; k++) if (bmx[k] - bmn[k] < bmx[thin] - bmn[thin]) thin = k;
        float size = std::max({bmx[0] - bmn[0], bmx[1] - bmn[1], bmx[2] - bmn[2]});
        int count = std::clamp((int)(8.f + 14.f * rb + 2.f * size), 8, 96);
        float spread = std::clamp(rb / std::max(size, 0.1f), 0.f, 1.f) * 0.6f;
        std::vector<fracture::Poly> cells;
        fracture::shatter(whole, {p.x, p.y, p.z}, {dir.x, dir.y, dir.z}, count, spread, frameIndex * 131u + (uint32_t)pi, cells);
        for (auto& cell : cells) {
            fracture::Vec c;
            float vol = fracture::volume(cell, &c);
            float r = fracture::radius(cell, c);
            V3 cw{c.x, c.y, c.z};
            V3 d = cw - p;
            bool held = false;
            for (auto& f : cell.faces)
                for (auto& v : f.pts)
                    for (int k = 0; k < 3; k++)
                        if (k != thin && (fabsf((&v.x)[k] - bmn[k]) < 0.01f || fabsf((&v.x)[k] - bmx[k]) < 0.01f)) held = true;
            bool flies = sqrtf(dot(d, d)) < rb || !held;
            if (!flies) {  // stays: into the damaged copy, in the mesh's local space
                fracture::Poly lp = cell;
                for (auto& f : lp.faces) for (auto& v : f.pts) v = toLocal(v);
                MeshData pm;
                fracture::toMesh(lp, pm, outer, inner);
                append(dm, pm);
                continue;
            }
            flyCell(std::move(cell), outer, inner, sqrtf(dot(d, d)) < rb);
        }
        for (int i = 0; i < (int)std::min(40.f * rb, 150.f); i++)  // dust and chips from the breach
            addDebris(DEB_DIRT, p + rndDir() * (rb * 0.6f), (dir * 2.f + rndDir()) * (1.f + 3.f * rnd01()), 0.02f + 0.05f * rnd01() * rnd01());
    }
    for (int k = 0; k < 3; k++) { dm.bmin[k] = 1e30f; dm.bmax[k] = -1e30f; }
    for (size_t i = 0; i < dm.verts.size(); i += 8)
        for (int k = 0; k < 3; k++) { dm.bmin[k] = std::min(dm.bmin[k], dm.verts[i + k]); dm.bmax[k] = std::max(dm.bmax[k], dm.verts[i + k]); }
    if (!dm.idx.empty())
        dm.shape = physics::meshShape(dm.verts.data(), 8, (uint32_t)(dm.verts.size() / 8), dm.idx.data(), (uint32_t)(dm.idx.size() / 3));
    else {  // nothing left: a degenerate sliver far below keeps the slot valid without being seen
        dm.verts = {0, -1000, 0, 0, 1, 0, 0, 0, 0.01f, -1000, 0, 0, 1, 0, 0, 0, 0, -1000, 0.01f, 0, 1, 0, 0, 0};
        dm.idx = {0, 2, 1};
        dm.triMat = {0};
        for (int k = 0; k < 3; k++) { dm.bmin[k] = dm.verts[k]; dm.bmax[k] = dm.verts[8 + k] + 0.01f; }
    }
    dm.strength = src.strength;
    dm.breakKind = src.breakKind;
    cd.meshes.push_back(std::move(dm));
    if (!uploadModel(*copy)) {
        if (cd.meshes[0].shape) physics::releaseShape(cd.meshes[0].shape);
        return false;
    }
    std::shared_ptr<Model> old = stream::overridePlaced(stream::placedKey(hit), copy);
    damagedModels.push_back(copy);
    if (old) {  // the previous damaged copy: its bodies are gone (cell re-placed), free it after this frame
        damagedModels.erase(std::remove(damagedModels.begin(), damagedModels.end(), old), damagedModels.end());
        retireNext.push_back([old]() {
            if (old->gpu) releaseModel(*old);
            if (old->data.meshes[0].shape) physics::releaseShape(old->data.meshes[0].shape);
        });
    }
    onStreamChanged();

    if (!spawns.empty()) {
        uploadModel(fm);
        FragSet* set = fs.get();
        fragSets.push_back(std::move(fs));
        for (auto& sp : spawns) {
            if (thrown.size() >= (size_t)MAX_THROWN) dropThrown(0);
            const MeshData& mesh = set->model.data.meshes[sp.mesh];
            Thrown f;
            f.sc = 1.f;
            f.radius = sp.r;
            f.frag = set;
            f.fragMesh = (uint32_t)sp.mesh;
            f.shape = physics::convexShape(mesh.verts.data(), 8, (uint32_t)mesh.verts.size() / 8);
            if (!f.shape) continue;
            f.body = physics::addDynamicPose(f.shape, sp.xf, &sp.v.x, &sp.w.x, 2000.f);
            f.xf = sp.xf;
            f.poly = std::make_shared<fracture::Poly>(std::move(sp.poly));
            f.cooldown = 0.2f;
            f.density = 2000.f;
            memcpy(f.v, &sp.v.x, 12);
            set->live++;
            thrown.push_back(std::move(f));
        }
        if (set->live == 0) {
            auto keep = std::shared_ptr<FragSet>(fragSets.back().release());
            fragSets.pop_back();
            retireNext.push_back([keep]() { if (keep->model.gpu) releaseModel(keep->model); });
        }
    }
    return true;
}

// Hard contact on land: the kinetic energy the last step took away, E = m/2 (|v0|^2 - |v1|^2), goes into what was
// hit. A tree snaps past a threshold (the rock then mostly carries on through), the ground gets a crater, and
// anything else (a building) throws rubble. The rock itself breaks if the blow's normal part is hard enough.
static void ensureRockShape() {
    for (int v = 0; v < ROCK_VARIANTS; v++) {
        const float* pts = rockPts.data() + (size_t)v * rockVerts * 6;
        if (!rockHull[v]) rockHull[v] = physics::convexShape(pts, 6, rockVerts);
        if (rockPoly[v].faces.empty()) rockPoly[v] = fracture::fromPoints(pts, 6, rockVerts);
    }
}

static void syncRockBodies() {
    if (thrown.empty() && rockBodies.empty()) return;
    PROF_FUNC();
    ensureRockShape();
    std::vector<V3> around;
    for (const Thrown& t : thrown) around.push_back({t.xf.m[0][3], t.xf.m[1][3], t.xf.m[2][3]});
    const float* xf = (const float*)rockXf.map;
    std::unordered_set<uint64_t> near;
    std::vector<void*> shapes;
    std::vector<Xf> xfs;
    std::vector<std::pair<uint64_t, float>> keys;
    std::vector<uint32_t> vars;
    for (uint32_t i = 0; i < numBreakableRocks && !around.empty(); i++) {
        const float* m = xf + i * 16;
        uint32_t var;
        memcpy(&var, m + 12, 4);
        var /= std::max(rockVerts, 1u);
        if (!rockHull[var]) continue;
        float x = m[3], z = m[11], d2 = 1e30f;
        for (const V3& a : around) d2 = std::min(d2, (x - a.x) * (x - a.x) + (z - a.z) * (z - a.z));
        if (d2 > 35.f * 35.f) continue;
        uint64_t k = rockKey(x, z);
        near.insert(k);
        if (d2 > 25.f * 25.f || rockBodies.count(k)) continue;
        Xf x3;
        memcpy(x3.m, m, 48);
        float sc = sqrtf(m[0] * m[0] + m[4] * m[4] + m[8] * m[8]);
        xfs.push_back(x3);
        shapes.push_back(rockHull[var]);
        keys.push_back({k, sc});
        vars.push_back(var);
    }
    std::vector<uint32_t> gone;
    for (auto it = rockBodies.begin(); it != rockBodies.end();)
        if (!near.count(it->first)) { gone.push_back(it->second.body); it = rockBodies.erase(it); } else ++it;
    if (!gone.empty()) physics::removeBodies(gone.data(), gone.size());
    if (shapes.empty()) return;
    std::vector<uint32_t> ids(shapes.size());
    physics::addStatic(shapes.data(), xfs.data(), shapes.size(), ids.data());
    for (size_t i = 0; i < keys.size(); i++) rockBodies[keys[i].first] = {ids[i], xfs[i], keys[i].second, vars[i]};
}

// a ground rock hit hard enough: off the placement, into the physics as a thrown rock carrying the blow, cracked
static bool knockRockLoose(V3 p, float reach, V3 J, float E) {
    uint64_t best = 0;
    float bd = 1e30f;
    for (auto& [k, rb] : rockBodies) {
        float dx = rb.xf.m[0][3] - p.x, dy = rb.xf.m[1][3] - p.y, dz = rb.xf.m[2][3] - p.z, d2 = dx * dx + dy * dy + dz * dz;
        float lim = reach + 1.1f * rb.sc;
        if (d2 < lim * lim && d2 < bd) { bd = d2; best = k; }
    }
    if (bd == 1e30f) return false;
    RockBody rb = rockBodies[best];
    float mass = 2600.f * 2.6f * rb.sc * rb.sc * rb.sc;
    if (E < 0.5f * fractureThreshold(0.85f * rb.sc)) return false;  // it holds
    physics::removeBodies(&rb.body, 1);
    rockBodies.erase(best);
    brokenRocks.insert(best);
    placeDirty = true;
    if (thrown.size() >= (size_t)MAX_THROWN) dropThrown(0);
    V3 pos{rb.xf.m[0][3], rb.xf.m[1][3] + 0.05f, rb.xf.m[2][3]};
    V3 v = J * (0.6f / mass);
    float vl = sqrtf(dot(v, v));
    if (vl > 25.f) v = v * (25.f / vl);
    float w[3] = {rnd01() * 4.f - 2.f, rnd01() * 4.f - 2.f, rnd01() * 4.f - 2.f};
    Thrown t;
    t.sc = rb.sc;
    t.radius = 0.85f * rb.sc;
    t.rockVar = rb.var;
    t.body = physics::addDynamic(rockHull[rb.var], rb.sc, &pos.x, &v.x, w, 2600.f);
    t.xf = rb.xf;
    t.poly = std::make_shared<fracture::Poly>(rockPoly[rb.var]);
    for (auto& fc : t.poly->faces) for (auto& q : fc.pts) q = {q.x * rb.sc, q.y * rb.sc, q.z * rb.sc};
    memcpy(t.v, &v.x, 12);
    t.cooldown = 0.2f;
    thrown.push_back(t);
    V3 dir = dot(J, J) > 1e-6f ? normalize(J) : V3{0, -1, 0};
    if (pendingFractures.size() < 32) pendingFractures.push_back({t.body, p, dir, E * 0.7f});
    return true;
}

static void landImpact(Thrown& t, const float v0[3], const float v1[3], float mass, float dt) {
    float s0 = v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2], s1 = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2];
    float E = 0.5f * mass * std::max(s0 - s1, 0.f);
    // contact normal ~ the direction of the impulse (velocity change minus gravity's share)
    V3 dv{v1[0] - v0[0], v1[1] - v0[1] + 9.8f * dt, v1[2] - v0[2]};
    V3 n = dot(dv, dv) > 1e-6f ? normalize(dv) : V3{0, 1, 0};
    float vn = std::max(-(v0[0] * n.x + v0[1] * n.y + v0[2] * n.z), 0.f);
    float En = 0.5f * mass * vn * vn;
    V3 p{t.xf.m[0][3], t.xf.m[1][3], t.xf.m[2][3]};
    if (t.poly && En > fractureThreshold(t.radius) && pendingFractures.size() < 32)
        pendingFractures.push_back({t.body, p - n * (t.radius * 0.8f), n * -1.f, En});
    // where it touched: a rock's surface toward the blow, a wood piece's furthest point that way
    V3 pc = p - n * (t.radius * 0.8f);
    WoodFrame wf;
    V3 wl{0, 0, 0};
    if (t.woodPart) {
        wf = thrownFrame(t);
        V3 ln = woodRotT(wf, n);
        trees::woodSupport(*t.woodPart, &ln.x, &wl.x);
        pc = woodToWorld(wf, wl);
    }
    // a procedural tree? Its limbs take the blow as beams (any impact, the stress decides)
    if (E > 5.f && !treeList.empty()) {
        V3 vel{v0[0], v0[1], v0[2]};
        if (strikeTrees(pc, vel, mass, t.radius * 0.3f + 0.25f)) {
            t.cooldown = 0.15f;
            for (auto& q : pendingWood)  // broke through: the impactor keeps what the snap didn't take
                if (q.body == UINT32_MAX && !q.hit.cuts.empty() && dot(q.at - pc, q.at - pc) < 1e-6f) {
                    V3 nv = vel - q.J * (1.f / mass);
                    physics::setVelocity(t.body, &nv.x);
                }
            return;
        }
    }
    // a loose piece of tree landing: its own joints may give
    if (t.woodPart && pendingWood.size() < 16 && dot(dv, dv) < 50.f * 50.f) {  // (bigger: a depenetration kick, not a landing)
        V3 Jw = dv * mass, Jl = woodRotT(wf, Jw);
        trees::WoodHit h = trees::woodLanding(*t.woodPart, wf.s, &wl.x, &Jl.x);
        if (treeLog) fprintf(stderr, "wood landed: %.0f kg, dv %.1f m/s: stress %.2f, %zu cut, %d leaves\n", mass, sqrtf(dot(dv, dv)), h.ratio, h.cuts.size(), h.lost);
        bool queued = false;
        for (auto& q : pendingWood) queued |= q.body == t.body;
        if (!queued && (!h.cuts.empty() || h.lost >= 8)) {
            t.cooldown = 0.15f;
            pendingWood.push_back({TreeInst{{0, 0, 0}, 0, 1, (uint32_t)t.woodPart->v, (int)(t.body & 0xffff), (int)(t.body >> 16)}, t.body, std::move(h), pc, Jw});
        }
    }
    if (E < 3e3f) return;
    t.cooldown = 0.15f;
    float r = t.radius;
    // another rock (a scattered ground rock) took the blow?
    if (!rockBodies.empty()) {
        V3 J = V3{v0[0] - v1[0], v0[1] - v1[1], v0[2] - v1[2]} * mass;
        if (knockRockLoose(p, r, J, E)) return;
    }
    // an imported tree's trunk within reach of the rock?
    const TreeInst* best = nullptr;
    float bd = 1e9f;
    for (const TreeInst& tr : treeList) {
        if (tr.v < (uint32_t)trees::VARIANTS) continue;  // procedural trees: strikeTrees above
        float dx = tr.p.x - p.x, dz = tr.p.z - p.z, d2 = dx * dx + dz * dz;
        if (d2 < bd && d2 < (r + 0.9f) * (r + 0.9f) && p.y < tr.p.y + 6.f * tr.scale) { bd = d2; best = &tr; }
    }
    if (best) {
        // ~6 kJ snaps a typical (scale 1, ~25 cm) trunk: a default throw (0.4 m rock, ~430 kg at 12 m/s, ~31 kJ)
        // breaks one even after a glancing hit, a fist-sized stone doesn't
        float Ebreak = 6e3f * best->scale * best->scale * best->scale;
        if (E > Ebreak) {
            V3 J = V3{v0[0] - v1[0], v0[1] - v1[1], v0[2] - v1[2]} * mass;
            TreeInst tr = *best;
            knockTree(tr, J, p, E / Ebreak);
            float keep = std::sqrt(std::max(1.f - Ebreak / E, 0.f));  // the rock keeps what breaking didn't use
            float nv[3] = {v0[0] * keep, v0[1] * keep, v0[2] * keep};
            physics::setVelocity(t.body, nv);
        }
        return;
    }
    float ground = heightFn(p.x, p.z);
    if (p.y - r * 0.7f < ground + 0.3f) {
        if (riverDry(p.x, p.z, 0.f) > 0.f && !t.wood) groundImpact({p.x, ground, p.z}, E);
        return;
    }
    float Ew = std::max(En, 0.5f * E);
    if (Ew > 0.02f * WALL_GC * 3.14159f * 0.15f * 0.15f && pendingWallBreaks.size() < 8) {  // maybe a breach (breakStatic)
        pendingWallBreaks.push_back({t.body, p, normalize(V3{v0[0], v0[1], v0[2]}), V3{v0[0], v0[1], v0[2]}, Ew, r + 0.4f});
        return;
    }
    int nd = std::min((int)(E / 4e3f), 120);  // building or other static mesh too strong to break: rubble and dust
    for (int i = 0; i < nd; i++)
        addDebris(DEB_DIRT, p + rndDir() * r, rndDir() * (1.f + 4.f * rnd01()) + V3{0, 1.5f, 0}, 0.03f + 0.08f * rnd01());
}

// Water forces on thrown rocks before the physics step: buoyancy from the submerged part (body ~ a sphere of its
// bounding radius), quadratic drag against the current (so it gets carried and tumbled), planing lift for a
// fast flat skim, extra angular damping in water. Every touch of the surface is a splash event; a rock moving
// through the surface leaves a trail of small ones (wake). Hard landings go to landImpact.
static void updateThrown(float dt) {
    for (size_t i = 0; i < thrown.size();) {
        Thrown& t = thrown[i];
        float v[3], w[3], mass;
        if (!physics::bodyState(t.body, t.sc, t.xf, v, w, mass) || t.xf.m[1][3] < -200.f) {
            dropThrown(i);
            continue;
        }
        // settled debris retires: chips after a few seconds at rest, bigger pieces once out of the way. Every piece
        // is its own body and BLAS, so a few hundred of them cost GI / BVH time for nothing.
        float sp2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        t.rest = sp2 < 0.04f ? t.rest + dt : 0.f;
        if (t.frag || t.wood) {
            float cdx = t.xf.m[0][3] - cam.pos.x, cdz = t.xf.m[2][3] - cam.pos.z, cd2 = cdx * cdx + cdz * cdz;
            if ((t.radius < 0.15f && t.rest > 3.f) || (t.radius < 0.4f && t.rest > 15.f && cd2 > 30.f * 30.f) ||
                (t.rest > 60.f && cd2 > 60.f * 60.f)) {
                dropThrown(i);
                continue;
            }
        }
        i++;
        t.age += dt;
        float v0[3];
        memcpy(v0, t.v, 12);
        memcpy(t.v, v, 12);
        t.cooldown -= dt;
        float x = t.xf.m[0][3], y = t.xf.m[1][3], z = t.xf.m[2][3], r = t.radius;
        if (t.woodPart) {  // a broken limb tumbles no faster than air drag on its leaves and twigs lets it (~2 turns/s)
            float w2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
            if (w2 > 12.f * 12.f) { float k = 12.f / sqrtf(w2); for (float& c : w) c *= k; physics::setAngularVelocity(t.body, w); }
        }
        if (sp2 > 4.f && !t.woodPart && t.radius > 0.03f) {  // a rock (or chunk of one) moving: through any thin limbs?
            V3 p1{x, y, z}, vv{v[0], v[1], v[2]};
            sweepThinLimbs(t, p1 - vv * dt, p1, vv, mass);
        }
        float dv[3] = {v[0] - v0[0], v[1] - v0[1], v[2] - v0[2]};
        bool hardHit = dv[0] * dv[0] + (dv[1] + 9.8f * dt) * (dv[1] + 9.8f * dt) + dv[2] * dv[2] > 9.f;
        float level = waterLevelAt(x, z);
        bool overWater = level > -1e8f;
        RiverInfo ri = riverInfo(x, z);
        if (!overWater || y - r > level) {
            if (y - r > level + 0.05f) t.wet = false;
            if (hardHit && t.cooldown <= 0.f) landImpact(t, v0, v, mass, dt);
            continue;
        }
        float h = std::clamp(level - (y - r), 0.f, 2.f * r), frac = h * h * (3.f * r - h) / (4.f * r * r * r);  // submerged volume fraction
        float vh = sqrtf(v[0] * v[0] + v[2] * v[2]);
        float sz = r / 0.85f;
        if (!t.wet && frac > 0.02f) {  // splash-down (again on every skip)
            t.wet = true;
            waterImpact({x, level, z}, sz, v0);
        } else if (t.wet && frac < 0.95f && vh > 1.f && (t.trail -= dt) <= 0.f) {  // cutting through the surface
            t.trail = 0.07f;
            waterImpact({x, level, z}, sz * 0.5f, v, std::min(0.12f * vh, 0.5f));
        }
        if (hardHit && t.cooldown <= 0.f && y - r < heightFn(x, z) + 0.3f) landImpact(t, v0, v, mass, dt);  // hit the bed
        float cur[2];
        riverVelocity(x, z, S.flow * (1.f + 1.6f * riverSwell / std::max(gRiver.depth, 0.3f)), cur);
        if (fabsf(ri.d) > ri.w + 1.f) cur[0] = cur[1] = 0.f;  // a crater pond: still water
        // current is fastest at the surface, ~half of it near the bed
        float depthFrac = std::clamp((level - y) / std::max(ri.depth, 0.3f), 0.f, 1.f);
        float k = 1.f - 0.5f * depthFrac;
        float rel[3] = {v[0] - cur[0] * k, v[1], v[2] - cur[1] * k};
        float rl = sqrtf(rel[0] * rel[0] + rel[1] * rel[1] + rel[2] * rel[2]);
        const float rhoW = 1000.f, Cd = 1.0f, g = 9.8f;
        float A = 3.14159f * r * r * std::min(frac * 1.5f, 1.f);
        float vol = mass / t.density;
        float F[3], T[3];
        for (int c = 0; c < 3; c++) F[c] = -0.5f * rhoW * Cd * A * rl * rel[c];
        F[1] += rhoW * g * vol * frac;
        // planing lift (stone skipping, Clanet et al. 2004): a shallow, fast contact pushes the rock back up
        if (frac < 0.5f && vh > 4.f && v[1] < 2.f) F[1] += std::min(0.5f * rhoW * 0.35f * 3.14159f * r * r * std::min(frac * 4.f, 1.f) * vh * vh, 4.f * mass * g);
        float ang = 0.4f * mass * r * r * 6.f * frac;  // rotational drag ~ I * rate
        for (int c = 0; c < 3; c++) T[c] = -ang * w[c];
        physics::addForce(t.body, F, T);
    }
}

// Moving things into the TLAS after the placed instances (every frame): thrown rocks (rock-mesh ones also into
// the raster rock list), fragments and tree pieces (their own meshes), debris particles, stumps.
static void writeThrown() {
    numThrown = (uint32_t)thrown.size();
    numThrownRock = 0;
    auto* inst = (VkAccelerationStructureInstanceKHR*)instBuf.map;
    float* xf = (float*)rockXf.map;
    uint32_t base = INST_MODELS + numModelInst + numRocks + numTrees + numPlants;
    for (uint32_t i = 0; i < numThrown; i++) {
        const Thrown& t = thrown[i];
        auto* g = t.frag ? (GpuModel*)t.frag->model.gpu : nullptr;
        if (t.frag && g) {
            const GpuMesh& gm = g->meshes[t.fragMesh];
            setInstance(inst[base + i], 3 + gm.slot, gm.blasAddr, t.xf.m);
        } else {
            setRockInstance(inst[base + i], t.rockVar, t.xf.m);
            if (!t.frag) {
                writeRockXf(xf, numRocks + numThrownRock, t.xf.m, t.hasPrev ? t.prevXf.m : t.xf.m, t.rockVar);
                numThrownRock++;
            }
        }
    }
    numSprayInst = numFallen = 0;
    auto* pg = (GpuModel*)plantModel.gpu;
    if (pg)
        for (const Droplet& d : spray) {
            if (d.delay > 0.f) continue;
            const GpuMesh& dm = pg->meshes[debrisMesh[d.kind]];
            float m[3][4];
            dropletXf(d, m);
            setInstance(inst[base + numThrown + numSprayInst++], 3 + dm.slot, dm.blasAddr, m);
        }
    for (const Stump& st : stumps) {
        auto* sg = (GpuModel*)st.frag->model.gpu;
        if (!sg) continue;
        const GpuMesh& sm = sg->meshes[st.mesh];
        setInstance(inst[base + numThrown + numSprayInst + numFallen++], 3 + sm.slot, sm.blasAddr, st.xf.m);
    }
}

static void updateCamera(float dt) {
    ImGuiIO& io = ImGui::GetIO();
    static double lx, ly;
    static bool looking = false;
    double mx, my;
    glfwGetCursorPos(window, &mx, &my);
    bool rmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    if (rmb && (looking || !io.WantCaptureMouse)) {
        if (looking) {
            cam.yaw -= (float)(mx - lx) * 0.003f;
            cam.pitch = std::clamp(cam.pitch - (float)(my - ly) * 0.003f, -1.5f, 1.5f);
        }
        looking = true;
    } else looking = false;
    lx = mx; ly = my;
    // throw: G, left click (outside the UI) or right trigger, on the press edge
    static bool throwHeld = false;
    bool th = glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS ||
              (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && !io.WantCaptureMouse && !editor::wantsMouse()) || input::pad().rt > 0.6f;
    if (th && !throwHeld) throwRock();
    throwHeld = th;
    // P: primary visibility raster G-buffer <-> path traced camera rays (A/B comparison); lighting is traced either way
    static bool pHeld = false;
    bool pk = glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS && !io.WantCaptureKeyboard;
    if (pk && !pHeld) {
        S.rasterPrimary = !S.rasterPrimary;
        resetHistory = true;
        glfwSetWindowTitle(window, S.rasterPrimary ? "ptx - primary: raster G-buffer (P to toggle)" : "ptx - primary: path traced (P to toggle)");
    }
    pHeld = pk;
    // I: global illumination on / off (direct light only), to see what the bounced light adds
    static bool iHeld = false;
    bool ik = glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS && !io.WantCaptureKeyboard;
    if (ik && !iHeld) {
        S.giEnabled = !S.giEnabled;
        resetHistory = true;
        glfwSetWindowTitle(window, S.giEnabled ? "ptx - GI on (I to toggle)" : "ptx - GI off: direct light only (I to toggle)");
    }
    iHeld = ik;
    // gamepad look: stick (squared response for fine aim) + gyro (Steam Controller / DualSense)
    const input::Pad& gp = input::pad();
    if (gp.connected) {
        cam.yaw -= gp.look[0] * fabsf(gp.look[0]) * 2.8f * dt;
        cam.pitch += gp.look[1] * fabsf(gp.look[1]) * 2.0f * dt;
        if (gp.hasGyro && gyroAim) {
            cam.yaw += gp.gyro[1] * gyroSens * dt;
            cam.pitch += gp.gyro[0] * gyroSens * dt;
        }
        cam.pitch = std::clamp(cam.pitch, -1.5f, 1.5f);
    }
    if (io.WantCaptureKeyboard && !gp.connected) return;
    static bool fPrev = false;
    bool fKey = (!io.WantCaptureKeyboard && glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS) || gp.togglePressed;
    if (fKey && !fPrev) {
        walkMode = !walkMode;
        if (walkMode) {
            V3 feet = cam.pos - V3{0, 1.65f, 0};
            feet.y = std::max(feet.y, heightFn(feet.x, feet.z) + 0.05f);
            physics::spawnPlayer(&feet.x);
            playerFeet = feet;
            walkReset = true;
        }
    }
    fPrev = fKey;
    if (walkMode) {
        // first person: horizontal wish velocity, physics resolves collisions, stairs and gravity
        float spd = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) || gp.sprint) ? 7.f : 3.5f;
        V3 fw = normalize(V3{sinf(cam.yaw), 0, cosf(cam.yaw)}), rt = cam.right();
        V3 w{0, 0, 0};
        if (glfwGetKey(window, GLFW_KEY_W)) w = w + fw;
        if (glfwGetKey(window, GLFW_KEY_S)) w = w - fw;
        if (glfwGetKey(window, GLFW_KEY_D)) w = w + rt;
        if (glfwGetKey(window, GLFW_KEY_A)) w = w - rt;
        if (dot(w, w) > 0) w = normalize(w) * spd;
        else w = (fw * gp.move[1] + rt * gp.move[0]) * spd;  // analog: partial tilt walks slower
        float wish[2] = {w.x, w.z};
        bool jump = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS || gp.jump;
        // fixed 120 Hz character step, rendered position interpolated between the last two steps:
        // a variable dt makes the controller's ground snapping / velocity resolve differently every frame
        const float step = 1.f / 120.f;
        static float acc = 0;
        static V3 feetPrev = playerFeet, feetCur = playerFeet, eye = playerFeet;
        if (walkReset) { acc = 0; feetPrev = feetCur = eye = playerFeet; walkReset = false; }
        acc += dt;
        while (acc >= step) {
            bool wasGrounded = grounded;
            feetPrev = feetCur;
            physics::movePlayer(step, wish, jump, &feetCur.x, grounded);
            if (grounded && !wasGrounded) input::rumble(0.4f, 0.1f, 80);  // landing thump
            acc -= step;
            // clipped through the ground: once 50 m under the terrain, put the player back on top of it
            float gy = heightFn(feetCur.x, feetCur.z);
            if (feetCur.y < gy - 50.f) {
                feetCur.y = gy + 1.f;
                physics::spawnPlayer(&feetCur.x);
                feetPrev = eye = feetCur;
                grounded = false;
            }
        }
        float a = acc / step;
        playerFeet = feetPrev + (feetCur - feetPrev) * a;
        // stair step-up / stick-to-floor teleport the capsule vertically: ease the eye height over ~60 ms
        eye.x = playerFeet.x; eye.z = playerFeet.z;
        eye.y += (playerFeet.y - eye.y) * (1.f - expf(-dt / 0.06f));
        if (fabsf(playerFeet.y - eye.y) > 1.f) eye.y = playerFeet.y;  // falls / respawn: don't lag behind
        cam.pos = eye + V3{0, 1.65f, 0};
        return;
    }
    float sp = ((glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) || gp.sprint) ? 12.f : 4.f) * flySpeed * dt;
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) sp *= 8.f;  // turbo: cross the map quickly
    V3 f = cam.fwd(), r = cam.right();
    cam.pos = cam.pos + f * (gp.move[1] * sp) + r * (gp.move[0] * sp);
    if (gp.up) cam.pos.y += sp;
    if (gp.down) cam.pos.y -= sp;
    if (glfwGetKey(window, GLFW_KEY_W)) cam.pos = cam.pos + f * sp;
    if (glfwGetKey(window, GLFW_KEY_S)) cam.pos = cam.pos - f * sp;
    if (glfwGetKey(window, GLFW_KEY_D)) cam.pos = cam.pos + r * sp;
    if (glfwGetKey(window, GLFW_KEY_A)) cam.pos = cam.pos - r * sp;
    if (glfwGetKey(window, GLFW_KEY_E)) cam.pos.y += sp;
    if (glfwGetKey(window, GLFW_KEY_Q)) cam.pos.y -= sp;
    // the flying camera stays above the ground (craters, sculpted hills): no seeing through the terrain from inside
    cam.pos.y = std::max(cam.pos.y, heightFn(cam.pos.x, cam.pos.z) + 0.3f);
}

// ---------------------------------------------------------------- audio: listener, ground patch, footsteps, orb hums
static void updateAudio(float dt) {
    PROF_FUNC();
    V3 f = cam.fwd(), u = cam.up();
    audio::setListener(&cam.pos.x, &f.x, &u.x);

    // acoustic ground: 96 m patch at 3 m spacing, rebuilt when the listener moves 16 m
    static float gx = 1e9f, gz = 1e9f;
    if (fabsf(cam.pos.x - gx) > 16.f || fabsf(cam.pos.z - gz) > 16.f) {
        gx = floorf(cam.pos.x / 3.f) * 3.f;
        gz = floorf(cam.pos.z / 3.f) * 3.f;
        const int N = 33;
        std::vector<float> v;
        std::vector<uint32_t> idx;
        for (int j = 0; j < N; j++)
            for (int i = 0; i < N; i++) {
                float x = gx + (i - N / 2) * 3.f, z = gz + (j - N / 2) * 3.f;
                v.insert(v.end(), {x, heightFn(x, z), z});
            }
        for (int j = 0; j + 1 < N; j++)
            for (int i = 0; i + 1 < N; i++) {
                uint32_t a = j * N + i;
                idx.insert(idx.end(), {a, a + N, a + 1, a + 1, a + N, a + N + 1});
            }
        audio::setTerrain(v.data(), (uint32_t)v.size() / 3, idx.data(), (uint32_t)idx.size() / 3);
    }

    // footsteps every stride while walking on the ground, a heavier one on landing
    static std::mt19937 rng(3);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    static V3 lastFeet = playerFeet;
    static float stride = 0;
    static bool wasGrounded = true;
    if (walkMode && !bench.running) {
        V3 d = playerFeet - lastFeet;
        float moved = sqrtf(d.x * d.x + d.z * d.z);
        if (grounded && moved < 1.f) stride += moved;
        float len = moved / std::max(dt, 1e-4f) > 5.f ? 1.0f : 0.75f;
        if (grounded && stride > len) {
            stride = 0;
            audio::emitterPlay(audio::SND_FOOTSTEP0 + (uint32_t)(U(rng) * 3.99f), &playerFeet.x, 0.5f + 0.2f * U(rng), false, 0.9f + 0.2f * U(rng));
        }
        if (grounded && !wasGrounded) audio::emitterPlay(audio::SND_LAND, &playerFeet.x, 0.9f, false, 0.95f + 0.1f * U(rng));
    }
    lastFeet = playerFeet;
    wasGrounded = grounded;

    // light orbs hum: the nearest few get an emitter (positions from this frame's light buffer)
    constexpr int HUMS = 8;
    static uint32_t humEmitter[MAX_ORBS];
    static bool humInit = false;
    if (!humInit) { std::fill_n(humEmitter, MAX_ORBS, UINT32_MAX); humInit = true; }
    const float* L = (const float*)lightBuf.map;
    int orbs = std::min(S.lightCount, MAX_ORBS);
    std::vector<std::pair<float, int>> near;
    for (int i = 0; i < orbs; i++) {
        V3 p{L[i * 8], L[i * 8 + 1], L[i * 8 + 2]}, d = p - cam.pos;
        near.push_back({dot(d, d), i});
    }
    int keep = std::min(HUMS, (int)near.size());
    std::partial_sort(near.begin(), near.begin() + keep, near.end());
    bool want[MAX_ORBS] = {};
    for (int k = 0; k < keep; k++) want[near[k].second] = S.lightIntensity > 0.f;
    for (int i = 0; i < MAX_ORBS; i++) {
        uint32_t& e = humEmitter[i];
        if (e != UINT32_MAX && !audio::emitterAlive(e)) e = UINT32_MAX;
        if (!want[i]) { if (e != UINT32_MAX) { audio::emitterStop(e); e = UINT32_MAX; } continue; }
        const float* p = L + i * 8;
        float gain = 0.25f * lightBase[i].inten * std::min(S.lightIntensity, 4.f) * orbFade[i];
        if (e == UINT32_MAX) e = audio::emitterPlay(audio::SND_HUM, p, gain, true, 0.8f + 0.45f * fmodf(lightBase[i].phase, 1.f));
        else { audio::emitterSetPos(e, p); audio::emitterSetGain(e, gain); }
    }
    audio::setAmbience(S.windStrength * 0.35f * (0.4f + S.windSpeed / 6.f));
}

static int pendingRes = -1;

// --shot <file.ppm>: save the internal-res output after a warm-up and exit (for testing)
static std::string newShotPath() {
    std::error_code ec;
    std::filesystem::create_directories("screenshots", ec);
    char name[96];
    time_t t = time(nullptr);
    strftime(name, sizeof name, "screenshots/ptx_%Y%m%d_%H%M%S", localtime(&t));
    std::string base = name, p = base + ".png";
    for (int i = 2; std::filesystem::exists(p, ec); i++) p = base + "_" + std::to_string(i) + ".png";
    return std::filesystem::absolute(p).string();
}
// ---------------------------------------------------------------- movement log + replay
// Every session records logs/ptx_<start>.movelog: the starting scene (settings, river, craters), then one line per
// frame (frame, time, step, camera position / yaw / pitch, walking) and events (rocks thrown, screenshots). A
// screenshot's .txt names the log and frame it was taken at. --replay <log> drives the camera from it frame by frame
// with the recorded steps and re-throws the rocks; --replay-shots retakes its screenshots.
struct MoveFrame { uint32_t f; double t; float dt, x, y, z, yaw, pitch; int walk; };
struct MoveEvent { uint32_t f; std::string kind, arg; float a = 0, b = 0; };
static FILE* moveLog = nullptr;
static std::string moveLogPath;
static uint32_t moveLogFrames = 0;
static std::vector<MoveFrame> replayFrames;
static std::vector<MoveEvent> replayEvents;
static size_t replayAt = 0, replayEvAt = 0;
static bool replaying = false, replayShots = false, replayExit = false;
static void writeSceneHeader(FILE* f);
static void moveLogEvent(const char* kind, const std::string& arg) {
    if (moveLog) fprintf(moveLog, "e %u %s %s\n", moveLogFrames, kind, arg.c_str());
}
static void openMoveLog() {
    std::error_code ec;
    std::filesystem::create_directories("logs", ec);
    char name[96];
    time_t t = time(nullptr);
    strftime(name, sizeof name, "logs/ptx_%Y%m%d_%H%M%S.movelog", localtime(&t));
    moveLogPath = std::filesystem::absolute(name).string();
    moveLog = fopen(name, "w");
    if (!moveLog) return;
    fprintf(moveLog, "# ptx movement log (replay: ./build/ptx --replay %s)\n", moveLogPath.c_str());
    writeSceneHeader(moveLog);
    fprintf(moveLog, "# f <frame> <time> <dt> <x y z> <yaw pitch deg> <walk>;  e <frame> <event> <args>\n");
}
static void moveLogFrame(float dt) {
    if (!moveLog || replaying) return;
    fprintf(moveLog, "f %u %.4f %.5f %.4f %.4f %.4f %.3f %.3f %d\n", moveLogFrames, simTime, dt, cam.pos.x, cam.pos.y, cam.pos.z,
            cam.yaw * 180.f / 3.14159265f, cam.pitch * 180.f / 3.14159265f, walkMode ? 1 : 0);
    moveLogFrames++;
    if (moveLogFrames % 300 == 0) fflush(moveLog);  // a crash loses at most a few seconds
}
static bool loadReplay(const std::string& path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "--replay: can't read %s\n", path.c_str()); return false; }
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ls(line);
        std::string k;
        ls >> k;
        if (k == "f") {
            MoveFrame m;
            ls >> m.f >> m.t >> m.dt >> m.x >> m.y >> m.z >> m.yaw >> m.pitch >> m.walk;
            replayFrames.push_back(m);
        } else if (k == "e") {
            MoveEvent e;
            ls >> e.f >> e.kind;
            std::getline(ls >> std::ws, e.arg);
            std::istringstream as(e.arg);
            as >> e.a >> e.b;
            replayEvents.push_back(e);
        }
    }
    // the starting scene (settings, river, craters) is in the header: same format as a screenshot's .txt
    extern bool loadShotInfoPath(const std::string&);
    loadShotInfoPath(path);
    replaying = !replayFrames.empty();
    printf("replay: %zu frames, %zu events from %s\n", replayFrames.size(), replayEvents.size(), path.c_str());
    return replaying;
}

// The scene beside each screenshot (<shot>.txt): camera, time, river, craters and every setting, so
// ./build/ptx --from-shot <shot.png|txt> reopens exactly this view.
static void writeShotInfo(const std::string& png) {
    std::string txt = png.substr(0, png.size() - 4) + ".txt";
    FILE* f = fopen(txt.c_str(), "w");
    if (!f) return;
    fprintf(f, "# ptx scene for %s\n# reopen: ./build/ptx --from-shot %s\n", png.c_str(), txt.c_str());
    if (!moveLogPath.empty())
        fprintf(f, "# movement log: %s at frame %u (replay up to here: ./build/ptx --replay %s)\nmovelog %s %u\n", moveLogPath.c_str(),
                moveLogFrames, moveLogPath.c_str(), moveLogPath.c_str(), moveLogFrames);
    writeSceneHeader(f);
    fclose(f);
    moveLogEvent("shot", png);
}
static void writeSceneHeader(FILE* f) {
    float yawD = cam.yaw * 180.f / 3.14159265f, pitchD = cam.pitch * 180.f / 3.14159265f;
    fprintf(f, "# GPU: %s, render %ux%u -> output %ux%u, %.1f ms GPU, %.1f ms CPU frame\n", gpuName, W, H, OW, OH, gpuEmaTotal, cpuFrameMs);
    fprintf(f, "pos %.4f %.4f %.4f\nlook %.4f %.4f\nfov %.3f\ntime %.4f\nwalk %d\n", cam.pos.x, cam.pos.y, cam.pos.z, yawD, pitchD,
            cam.fovDeg, simTime, walkMode ? 1 : 0);
    fprintf(f, "river %g %g %g %g %g\n", gRiver.width, gRiver.depth, gRiver.level, gRiver.slope, gRiver.bumps);
    for (int i = 0; i < std::min(gCraterCount.load(), MAX_CRATERS); i++)
        fprintf(f, "crater %g %g %g %g\n", gCraters[i].x, gCraters[i].z, gCraters[i].r, gCraters[i].depth);
    fprintf(f, "settings %zu ", sizeof(Settings));
    const uint8_t* b = (const uint8_t*)&S;
    for (size_t i = 0; i < sizeof(Settings); i++) fprintf(f, "%02x", b[i]);
    fprintf(f, "\n# readable: sun %.1f / %.1f deg, precipitation %.2f, fog %g, clouds %.2f, view %.0f m, trees %.2f, ground mode %d\n",
            S.sunElev, S.sunAzim, S.rain, S.fogDensity, S.cloudCoverage, S.viewDist, S.treeDensity, S.groundMode);
    fprintf(f, "# command: ./build/ptx --pos %.3f %.3f %.3f --look %.2f %.2f\n", cam.pos.x, cam.pos.y, cam.pos.z, yawD, pitchD);
}
// --from-shot: apply a saved scene (before the world starts)
bool loadShotInfoPath(const std::string& path);
static bool loadShotInfo(std::string path) { return loadShotInfoPath(path); }
bool loadShotInfoPath(const std::string& p0) {
    std::string path = p0;
    if (path.size() > 4 && path.compare(path.size() - 4, 4, ".png") == 0) path = path.substr(0, path.size() - 4) + ".txt";
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "--from-shot: can't read %s\n", path.c_str()); return false; }
    std::string line;
    int nc = 0;
    while (std::getline(f, line)) {
        std::istringstream ls(line);
        std::string k;
        ls >> k;
        if (k == "pos") ls >> cam.pos.x >> cam.pos.y >> cam.pos.z;
        else if (k == "look") { float y, p; ls >> y >> p; cam.yaw = y * 3.14159265f / 180.f; cam.pitch = p * 3.14159265f / 180.f; }
        else if (k == "fov") ls >> cam.fovDeg;
        else if (k == "time") ls >> simTime;
        else if (k == "walk") { int w; ls >> w; walkMode = w != 0; }
        else if (k == "river") ls >> gRiver.width >> gRiver.depth >> gRiver.level >> gRiver.slope >> gRiver.bumps;
        else if (k == "crater" && nc < MAX_CRATERS) { Crater c; ls >> c.x >> c.z >> c.r >> c.depth; gCraters[nc++] = c; }
        else if (k == "settings") {
            size_t n;
            std::string hex;
            ls >> n >> hex;
            if (n != sizeof(Settings) || hex.size() != 2 * n) { fprintf(stderr, "--from-shot: settings from another build, camera only\n"); continue; }
            uint8_t* b = (uint8_t*)&S;
            for (size_t i = 0; i < n; i++) b[i] = (uint8_t)strtoul(hex.substr(2 * i, 2).c_str(), nullptr, 16);
        }
    }
    gCraterCount.store(nc);
    printf("scene from %s\n", path.c_str());
    return true;
}
static void shotDone(const std::string& p, bool ok) {
    if (ok) writeShotInfo(p);
    shotMsg = ok ? "saved " + p + " + .txt scene (path copied: paste it to Claude)" : "screenshot failed: " + p;
    shotMsgT = glfwGetTime();
    if (ok) glfwSetClipboardString(window, p.c_str());
    printf("%s\n", shotMsg.c_str());
}
// rgba (or bgra) rows -> PNG, alpha forced opaque
static bool writePng(const std::string& p, const uint8_t* px, uint32_t w, uint32_t h, bool bgr) {
    std::vector<uint8_t> rgb((size_t)w * h * 3);
    for (size_t i = 0; i < (size_t)w * h; i++) {
        rgb[i * 3 + 0] = px[i * 4 + (bgr ? 2 : 0)];
        rgb[i * 3 + 1] = px[i * 4 + 1];
        rgb[i * 3 + 2] = px[i * 4 + (bgr ? 0 : 2)];
    }
    return stbi_write_png(p.c_str(), (int)w, (int)h, 3, rgb.data(), (int)w * 3) != 0;
}
// the scene without UI (the output image), as PNG
static void saveScenePng() {
    vkDeviceWaitIdle(dev);
    Buffer st = mkBuf((VkDeviceSize)OW * OH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    oneShot([&](VkCommandBuffer cb) {
        VkBufferImageCopy r{};
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.imageExtent = {OW, OH, 1};
        vkCmdCopyImageToBuffer(cb, imgs[I_OUT].img, VK_IMAGE_LAYOUT_GENERAL, st.b, 1, &r);
    });
    std::string p = newShotPath();
    shotDone(p, writePng(p, (const uint8_t*)st.map, OW, OH, false));
    freeBuf(st);
}

static void saveShot(const char* path) {
    vkDeviceWaitIdle(dev);
    Buffer st = mkBuf((VkDeviceSize)OW * OH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    oneShot([&](VkCommandBuffer cb) {
        VkBufferImageCopy r{};
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.imageExtent = {OW, OH, 1};
        vkCmdCopyImageToBuffer(cb, imgs[I_OUT].img, VK_IMAGE_LAYOUT_GENERAL, st.b, 1, &r);
    });
    const uint8_t* px = (const uint8_t*)st.map;
    std::string ps(path);
    if (ps.size() > 4 && ps.compare(ps.size() - 4, 4, ".png") == 0) {  // PNG + the scene beside it
        if (writePng(ps, px, OW, OH, false)) writeShotInfo(ps);
    } else {
        FILE* f = fopen(path, "wb");
        fprintf(f, "P6\n%u %u\n255\n", OW, OH);
        for (uint32_t i = 0; i < OW * OH; i++) fwrite(px + i * 4, 1, 3, f);
        fclose(f);
    }
    freeBuf(st);
    printf("GPU %.2f ms:", gpuEmaTotal);
    for (auto& t : graph.timings()) printf(" %s%s=%.2f", t.queue == rg::COMPUTE ? "[async] " : "", t.name.c_str(), t.ms);
    printf("\n");
    auto as = audio::stats();
    printf("audio: %u emitters, %u meshes, %u instances, %u scene tris, sim %.1f ms, dsp %.2f ms\n", as.emitters, as.meshes, as.instances,
           as.sceneTris, as.simMs, as.dspMs);
}
static uint32_t resOptions[5][2] = {{1280, 720}, {1600, 900}, {1920, 1080}, {2560, 1440}, {0, 0}};  // [4]: the window's size
static bool resFollowWindow = true;  // output resolution = window size (follows resizes)
static int windowMode = 0;           // 0 windowed, 1 borderless (fills the screen, no decorations), 2 exclusive fullscreen
static int pendingWindowMode = -1;

static void applyWindowMode(int mode) {
    GLFWmonitor* mon = glfwGetPrimaryMonitor();
    const GLFWvidmode* vm = mon ? glfwGetVideoMode(mon) : nullptr;
    if (!vm) return;
    if (mode == 2) {
        glfwSetWindowMonitor(window, mon, 0, 0, vm->width, vm->height, vm->refreshRate);
    } else {
        if (glfwGetWindowMonitor(window)) glfwSetWindowMonitor(window, nullptr, 0, 0, vm->width, vm->height, 0);
        glfwSetWindowAttrib(window, GLFW_DECORATED, mode == 1 ? GLFW_FALSE : GLFW_TRUE);
        if (mode == 1) { glfwSetWindowPos(window, 0, 0); glfwSetWindowSize(window, vm->width, vm->height); }
        else glfwMaximizeWindow(window);
    }
    windowMode = mode;
    swapDirty = true;
}

// Tree / bush generator sliders (trees.h). Changes regrow the meshes when a slider is let go (regenVegetation).
static void drawVegetationUI() {
    auto R = [] { if (ImGui::IsItemDeactivatedAfterEdit()) vegDirty = true; };
    auto C = [&](const char* l, float* c) { ImGui::ColorEdit3(l, c, ImGuiColorEditFlags_Float); R(); };
    if (ImGui::TreeNode("Tree generator")) {
        for (int sp = 0; sp < trees::SPECIES; sp++) {
            trees::TreeParams& P = trees::treeParams[sp];
            ImGui::PushID(sp);
            const char* spName[trees::SPECIES] = {"Species A", "Species B", "Species C", "Species D"};
            if (ImGui::TreeNode(spName[sp])) {
                static int preset[trees::SPECIES] = {0, 3, 1, 4};
                if (ImGui::Combo("Preset", &preset[sp], trees::TREE_PRESETS, trees::TREE_PRESET_COUNT)) {
                    uint32_t seed = P.seed;
                    trees::treePreset(preset[sp], P);
                    P.seed = seed;
                    vegDirty = true;
                }
                int seed = (int)P.seed;
                if (ImGui::InputInt("Seed", &seed)) { P.seed = (uint32_t)std::max(seed, 0); vegDirty = true; }
                ImGui::SeparatorText("Trunk");
                ImGui::SliderFloat("Height", &P.height, 2.f, 30.f, "%.1f m"); R();
                ImGui::SliderFloat("Trunk radius", &P.trunkRadius, 0.05f, 1.f, "%.2f m"); R();
                ImGui::SliderFloat("Clear trunk", &P.bole, 0.05f, 0.8f, "%.2f of height"); R();
                ImGui::SliderFloat("Lean", &P.lean, 0.f, 0.5f); R();
                ImGui::SliderFloat("Gnarl", &P.gnarl, 0.f, 1.f); R();
                ImGui::SliderFloat("Root flare", &P.flare, 0.f, 2.f); R();
                ImGui::SliderFloat("Trunk lumpiness", &P.trunkWave, 0.f, 1.5f); R();
                ImGui::SetItemTooltip("Irregular, fluted cross-section and a wandering trunk line (big limbs too)");
                ImGui::SliderFloat("Wood density", &P.density, 300.f, 1200.f, "%.0f kg/m3");
                ImGui::SliderFloat("Wood strength", &P.strength, 10.f, 100.f, "%.0f MPa");
                ImGui::SetItemTooltip("Bending strength of the green wood: what a limb takes before it snaps");
                ImGui::SeparatorText("Branches");
                ImGui::Combo("Crown shape", &P.crown, trees::CROWN_NAMES, trees::CROWN_COUNT); R();
                ImGui::SliderInt("Levels", &P.levels, 1, 4); R();
                ImGui::SliderInt("Main branches", &P.branches, 2, 50); R();
                ImGui::SliderInt("Sub-branches", &P.children, 1, 12); R();
                ImGui::SliderFloat("Branch angle", &P.branchAngle, 5.f, 110.f, "%.0f deg"); R();
                ImGui::SliderFloat("Angle variation", &P.angleVar, 0.f, 40.f, "%.0f deg"); R();
                ImGui::SliderFloat("Branch length", &P.lengthRatio, 0.1f, 1.2f); R();
                ImGui::SliderFloat("Sub-branch length", &P.childLength, 0.1f, 1.2f); R();
                ImGui::SliderFloat("Branch thickness", &P.radiusRatio, 0.2f, 1.f); R();
                ImGui::SliderFloat("Droop (gravity)", &P.gravity, 0.f, 2.f); R();
                ImGui::SliderFloat("Reach for light", &P.upturn, 0.f, 1.f); R();
                ImGui::SeparatorText("Leaves");
                ImGui::SliderFloat("Leaf size", &P.leafSize, 0.02f, 0.5f, "%.2f m"); R();
                ImGui::SliderFloat("Leaf width", &P.leafWidth, 0.05f, 1.2f); R();
                ImGui::SetItemTooltip("Low values give needles");
                ImGui::SliderFloat("Leaf density", &P.leafDensity, 0.f, 4.f); R();
                ImGui::SliderFloat("Cluster spread", &P.leafSpread, 0.05f, 1.f, "%.2f m"); R();
                ImGui::SliderFloat("Leaf curl", &P.leafDroop, 0.f, 1.f); R();
                C("Leaf colour", P.leafColor);
                ImGui::SliderFloat("Colour variation", &P.hueVar, 0.f, 1.f); R();
                C("Bark colour", P.barkColor);
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        ImGui::TreePop();
    }
    if (ImGui::TreeNode("Bush generator")) {
        const char* names[trees::BUSH_KINDS] = {"Bush", "Berry bush", "Rose", "Flowering bush"};
        const char* fruit[trees::BUSH_KINDS] = {"Berries", "Berries", "Flowers", "Blossoms"};
        for (int k = 0; k < trees::BUSH_KINDS; k++) {
            trees::BushParams& B = trees::bushParams[k];
            ImGui::PushID(100 + k);
            if (ImGui::TreeNode(names[k])) {
                int seed = (int)B.seed;
                if (ImGui::InputInt("Seed", &seed)) { B.seed = (uint32_t)std::max(seed, 0); vegDirty = true; }
                ImGui::SliderFloat("Size", &B.size, 0.2f, 3.f, "%.2f m"); R();
                ImGui::SliderInt("Stems", &B.stems, 1, 16); R();
                ImGui::SliderFloat("Spread", &B.spread, 0.f, 1.5f); R();
                ImGui::SliderInt("Branch levels", &B.levels, 1, 3); R();
                ImGui::SliderFloat("Branch angle", &B.branchAngle, 10.f, 90.f, "%.0f deg"); R();
                ImGui::SliderFloat("Gnarl", &B.gnarl, 0.f, 1.f); R();
                ImGui::SliderFloat("Leaf size", &B.leafSize, 0.01f, 0.2f, "%.3f m"); R();
                ImGui::SliderFloat("Leaf density", &B.leafDensity, 0.2f, 4.f); R();
                C("Leaf colour", B.leafColor);
                char l[64];
                snprintf(l, sizeof l, "%s amount", fruit[k]);
                ImGui::SliderFloat(l, &B.fruitDensity, 0.f, 2.5f); R();
                snprintf(l, sizeof l, "%s size", fruit[k]);
                ImGui::SliderFloat(l, &B.fruitSize, 0.003f, 0.06f, "%.3f m"); R();
                snprintf(l, sizeof l, "%s colour", fruit[k]);
                C(l, B.fruitColor);
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        ImGui::TreePop();
    }
    if (ImGui::Button("Regenerate vegetation")) vegDirty = true;
}

static void drawUI() {
    ImGui::SetNextWindowPos({10, 10}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({420, 860}, ImGuiCond_FirstUseEver);
    // never taller than the screen (a saved size can be), so the panel scrolls instead of running off the bottom
    float maxH = std::max(200.f, ImGui::GetIO().DisplaySize.y - 20.f);
    ImGui::SetNextWindowSizeConstraints({250, 150}, {FLT_MAX, maxH});
    ImGui::Begin("Path Tracer", nullptr, ImGuiWindowFlags_AlwaysVerticalScrollbar);
    if (ImGui::GetWindowPos().y + ImGui::GetWindowHeight() > ImGui::GetIO().DisplaySize.y)
        ImGui::SetWindowPos({ImGui::GetWindowPos().x, std::max(0.f, ImGui::GetIO().DisplaySize.y - ImGui::GetWindowHeight())});
    ImGui::Text("%s", gpuName);
    ImGui::Text("Render %ux%u -> output %ux%u   window %ux%u", W, H, OW, OH, swapExtent.width, swapExtent.height);
    ImGui::Separator();
    ImGui::Text("FPS %.1f  (CPU frame %.2f ms)", 1000.f / cpuFrameMs, cpuFrameMs);
    ImGui::Text("GPU %.2f ms  -> GPU-bound max %.0f fps", gpuEmaTotal, gpuEmaTotal > 0 ? 1000.f / gpuEmaTotal : 0.f);
    if (ImGui::CollapsingHeader("Performance stats")) {
        if (ImGui::TreeNode("GPU passes (render graph)")) {
            // timeline: one row per pass, bar = [start, start + ms] within the frame; async compute in orange
            const auto& tl = graph.timings();
            float total = std::max(graph.frameMs(), 0.01f), width = ImGui::GetContentRegionAvail().x - 190.f;
            ImDrawList* dl = ImGui::GetWindowDrawList();
            for (auto& t : tl) {
                ImGui::Text("%-20s %5.2f", t.name.c_str(), t.ms);
                ImGui::SameLine(190.f);
                ImVec2 p = ImGui::GetCursorScreenPos();
                float x0 = p.x + width * t.startMs / total, x1 = std::max(x0 + 1.f, p.x + width * (t.startMs + t.ms) / total);
                dl->AddRectFilled({x0, p.y + 2}, {x1, p.y + ImGui::GetTextLineHeight() - 2},
                                  t.queue == rg::COMPUTE ? IM_COL32(240, 150, 40, 255) : IM_COL32(80, 160, 240, 255));
                ImGui::NewLine();
            }
            ImGui::BeginDisabled(!graph.asyncCompute());
            ImGui::Checkbox("Async compute (BVH build overlaps raster)", &asyncCompute);
            ImGui::BeginDisabled(!asyncCompute);
            ImGui::Checkbox("Async fog (overlaps ReSTIR DI + GI)", &asyncFog);
            ImGui::EndDisabled();
            ImGui::EndDisabled();
            ImGui::TreePop();
        }
        int rpp = raysPerPixel(S.raysFirst, S.bounces);
        ImGui::Text("~%d rays/pixel, %.1f M rays/frame, %.2f Grays/s", rpp, rpp * W * H / 1e6f,
                    gpuEmaTotal > 0 ? rpp * W * H / (gpuEmaTotal * 1e6f) : 0.f);
        ImGui::Text("Blades: %u generated (BLAS slots %u, max %u)", lastBladeCount, bladeCap, maxBlades);
        ImGui::Text("Tris: terrain %u, grass %u, rocks %u x %u (16 shapes)", terrainTris, std::min(lastBladeCount, bladeCap) * 5, numRocks, rockTris);
        ImGui::Text("BVH alloc: terrain %.1f MB, grass %.1f MB (max), rock %.2f MB", terrainAS.size / 1048576.0, grassAS.size / 1048576.0,
                    rockAS[0].size * ROCK_VARIANTS / 1048576.0);

    }
    if (ImGui::CollapsingHeader("Benchmark")) {
        if (bench.running) {
            float prog = ((bench.tier - 1) * (Bench::WARM + Bench::MEAS) + bench.frame) / float(4 * (Bench::WARM + Bench::MEAS));
            ImGui::ProgressBar(prog, {-1, 0}, "running...");
            ImGui::Text("Testing %d ray(s) per first bounce", bench.tier);
        } else if (ImGui::Button("Run benchmark (~20 s)", {-1, 0})) {
            startBenchmark();
        }
        if (bench.done) {
            if (ImGui::BeginTable("bench", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
                ImGui::TableSetupColumn("Rays/bounce");
                ImGui::TableSetupColumn("avg ms");
                ImGui::TableSetupColumn("p95 ms");
                ImGui::TableSetupColumn("fps (p95)");
                ImGui::TableHeadersRow();
                for (int t = 1; t <= 4; t++) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn(); ImGui::Text("%d  (%d rpp)", t, raysPerPixel(t, 2));
                    ImGui::TableNextColumn(); ImGui::Text("%.2f", bench.avg[t]);
                    ImGui::TableNextColumn(); ImGui::Text("%.2f", bench.p95[t]);
                    ImGui::TableNextColumn(); ImGui::Text("%.0f", bench.p95[t] > 0 ? 1000.f / bench.p95[t] : 0.f);
                }
                ImGui::EndTable();
            }
            ImGui::TextWrapped("What your system can do at %ux%u:", W, H);
            for (int fps : {60, 40, 30, 25}) {
                int t = benchBestTier(fps);
                if (t) ImGui::BulletText("%d fps -> %d ray(s) per bounce", fps, t);
                else ImGui::BulletText("%d fps -> not reachable (lower resolution)", fps);
            }
        } else if (!bench.running) {
            ImGui::TextDisabled("Run the benchmark to see what fps / quality you can get.");
        }

    }
    if (ImGui::CollapsingHeader("Frame rate target")) {
        ImGui::BeginDisabled(bench.running);
        const int opts[] = {0, 60, 40, 30, 25};
        for (int i = 0; i < 5; i++) {
            if (i) ImGui::SameLine();
            char label[16];
            snprintf(label, sizeof label, i == 0 ? "Manual" : "%d", opts[i]);
            if (ImGui::RadioButton(label, targetFps == opts[i])) {
                targetFps = opts[i];
                if (targetFps && bench.done) { int t = benchBestTier(targetFps); if (t) S.raysFirst = t; }
            }
        }
        if (ImGui::Checkbox("VSync", &vsync)) swapDirty = true;
        if (targetFps) {
            ImGui::Text("Budget %.1f ms -> auto: %d ray(s) per bounce", 1000.f / targetFps, S.raysFirst);
            if (S.raysFirst == 1 && gpuEmaTotal > 1000.f / targetFps) ImGui::TextColored({1, 0.4f, 0.3f, 1}, "Can't hold target even at 1 ray: lower resolution");
        } else {
            ImGui::SliderInt("Rays per first bounce", &S.raysFirst, 1, 4);
        }
        ImGui::EndDisabled();

    }
    if (ImGui::CollapsingHeader("Rendering")) {
        ImGui::BeginDisabled(bench.running);
        int resIdx = 2;
        for (int i = 0; i < 4; i++) if (resOptions[i][0] == OW && resOptions[i][1] == OH) resIdx = i;
        if (resFollowWindow) resIdx = 4;
        char winRes[32];
        snprintf(winRes, sizeof winRes, "Window (%ux%u)", swapExtent.width, swapExtent.height);
        const char* resNames[] = {"720p", "900p", "1080p", "1440p", winRes};
        if (ImGui::Combo("Output res", &resIdx, resNames, 5)) {
            resFollowWindow = resIdx == 4;
            if (resFollowWindow) { resOptions[4][0] = swapExtent.width; resOptions[4][1] = swapExtent.height; }
            if (resOptions[resIdx][0] != OW || resOptions[resIdx][1] != OH) pendingRes = resIdx;
        }
        ImGui::BeginDisabled(!dlss::available());
        int um = upscaler;
        const char* umNames[dlss::MODE_COUNT];
        for (int i = 0; i < dlss::MODE_COUNT; i++) umNames[i] = dlss::modeName((dlss::Mode)i);
        if (ImGui::Combo("DLSS Ray Reconstruction", &um, umNames, dlss::MODE_COUNT) && um != upscaler) { upscaler = (dlss::Mode)um; pendingRes = resIdx; }
        ImGui::EndDisabled();
        if (!dlss::available()) ImGui::TextDisabled("DLSS-RR unavailable (needs RTX GPU + driver, SDK in external/dlss)");
        else if (rrActive) ImGui::TextDisabled("RR denoises + upscales; denoiser settings below are bypassed");
        ImGui::Checkbox("Raster primary (G-buffer)", &S.rasterPrimary);
        ImGui::SameLine();
        if (ImGui::Checkbox("Global illumination (I)", &S.giEnabled)) resetHistory = true;
        ImGui::SliderInt("Bounces", &S.bounces, 1, 4);
        ImGui::Checkbox("ReSTIR DI", &S.restir);
        ImGui::BeginDisabled(!S.restir);
        ImGui::SameLine(); ImGui::Checkbox("temporal", &S.restirTemporal);
        ImGui::SameLine(); ImGui::Checkbox("spatial", &S.restirSpatial);
        ImGui::SliderInt("Candidates (M)", &S.candidates, 1, 32);
        ImGui::SliderInt("Spatial samples", &S.spatialSamples, 1, 8);
        ImGui::SliderFloat("Spatial radius px", &S.spatialRadius, 2, 40);
        ImGui::EndDisabled();
        ImGui::Checkbox("ReSTIR GI", &S.restirGI);
        ImGui::BeginDisabled(!S.restirGI);
        ImGui::SameLine(); ImGui::Checkbox("temporal##gi", &S.giTemporal);
        ImGui::SameLine(); ImGui::Checkbox("spatial##gi", &S.giSpatial);
        ImGui::SliderInt("GI spatial samples", &S.giSpatialSamples, 1, 8);
        if (ImGui::Checkbox("radiance cache", &S.giCache)) resetHistory = true;
        ImGui::SameLine(); if (ImGui::Checkbox("checkerboard", &S.giChecker)) resetHistory = true;
        ImGui::SameLine(); if (ImGui::Checkbox("probes at bounce 2", &S.giProbes)) resetHistory = true;
        ImGui::SliderFloat("GI spatial radius px", &S.giRadius, 2, 60);
        ImGui::SliderFloat("GI history cap (M)", &S.giMaxM, 1, 100);
        ImGui::EndDisabled();

        ImGui::EndDisabled();
    }
    if (ImGui::CollapsingHeader("Display & window")) {
        const char* wm[] = {"Windowed (maximised)", "Borderless (fills the screen)", "Fullscreen (exclusive)"};
        int m = windowMode;
        if (ImGui::Combo("Window", &m, wm, 3) && m != windowMode) pendingWindowMode = m;
        if (ImGui::Checkbox("VSync##disp", &vsync)) swapDirty = true;
        const char* hm[] = {"SDR", "HDR10 (PQ)", "scRGB (fp16)"};
        int h = S.hdrMode;
        if (ImGui::Combo("Output", &h, hm, 3) && h != S.hdrMode) { S.hdrMode = h; swapDirty = true; }
        if (S.hdrMode && !hdrModeAvail[S.hdrMode])
            ImGui::TextColored({1, 0.6f, 0.3f, 1}, haveColorspaceExt ? "The display / compositor doesn't offer this HDR mode: showing SDR"
                                                                     : "No VK_EXT_swapchain_colorspace: HDR unavailable, showing SDR");
        ImGui::BeginDisabled(hdrActive == 0);
        ImGui::SliderFloat("Paper white (nits)", &S.paperWhite, 80, 500, "%.0f");
        ImGui::SliderFloat("Peak brightness (nits)", &S.peakNits, 400, 4000, "%.0f");
        ImGui::EndDisabled();
    }
    if (ImGui::CollapsingHeader("Exposure, tonemap & bloom")) {
        ImGui::Checkbox("Physical camera (f-stop / shutter / ISO)", &S.physCam);
        if (S.physCam) {
            ImGui::SliderFloat("f-stop", &S.fStop, 1.f, 22.f, "f/%.1f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("Shutter", &S.shutterDen, 1.f, 8000.f, "1/%.0f s", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("ISO", &S.iso, 50.f, 25600.f, "%.0f", ImGuiSliderFlags_Logarithmic);
            float ev100 = log2f(S.fStop * S.fStop * S.shutterDen) - log2f(S.iso / 100.f);
            ImGui::Text("EV100 %.1f (sunny day ~15, overcast ~12, indoors ~7, night ~0)", ev100);
            ImGui::SliderFloat("Scene luminance scale", &S.sceneLum, 100.f, 1e6f, "%.0f cd/m2", ImGuiSliderFlags_Logarithmic);
            ImGui::SetItemTooltip("cd/m^2 per unit of scene radiance (calibration of the renderer's units)");
        } else ImGui::Checkbox("Auto exposure (eye adaptation)", &S.autoExposure);
        if (S.physCam) {
        } else if (S.autoExposure) {
            ImGui::SliderFloat("Exposure compensation (EV)", &S.evComp, -4, 4, "%.1f");
            ImGui::SliderFloat("Adapt to brighter (1/s)", &S.adaptUp, 0.1f, 10, "%.1f", ImGuiSliderFlags_Logarithmic);
            ImGui::SliderFloat("Adapt to darker (1/s)", &S.adaptDown, 0.1f, 10, "%.1f", ImGuiSliderFlags_Logarithmic);
            const float* ex = (const float*)((const char*)exposureBuf.map + 512);
            ImGui::Text("Metered %.1f EV, exposure x%.2f", ex[1], ex[0]);
        } else {
            ImGui::SliderFloat("Exposure", &S.exposure, 0.01f, 64, "%.2f", ImGuiSliderFlags_Logarithmic);
        }
        const char* tm[] = {"AgX (neutral, filmic)", "AgX punchy", "ACES (fitted)", "None (clip)"};
        ImGui::Combo("Tonemapper", &S.tonemapper, tm, 4);
        ImGui::SliderFloat("Bloom", &S.bloom, 0, 0.3f, "%.3f");
        ImGui::SliderFloat("Sharpen (CAS)", &S.sharpen, 0, 1, "%.2f");
    }
    if (ImGui::CollapsingHeader("Lens & camera effects")) {
        ImGui::Checkbox("Depth of field", &S.dof);
        ImGui::SliderFloat("Focus distance", &S.focusDist, 0.f, 200.f, S.focusDist <= 0.f ? "auto (centre)" : "%.1f m", ImGuiSliderFlags_Logarithmic);
        if (!S.physCam) ImGui::SliderFloat("Aperture", &S.fStop, 1.f, 22.f, "f/%.1f", ImGuiSliderFlags_Logarithmic);
        ImGui::Text("Focal length %.0f mm (from the FOV, full frame)", 12.f / tanf(cam.fovDeg * 0.5f * 3.14159265f / 180.f));
        ImGui::Checkbox("Motion blur", &S.motionBlur);
        if (!S.physCam) ImGui::SliderFloat("Shutter angle", &S.mbShutter, 0.f, 1.f, "%.2f of the frame");
        else ImGui::TextDisabled("Shutter time from the physical camera");
        ImGui::SliderFloat("Vignetting", &S.vignette, 0.f, 1.f, "%.2f");
        ImGui::SliderFloat("Chromatic aberration", &S.chromAb, 0.f, 8.f, "%.1f px");
        ImGui::SliderFloat("Sensor noise", &S.grain, 0.f, 3.f, "%.2f");
        if (S.physCam) ImGui::SameLine(), ImGui::TextDisabled("x ISO");
    }
    if (ImGui::CollapsingHeader("Denoiser & debug view")) {
        ImGui::BeginDisabled(bench.running);
        ImGui::Checkbox("Temporal accumulation", &S.accumulate);
        ImGui::SameLine(); ImGui::Checkbox("A-trous filter", &S.denoise);
        ImGui::SliderInt("History max", &S.historyMax, 1, 128);
        ImGui::SliderInt("A-trous iterations", &S.atrousIters, 0, 5);
        ImGui::SliderFloat("Firefly clamp", &S.fireflyClamp, 1, 100, "%.0f", ImGuiSliderFlags_Logarithmic);
        if (ImGui::Button("Reset history")) resetHistory = true;
        const char* views[] = {"Final", "Raw (no denoise)", "Direct only", "Indirect only", "Albedo", "Normals", "History length", "Variance"};
        ImGui::Combo("View", &S.viewMode, views, 8);

        ImGui::EndDisabled();
    }
    if (ImGui::CollapsingHeader("Camera & movement")) {
        ImGui::SliderFloat("FOV", &cam.fovDeg, 30, 110, "%.0f deg");
        if (ImGui::Checkbox("Walk (F)", &walkMode) && walkMode) {
            V3 feet = cam.pos - V3{0, 1.65f, 0};
            feet.y = std::max(feet.y, heightFn(feet.x, feet.z) + 0.05f);
            physics::spawnPlayer(&feet.x);
        }
        if (walkMode) { ImGui::SameLine(); ImGui::Text("%s  feet %.1f %.1f %.1f", grounded ? "grounded" : "airborne", playerFeet.x, playerFeet.y, playerFeet.z); }
        ImGui::TextDisabled("RMB + mouse: look   WASD/QE: move   Shift: fast   Ctrl: turbo (fly)   F: walk/fly   Space: jump");
        ImGui::SliderFloat("Fly speed", &flySpeed, 0.25f, 20.f, "%.2fx", ImGuiSliderFlags_Logarithmic);

    }
    if (ImGui::CollapsingHeader("Sun, sky & exposure")) {
        ImGui::BeginDisabled(bench.running);
        ImGui::SliderFloat("Haze", &S.fogScale, 0, 100, "%.0f");
        ImGui::SliderFloat("Sun elevation", &S.sunElev, -10, 90);
        ImGui::SliderFloat("Sun azimuth", &S.sunAzim, 0, 360);
        ImGui::SliderFloat("Sun intensity", &S.sunIntensity, 0, 20);
        ImGui::SliderFloat("Sun size (deg)", &S.sunRadiusDeg, 0.1f, 5);
        ImGui::SliderFloat("Sky intensity", &S.skyIntensity, 0, 3);
        ImGui::SliderFloat("Exposure", &S.exposure, 0.1f, 8, "%.2f", ImGuiSliderFlags_Logarithmic);
        if (ImGui::Button("Night")) { S.sunElev = -8; S.lightIntensity = 3; S.exposure = 3; }
        ImGui::SameLine();
        if (ImGui::Button("Sunset")) { S.sunElev = 6; S.sunIntensity = 6; S.exposure = 1.8f; }
        ImGui::SameLine();
        if (ImGui::Button("Day")) { S.sunElev = 45; S.sunIntensity = 6; S.exposure = 1.2f; }
        ImGui::EndDisabled();
    }
    if (ImGui::CollapsingHeader("Clouds")) {
        ImGui::BeginDisabled(bench.running);
        ImGui::Checkbox("Enabled##clouds", &S.clouds);
        ImGui::SliderFloat("Cloud coverage", &S.cloudCoverage, 0, 1);
        ImGui::SetItemTooltip("How much of the sky is cloud. Every cloud type below is a share of this: 0 = clear sky");
        ImGui::SliderFloat("Cloud density", &S.cloudDensity, 0.002f, 0.2f, "%.3f", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Cloud drift", &S.cloudSpeed, 0, 10);
        ImGui::SliderFloat("Cloud shadows", &S.cloudShadow, 0, 1);
        ImGui::SeparatorText("Cloud types (share of coverage)");
        auto share = [](float w) { return S.cloudCoverage * w * 100.f; };
        ImGui::SliderFloat("Cumulus", &S.cumulus, 0, 1);
        ImGui::SameLine(); ImGui::TextDisabled("%.0f%%", share(S.cumulus));
        ImGui::SliderFloat("Cumulus base (m)", &S.cloudBase, 300, 4000, "%.0f");
        ImGui::SliderFloat("Cumulus thickness (m)", &S.cloudThickness, 200, 4000, "%.0f");
        ImGui::Spacing();
        ImGui::SliderFloat("Stratus", &S.stratus, 0, 1);
        ImGui::SameLine(); ImGui::TextDisabled("%.0f%%", share(S.stratus));
        ImGui::SliderFloat("Stratus base (m)", &S.stratusBase, 100, 2500, "%.0f");
        ImGui::SliderFloat("Stratus thickness (m)", &S.stratusThick, 50, 1500, "%.0f");
        ImGui::SliderFloat("Stratus lumpiness", &S.stratusLumpy, 0, 1);
        ImGui::SetItemTooltip("0 = flat stratus sheet, 1 = broken stratocumulus cells");
        ImGui::Spacing();
        ImGui::SliderFloat("Cumulonimbus", &S.cumulonimbus, 0, 1);
        ImGui::SameLine(); ImGui::TextDisabled("%.0f%%", share(S.cumulonimbus));
        ImGui::SliderFloat("Cumulonimbus top (m)", &S.cbTop, 5000, 15000, "%.0f");
        ImGui::Spacing();
        ImGui::SliderFloat("Cirrus", &S.cirrus, 0, 1);
        ImGui::SameLine(); ImGui::TextDisabled("%.0f%%", share(S.cirrus));
        ImGui::SliderFloat("Cirrus altitude (m)", &S.cirrusAlt, 6000, 13000, "%.0f");
        ImGui::SliderFloat("Cirrus strand length", &S.cirrusLength, 0.3f, 3.f, "%.2fx");
        ImGui::SliderFloat("Cirrus strand fineness", &S.cirrusFine, 0, 1);
        ImGui::SliderFloat("Cirrus puffiness", &S.cirrusPuff, 0, 1);
        ImGui::SetItemTooltip("0 = fine hair-like strands, 1 = cotton tufts trailing wisps");
        ImGui::EndDisabled();
    }
    if (ImGui::CollapsingHeader("Wind")) {
        ImGui::Checkbox("Simulated wind map", &S.windMap);
        ImGui::Checkbox("Refit grass BVH (rebuild every 8 frames / on camera move)", &S.grassRefit);
        ImGui::SetItemTooltip("Gust fronts from different directions sweep through; where they collide, pressure waves ripple the grass and water");
        ImGui::BeginDisabled(!S.windMap);
        ImGui::SliderFloat("Gust fronts", &S.gustStrength, 0, 1, "%.2fx");
        ImGui::SliderFloat("Front direction spread", &S.gustSpreadDeg, 0, 180, "%.0f deg");
        ImGui::SliderFloat("Wave speed", &S.windWaveSpeed, 2, 30, "%.1f m/s");
        ImGui::SliderFloat("Wave damping", &S.windDamping, 0, 2, "%.2f /s");
        ImGui::SliderFloat("Gust response", &S.windRelax, 0.1f, 5, "%.2f /s");
        ImGui::EndDisabled();
        ImGui::BeginDisabled(bench.running);
        ImGui::SliderFloat("Wind strength", &S.windStrength, 0, 1.5f);
        ImGui::SliderFloat("Wind speed", &S.windSpeed, 0, 6);
        ImGui::SliderFloat("Wind direction", &S.windDirDeg, 0, 360);
        ImGui::EndDisabled();
    }
    if (ImGui::CollapsingHeader("Lights")) {
        ImGui::BeginDisabled(bench.running);
        ImGui::SliderInt("Light orbs", &S.lightCount, 0, MAX_ORBS);
        ImGui::SliderFloat("Orb range (m)", &S.orbRange, 20.f, 400.f, "%.0f");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("Orbs live in a square this wide around you and fade out at its edge");
        ImGui::SliderFloat("Orb intensity", &S.lightIntensity, 0, 10);
        ImGui::SliderFloat("Model light scale", &lightScale, 1e-4f, 0.1f, "%.4f", ImGuiSliderFlags_Logarithmic);
        ImGui::EndDisabled();
    }
    if (ImGui::CollapsingHeader("Grass")) {
        ImGui::BeginDisabled(bench.running);
        ImGui::SliderFloat("Density", &S.grassDensity, 10, 400, "%.0f blades/m2", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Length", &S.grassLength, 0.2f, 3, "x%.2f");
        ImGui::SliderFloat("Full density until", &S.lodStart, 2, 30, "%.0f m");
        ImGui::SliderFloat("Draw distance", &S.drawDist, 15, 150, "%.0f m");

        ImGui::SeparatorText("Blades");
        ImGui::SliderFloat("Clumping", &S.grassClump, 0, 1);
        ImGui::SetItemTooltip("0 = evenly scattered blades, 1 = tufts that splay out from their centre");
        ImGui::SliderFloat("Clump size", &S.grassClumpSize, 0.08f, 1.5f, "%.2f m");
        ImGui::SliderFloat("Splay", &S.grassTilt, 0, 1);
        ImGui::SliderFloat("Blade width", &S.grassWidth, 0.3f, 3.f, "x%.2f");
        ImGui::SliderFloat("Curvature", &S.grassCurve, 0, 3, "x%.2f");
        ImGui::SliderFloat("Twist", &S.grassTwist, 0, 1);
        ImGui::SliderFloat("Height variation", &S.grassHeightVar, 0, 1);
        ImGui::SliderFloat("Dead blades", &S.grassDead, 0, 1, "%.2f");
        ImGui::SliderFloat("Colour variation", &S.grassColorVar, 0, 2);

        ImGui::SeparatorText("Culling");
        ImGui::SliderFloat("Cull margin", &S.fovMargin, 0, 40, "+%.0f deg");
        ImGui::SliderFloat("Keep radius", &S.keepRadius, 0, 15, "%.1f m");
        ImGui::SetItemTooltip("Grass within this distance is kept at full density even behind the camera");
        ImGui::SliderFloat("Off-screen grass", &S.offscreenDensity, 0, 1, "%.2f");
        ImGui::SetItemTooltip("Density of grass outside the view (thinner, wider blades) so it still casts shadows and reflects");

        ImGui::EndDisabled();
    }
    if (ImGui::CollapsingHeader("Ground textures")) {
        ImGui::Combo("Ground detail", &S.groundMode, "Tiled textures\0Fractal (no repeat)\0Hybrid\0Layered (virtual texture)\0");
        if (S.groundMode == 3) {
            ImGui::SliderInt("VT tiles / frame", &S.vtBudget, 8, 1024);
            ImGui::Text("VT: %u tiles generated this frame", vtQueued);
        }
        ImGui::SetItemTooltip("Fractal: generated per pixel from the world position (eroded fBm, pebbles, grit), detail down to\n"
                              "the pixel at any distance and nothing ever repeats. Hybrid: the textures, broken up by it.");
        ImGui::BeginDisabled(S.groundMode == 0);
        ImGui::SliderFloat("Fractal relief", &S.fractalRelief, 0.f, 3.f, "%.2fx");
        ImGui::SliderFloat("Pebbles", &S.fractalPebbles, 0.f, 1.f, "%.2f");
        ImGui::SliderFloat("Finest detail", &S.fractalFinest, 0.0005f, 0.05f, "%.4f m", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Large-scale warp", &S.groundWarp, 0.f, 4.f, "%.2fx");
        ImGui::SliderFloat("Grit", &S.groundGrit, 0.f, 3.f, "%.2fx");
        ImGui::SliderFloat("Mottling", &S.groundMottle, 0.f, 3.f, "%.2fx");
        ImGui::SliderFloat("Mottling scale", &S.groundMottleScale, 0.2f, 5.f, "%.2fx");
        ImGui::EndDisabled();
        ImGui::SeparatorText("Variation");
        ImGui::SliderFloat("Dark spots", &S.darkSpots, 0.f, 1.f, "%.2f");
        ImGui::SetItemTooltip("Damp, humus-rich patches");
        ImGui::SliderFloat("Dark spot size", &S.darkSpotSize, 0.1f, 5.f, "%.2fx");
        ImGui::SliderFloat("Hollow shading", &S.groundCavity, 0.f, 2.f, "%.2fx");
        ImGui::SliderFloat("Colour variation", &S.groundSat, 0.f, 2.f, "%.2f");
        ImGui::SetItemTooltip("Patches of drier (greyer) and lusher (more saturated) ground");
        ImGui::SliderFloat("Tile breakup", &S.tileBreakup, 0.f, 1.f, "%.2f");
        ImGui::BeginDisabled(!groundReady);
        ImGui::Checkbox("Generated ground textures", &S.groundTextures);
        ImGui::Text("%ux%u per 4 m tile (%.1f mm/texel), cached in texcache/ (--ground-tex N to change)", groundTexSize, groundTexSize,
                    4000.f / groundTexSize);
        ImGui::SliderFloat("Relief strength", &S.groundNormal, 0, 3, "%.2fx");
        ImGui::Checkbox("Parallax occlusion", &S.parallax);
        ImGui::SliderFloat("Parallax distance", &S.parallaxDist, 2, 40, "%.0f m");
        ImGui::EndDisabled();
    }
    if (ImGui::CollapsingHeader("Trees & plants")) {
        if (ImGui::SliderFloat("Ground clutter", &S.clutter, 0.f, 3.f, "%.2fx")) placeDirty = true;
        ImGui::SetItemTooltip("Ferns, fallen twigs, leaf litter under the trees, pebbles on bare ground and river banks");
        if (ImGui::SliderFloat("Clutter distance", &S.clutterRange, 10.f, 120.f, "%.0f m")) placeDirty = true;
        ImGui::Text("Trees %u in BVH, %u meshes + %u impostors drawn, %zu trunk colliders", numTrees, numTreeNear, numImpostors, treeBodies.size());
        ImGui::Text("Plants %u in BVH, %u drawn", numPlants, numPlantsDrawn);
        ImGui::SliderFloat("Tree density", &S.treeDensity, 0.f, 2.f, "%.2fx");
        ImGui::SliderFloat("Impostors from", &S.impostorDist, 30.f, 400.f, "%.0f m");
        if (ImGui::SliderFloat("Bushes", &S.bushAmount, 0.f, 4.f, "%.2fx")) placeDirty = true;
        if (ImGui::SliderFloat("Berry bush share", &S.berryShare, 0.f, 1.f, "%.2f")) placeDirty = true;
        drawVegetationUI();
    }
    if (ImGui::CollapsingHeader("River & water")) {
        bool shape = ImGui::SliderFloat("Half-width", &gRiver.width, 2.f, 14.f, "%.1f m");
        shape |= ImGui::SliderFloat("Depth", &gRiver.depth, 0.3f, 4.f, "%.2f m");
        shape |= ImGui::SliderFloat("Riverbed bumps", &gRiver.bumps, 0.f, 1.5f, "%.2f m");
        ImGui::SetItemTooltip("Uneven bed instead of a smooth half-pipe: humps, scours and ridges (also in the collision)");
        if (shape) { terrainDirty = placeDirty = true; stream::terrainChanged(); }  // carved terrain + collision follow
        ImGui::SliderFloat("Crater inflow", &S.craterInflow, 0.02f, 1.f, "%.2f of the river");
        ImGui::SliderFloat("Rise with rain", &S.riverRainRise, 0.f, 2.f, "%.2f m at full");
        ImGui::SetItemTooltip("Rain swells the river over a minute or so (faster current, wider water); it drains back slowly after");
        ImGui::Text("River swell now: +%.2f m", riverSwell);
        ImGui::SetItemTooltip("How much of the river's flow pours into a crater that breaks into the channel (fills it over time)");
        ImGui::SliderFloat("Flow strength", &S.flow, 0.f, 3.f, "%.2fx");
        ImGui::SliderFloat("Clarity", &S.clarity, 0.1f, 4.f, "%.2fx");
        ImGui::SliderFloat("Absorption", &S.absorb, 0.05f, 3.f, "%.2f /m");
        ImGui::ColorEdit3("Murk colour", S.murk);
        ImGui::SliderFloat("Foam", &S.foam, 0.f, 2.f);
        ImGui::SliderFloat("Foam trail length", &S.foamTrail, 0.f, 40.f, "%.1f m");
        ImGui::SetItemTooltip("How far the white wake behind a rock reaches downstream");
        ImGui::SliderFloat("Bank splashes", &S.splash, 0.f, 2.f);
        ImGui::SliderFloat("Caustics on the bed", &S.caustics, 0.f, 3.f, "%.2f");
    }
    if (ImGui::CollapsingHeader("Fog & god rays")) {
        ImGui::Checkbox("Volumetric fog", &S.fog);
        ImGui::BeginDisabled(!S.fog);
        {
            static const char* fogResNames[] = {"full", "1/2", "1/3", "1/4"};
            int fi = std::clamp(S.fogDiv, 1, 4) - 1;
            if (ImGui::SliderInt("Fog resolution", &fi, 0, 3, fogResNames[fi])) S.fogDiv = fi + 1;
        }
        ImGui::SliderFloat("Density (at the ground)", &S.fogDensity, 0.0005f, 0.1f, "%.4f /m", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Height falloff", &S.fogFalloff, 0.005f, 1.f, "%.3f /m", ImGuiSliderFlags_Logarithmic);
        ImGui::SetItemTooltip("Higher: the fog hugs the ground (valley mist). Lower: it fills the air (haze)");
        ImGui::SliderFloat("Forward scattering", &S.fogAniso, 0.f, 0.95f, "%.2f");
        ImGui::SetItemTooltip("How strongly the fog glows looking toward the sun (light shafts)");
        ImGui::SliderFloat("Patchiness", &S.fogNoise, 0.f, 1.f, "%.2f");
        ImGui::SliderFloat("Distance", &S.fogDist, 30.f, 400.f, "%.0f m");
        ImGui::Checkbox("Lamps glow in the fog", &S.fogLights);
        ImGui::SliderFloat("Temporal smoothing", &S.fogBlend, 0.03f, 1.f, "%.2f");
        ImGui::SetItemTooltip("Share of the new frame (lower: smoother, more lag)");
        ImGui::EndDisabled();
    }
    if (ImGui::CollapsingHeader("Weather")) {
        ImGui::SliderFloat("Precipitation", &S.rain, 0.f, 1.f, "%.2f");
        ImGui::SetItemTooltip("Rain below the snow line, snow above it (falls, settles and piles up there)");
        ImGui::SeparatorText("Snow, avalanches, melt water");
        ImGui::Checkbox("Simulate (2 km around you)", &S.worldSim);
        ImGui::Checkbox("Snow line follows the cloud base", &S.snowFollowsClouds);
        ImGui::BeginDisabled(S.snowFollowsClouds);
        ImGui::SliderFloat("Snow line", &S.snowLine, -20.f, 3000.f, "%.0f m");
        ImGui::EndDisabled();
        ImGui::SliderFloat("Snowfall", &S.snowfall, 0.f, 60.f, "%.1f m/h at full", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Melt rate", &S.meltRate, 0.f, 20.f, "%.2f m/h", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Angle of repose", &S.reposeDeg, 15.f, 60.f, "%.0f deg");
        ImGui::SetItemTooltip("Snow piled steeper than this slides. Real snow: ~30-45 deg (wet / cold powder)");
        ImGui::SliderFloat("Avalanche mobility", &S.avalanche, 0.1f, 4.f, "%.2f");
        ImGui::SliderFloat("River search range", &S.riverRange, 0.f, 1000.f, "%.0f m");
        ImGui::SetItemTooltip("Melt water within this distance of a river is pulled toward it (and drains into it);\n"
                              "further away it pools and soaks in, leaving wet ground");
        ImGui::SliderFloat("Ground drying", &S.dryRate, 0.0002f, 0.05f, "%.4f /s", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Sim speed", &S.simSpeed, 1.f, 100.f, "%.0fx", ImGuiSliderFlags_Logarithmic);
        ImGui::SetItemTooltip("Fast-forward snowfall, melt, runoff and drying (avalanches move faster too)");
        ImGui::SliderInt("Terrain refresh (frames)", &S.snowMeshEvery, 1, 30);
        if (ImGui::Button("Clear snow + water")) simClear = true;
        ImGui::SeparatorText("Ground under rocks");
        ImGui::SliderFloat("Soil friction", &S.groundFriction, 0.05f, 1.2f, "%.2f");
        ImGui::SliderFloat("Soil bounce", &S.groundBounce, 0.f, 0.8f, "%.2f");
        ImGui::SetItemTooltip("Natural ground. Painted / zone textures use their kind (stone and paving bounce, sand and mud\n"
                              "absorb, ice slides); craters are loose soil; wet ground is slicker; snow is slippery and dead.");
        ImGui::Checkbox("Rain wets the ground", &S.rainWets);
        ImGui::BeginDisabled(S.rainWets);
        ImGui::SliderFloat("Wetness", &S.wetness, 0.f, 1.f, "%.2f");
        ImGui::EndDisabled();
        if (S.rainWets) ImGui::Text("Wetness %.2f (soaks in while it rains, dries after)", S.wetness);
        ImGui::SliderFloat("Puddles", &S.puddles, 0.f, 1.f, "%.2f");
    }
    if (ImGui::CollapsingHeader("Impact marks & dust")) {
        ImGui::Checkbox("Decals (cracks, soil splatter, wood scars)", &S.decals);
        ImGui::Text("%zu decals, %zu dust clouds", decalList.size(), dustList.size());
        if (ImGui::Button("Clear decals")) decalList.clear();
    }
    if (ImGui::CollapsingHeader("Characters")) {
        static char asset[128] = "characters/CesiumMan";
        ImGui::InputText("Asset", asset, sizeof asset);
        ImGui::SetItemTooltip("A skinned glTF under world/assets (characters/Fox, characters/CesiumMan, ...)");
        if (ImGui::Button("Spawn in front of me")) {
            V3 f = cam.fwd();
            float l = std::max(sqrtf(f.x * f.x + f.z * f.z), 1e-4f);
            charSpawns.push_back({asset, "", {cam.pos.x + f.x / l * 4.f, 0.f, cam.pos.z + f.z / l * 4.f}, atan2f(-f.x, -f.z), 0.f, 0.f});
        }
        std::vector<character::Character*> gone;
        int n = 0;
        for (character::Character* c : character::all()) {
            ImGui::PushID(c);
            char title[160];
            snprintf(title, sizeof title, "#%d %s", n++, ((GpuChar*)c->user)->asset->model->path.substr(charWorldDir.size()).c_str());
            ImGui::SeparatorText(title);
            const ModelData& d = *c->data;
            int cur = c->next >= 0 ? c->next : c->clip;
            if (ImGui::BeginCombo("Clip", cur >= 0 ? d.clips[cur].name.c_str() : "(rest pose)")) {
                for (int i = 0; i < (int)d.clips.size(); i++)
                    if (ImGui::Selectable(d.clips[i].name.c_str(), i == cur)) character::play(*c, i, 0.3f);
                ImGui::EndCombo();
            }
            ImGui::SliderFloat("Speed", &c->rate, 0.f, 3.f, "%.2fx");
            ImGui::SliderFloat("Walk", &c->walkSpeed, 0.f, 6.f, "%.1f m/s");
            ImGui::SetItemTooltip("Wander forward over the terrain (match it to the clip's stride)");
            ImGui::SliderFloat("Get up after", &c->getUpAfter, 0.f, 20.f, c->getUpAfter > 0 ? "%.1f s at rest" : "never");
            if (c->mode == character::ANIMATED) { if (ImGui::Button("Ragdoll")) { float j[3] = {0, 0, 0}; character::ragdoll(*c, j, c->pos); } }
            else if (ImGui::Button("Get up")) character::getUp(*c);
            ImGui::SameLine();
            if (ImGui::Button("Remove")) gone.push_back(c);
            ImGui::TextDisabled("%zu joints, %zu ragdoll bodies, %.0f kg", c->globals.size(), c->parts.size(), c->mass);
            ImGui::PopID();
        }
        for (auto* c : gone) releaseCharacter(c);
    }
    if (ImGui::CollapsingHeader("Throw rocks & destruction")) {
        ImGui::TextDisabled("G / left click / right trigger");
        ImGui::SliderFloat("Rock size", &S.throwSize, 0.05f, 2.f, "%.2f m", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Throw speed", &S.throwSpeed, 0.f, 300.f, "%.0f m/s", ImGuiSliderFlags_Logarithmic);
        {
            float m = 2600.f * 2.6f * S.throwSize * S.throwSize * S.throwSize;  // rock hull ~2.6 m^3 at scale 1
            ImGui::Text("~%.0f kg, %.1f kJ at release", m, 0.5f * m * S.throwSpeed * S.throwSpeed / 1000.f);
        }
        ImGui::Text("%zu thrown", thrown.size());
        if (ImGui::Button("Clear thrown rocks")) while (!thrown.empty()) dropThrown(thrown.size() - 1);
        ImGui::Text("%d craters, %zu trees down, %zu debris", std::min(gCraterCount.load(), MAX_CRATERS), brokenTrees.size(), spray.size());
        if (ImGui::Button("Repair world")) {  // craters filled, trees stood back up
            gCraterCount.store(0);
            while (!stumps.empty()) dropStump(stumps.size() - 1);
            std::vector<std::shared_ptr<Model>> freed;
            stream::clearOverrides(freed);
            for (auto& m : freed) {
                damagedModels.erase(std::remove(damagedModels.begin(), damagedModels.end(), m), damagedModels.end());
                retireNext.push_back([m]() {
                    if (m->gpu) releaseModel(*m);
                    if (m->data.meshes[0].shape) physics::releaseShape(m->data.meshes[0].shape);
                });
            }
            onStreamChanged();
            brokenTrees.clear();
            damagedTrees.clear();
            terrainDirty = placeDirty = true;
            stream::terrainChanged();
        }
    }
    if (ImGui::CollapsingHeader("Baked interiors")) {
        ImGui::Checkbox("Use baked indoor GI", &bakedGI);
        ImGui::SameLine();
        ImGui::Checkbox("Bake", &bakeEnabled);
        ImGui::SameLine();
        if (ImGui::Button("Rebake all")) {
            for (auto& b : bakeVols) for (int st = 0; st < 2; st++) { b.ready[st] = b.saved[st] = false; b.pass[st] = b.next[st] = 0; }
        }
        bool allOn = true;
        for (auto& b : bakeVols) allOn &= b.on;
        if (ImGui::Checkbox("Indoor lights (L, or E while walking, toggles the building you're in)", &allOn))
            for (auto& b : bakeVols) b.on = allOn;
        int here = volumeAt(cam.pos.x, cam.pos.y, cam.pos.z);
        for (size_t i = 0; i < bakeVols.size(); i++) {
            auto& b = bakeVols[i];
            auto prog = [&](int st) { return b.ready[st] ? 100.f : std::min(99.f, 100.f * (b.pass[st] * b.count + b.next[st]) / (float)(BAKE_PASSES * b.count)); };
            ImGui::Text("%s%s: %ux%ux%u probes @ %.0f cm, lights %s   bake off %.0f%%  on %.0f%%", (int)i == here ? "> " : "  ", b.v.name.c_str(),
                        b.dims[0], b.dims[1], b.dims[2], b.spacing * 100.f, b.on ? "ON" : "off", prog(0), prog(1));
        }

    }
    if (ImGui::CollapsingHeader("World streaming")) {
        auto ss = stream::stats();
        ImGui::Text("Cells: %d on disk, %d loaded, %d active   models %d (%d loading)", ss.cellsKnown, ss.cellsLoaded, ss.cellsActive, ss.models, ss.loading);
        ImGui::Text("Instances %u (raster %u visible, %zu draws)  meshes %u  textures %u", numModelInst, numModelVisible, drawRuns.size(),
                    meshSlots.used(), texSlots.used() - 1);
        ImGui::Text("Model GPU memory %.1f MB   physics bodies %u   terrain tiles %d", gpuModelBytes / 1048576.0, physics::bodyCount(), ss.terrainTiles);
        ImGui::Text("Lights %u / %d  (%u streamed of %zu loaded)   zones %zu", numLights, MAX_LIGHTS, numWorldLights, stream::lights().size(),
                    stream::zones().size());
        if (ImGui::SliderFloat("View distance", &S.viewDist, 300.f, 6000.f, "%.0f m", ImGuiSliderFlags_Logarithmic)) {
            loadRadius = std::clamp((int)ceilf(S.viewDist / 64.f), 1, 96);  // streamed cells follow
            terrainDirty = true;
        }
        ImGui::SetItemTooltip("How far the world is drawn: terrain, trees, rocks and streamed cells.\n"
                              "Larger costs GPU time (more trees in the BVH, coarser far terrain) and memory.");
        ImGui::SliderInt("Load radius (cells)", &loadRadius, 1, 96);
        ImGui::SliderFloat("Upload budget", &uploadBudgetMB, 8, 512, "%.0f MB/frame", ImGuiSliderFlags_Logarithmic);
    }
    if (ImGui::CollapsingHeader("Audio (ray traced)")) {
        auto as = audio::stats();
        audio::Settings& A = audio::settings();
        if (as.running) {
            ImGui::Text("Emitters %u   scene %u meshes / %u inst / %u tris", as.emitters, as.meshes, as.instances, as.sceneTris);
            ImGui::Text("Sim %.1f ms/update (%.0f Hz)   DSP %.2f ms / 21.3 ms block", as.simMs, as.simMs > 0 ? 1000.f / std::max(as.simMs, 16.f) : 0.f, as.dspMs);
            ImGui::SliderFloat("Volume", &A.masterGain, 0, 2);
            ImGui::Checkbox("Occlusion", &A.occlusion);
            ImGui::SameLine(); ImGui::Checkbox("Transmission", &A.transmission);
            ImGui::SameLine(); ImGui::Checkbox("Reflections", &A.reflections);
            ImGui::BeginDisabled(!A.reflections);
            ImGui::SliderInt("Reflection rays", &A.reflectionRays, 256, 4096);
            ImGui::SliderInt("Reflection bounces", &A.bounces, 1, 16);
            ImGui::SliderFloat("Reflection updates", &A.reflectionHz, 2, 60, "%.0f Hz");
            ImGui::SliderFloat("Reverb gain", &A.reverbGain, 0, 3);
            ImGui::EndDisabled();
        } else {
            ImGui::TextDisabled("No audio device");
        }

    }
    if (ImGui::CollapsingHeader("Controller")) {
        const input::Pad& gp = input::pad();
        if (gp.connected) {
            ImGui::Text("%s%s", gp.name, gp.hasGyro ? "  (gyro)" : "");
            ImGui::BeginDisabled(!gp.hasGyro);
            ImGui::Checkbox("Gyro aim", &gyroAim);
            ImGui::SameLine();
            ImGui::SliderFloat("##gyro", &gyroSens, 0.2f, 4.f, "sens %.1fx");
            ImGui::EndDisabled();
            ImGui::TextDisabled("L: move  R: look  A: jump  Y: walk/fly  LT/L3: sprint  LB/RB: down/up");
        } else {
            ImGui::TextDisabled("No gamepad (Steam Controller, Xbox, DualSense... plug in any time)");
        }
    }
    ImGui::End();
}

// ---------------------------------------------------------------- main
static int editDemoFrame = -1, editSaveFrame = -1;

int main(int argc, char** argv) {
    bool validate = false;
    const char* shotPath = nullptr;
    int shotFrames = 240, shotSeq = 0;
    bool benchOnly = false;
    std::string worldDir = WORLD_DIR;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--validate")) validate = true;
        else if (!strcmp(argv[i], "--shot") && i + 1 < argc) shotPath = argv[++i];
        else if (!strcmp(argv[i], "--shot-frames") && i + 1 < argc) shotFrames = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--shot-seq") && i + 1 < argc) shotSeq = atoi(argv[++i]);  // also save the next N frames (<path>.<k>)
        else if (!strcmp(argv[i], "--rays") && i + 1 < argc) S.raysFirst = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--view") && i + 1 < argc) S.viewMode = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--bench")) { benchOnly = true; vsync = false; }
        else if (!strcmp(argv[i], "--edit")) editor::setActive(true);
        else if (!strcmp(argv[i], "--edit-demo") && i + 1 < argc) editDemoFrame = atoi(argv[++i]);  // test: example house at that frame
        else if (!strcmp(argv[i], "--edit-save") && i + 1 < argc) editSaveFrame = atoi(argv[++i]);  // test: save edits at that frame
        else if (!strcmp(argv[i], "--no-restir-gi")) S.restirGI = false;
        else if (!strcmp(argv[i], "--traced-primary")) S.rasterPrimary = false;
        else if (!strcmp(argv[i], "--no-gi")) S.giEnabled = false;
        else if (!strcmp(argv[i], "--no-gi-cache")) S.giCache = false;
        else if (!strcmp(argv[i], "--no-gi-checker")) S.giChecker = false;
        else if (!strcmp(argv[i], "--no-gi-probes")) S.giProbes = false;
        else if (!strcmp(argv[i], "--sun") && i + 2 < argc) { S.sunElev = (float)atof(argv[++i]); S.sunAzim = (float)atof(argv[++i]); }
        else if (!strcmp(argv[i], "--rain") && i + 1 < argc) { S.rain = (float)atof(argv[++i]); S.wetness = std::min(S.rain * 1.5f, 1.f); }
        else if (!strcmp(argv[i], "--no-grass-refit")) S.grassRefit = false;
        else if (!strcmp(argv[i], "--no-bc7")) texcomp::setEnabled(false);
        else if (!strcmp(argv[i], "--fog-div") && i + 1 < argc) S.fogDiv = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--fog") && i + 1 < argc) { S.fogDensity = (float)atof(argv[++i]); S.fog = S.fogDensity > 0.f; }
        else if (!strcmp(argv[i], "--borderless")) windowMode = 1;
        else if (!strcmp(argv[i], "--fullscreen")) windowMode = 2;
        else if (!strcmp(argv[i], "--hdr")) S.hdrMode = 1;
        else if (!strcmp(argv[i], "--night")) { S.sunElev = -8; S.lightIntensity = 3; S.exposure = 3; }
        else if (!strcmp(argv[i], "--max-blades") && i + 1 < argc) maxBlades = (uint32_t)atoi(argv[++i]);
        else if (!strcmp(argv[i], "--ground-tex") && i + 1 < argc) groundTexSize = std::clamp((uint32_t)atoi(argv[++i]), 256u, 8192u);
        else if (!strcmp(argv[i], "--world") && i + 1 < argc) { worldDir = argv[++i]; if (worldDir.back() != '/') worldDir += '/'; }
        else if (!strcmp(argv[i], "--pos") && i + 3 < argc) { cam.pos.x = (float)atof(argv[++i]); cam.pos.y = (float)atof(argv[++i]); cam.pos.z = (float)atof(argv[++i]); }
        else if (!strcmp(argv[i], "--crater") && i + 3 < argc) {  // test: a crater at x z of radius r (repeatable)
            float x = (float)atof(argv[++i]), z = (float)atof(argv[++i]), r = (float)atof(argv[++i]);
            int n = gCraterCount.load();
            if (n < MAX_CRATERS) { gCraters[n] = Crater{x, z, r, 0.33f * r}; gCraterCount.store(n + 1); }
        }
        else if (!strcmp(argv[i], "--prof") && i + 1 < argc) profFrames = atoi(argv[++i]);  // capture N frames from frame 30, export at exit
        else if (!strcmp(argv[i], "--from-shot") && i + 1 < argc) loadShotInfo(argv[++i]);  // reopen a screenshot's scene
        else if (!strcmp(argv[i], "--replay") && i + 1 < argc) loadReplay(argv[++i]);       // drive the camera from a movement log
        else if (!strcmp(argv[i], "--replay-shots")) replayShots = true;                      // ... and retake its screenshots
        else if (!strcmp(argv[i], "--replay-exit")) replayExit = true;                        // ... and quit at its end
        else if (!strcmp(argv[i], "--ground-mode") && i + 1 < argc) S.groundMode = atoi(argv[++i]);  // 0 textures, 1 fractal, 2 hybrid
        else if (!strcmp(argv[i], "--map-test")) mapTest = true;  // test: sculpt a hill and carve a river on the world map
        else if (!strcmp(argv[i], "--map-click") && i + 1 < argc) { mapClickFrame = atoi(argv[++i]); editor::setActive(true); }  // test: raise ground through the editor UI
        else if (!strcmp(argv[i], "--snow-line") && i + 1 < argc) { S.snowLine = (float)atof(argv[++i]); S.snowFollowsClouds = false; }
        else if (!strcmp(argv[i], "--snowfall") && i + 1 < argc) S.snowfall = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--sim-speed") && i + 1 < argc) S.simSpeed = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--view-dist") && i + 1 < argc) { S.viewDist = (float)atof(argv[++i]); loadRadius = std::clamp((int)ceilf(S.viewDist / 64.f), 1, 96); }
        else if (!strcmp(argv[i], "--look") && i + 2 < argc) { cam.yaw = (float)atof(argv[++i]) * 3.14159265f / 180.f; cam.pitch = (float)atof(argv[++i]) * 3.14159265f / 180.f; }
        else if (!strcmp(argv[i], "--walk")) walkMode = true;
        else if (!strcmp(argv[i], "--dof") && i + 2 < argc) { S.dof = true; S.focusDist = (float)atof(argv[++i]); S.fStop = (float)atof(argv[++i]); }  // test: focus (m, 0 auto), f-stop
        else if (!strcmp(argv[i], "--motion-blur") && i + 1 < argc) { S.motionBlur = true; S.mbShutter = (float)atof(argv[++i]); }  // test: shutter fraction
        else if (!strcmp(argv[i], "--phys-cam") && i + 3 < argc) {  // test: f-stop, shutter 1/x s, ISO
            S.physCam = true; S.fStop = (float)atof(argv[++i]); S.shutterDen = (float)atof(argv[++i]); S.iso = (float)atof(argv[++i]);
        }
        else if (!strcmp(argv[i], "--lens-fx") && i + 3 < argc) { S.vignette = (float)atof(argv[++i]); S.chromAb = (float)atof(argv[++i]); S.grain = (float)atof(argv[++i]); }
        else if (!strcmp(argv[i], "--veg-regen") && i + 1 < argc) vegRegenFrame = atoi(argv[++i]);  // test: regrow vegetation at frame F
        else if (!strcmp(argv[i], "--throw") && i + 1 < argc) autoThrow = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--tree-throw") && i + 1 < argc) { treeThrow = atoi(argv[++i]); treeLog = true; }  // test: throw N rocks, one per 20 frames
        else if (!strcmp(argv[i], "--throw-size") && i + 1 < argc) S.throwSize = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--throw-speed") && i + 1 < argc) S.throwSpeed = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--lights-off")) startLightsOff = true;
        else if (!strcmp(argv[i], "--no-baked-gi")) bakedGI = false;
        else if (!strcmp(argv[i], "--no-async")) asyncCompute = false;
        else if (!strcmp(argv[i], "--no-async-fog")) asyncFog = false;
        else if (!strcmp(argv[i], "--character") && i + 3 < argc) {  // test: <asset> <x> <z>, then optional --character-* for it
            CharSpawn sp{argv[i + 1], "", {(float)atof(argv[i + 2]), 0.f, (float)atof(argv[i + 3])}, 0.f, 0.f, 0.f};
            charSpawns.push_back(sp);
            i += 3;
        }
        else if (!strcmp(argv[i], "--character-clip") && i + 1 < argc && !charSpawns.empty()) charSpawns.back().clip = argv[++i];
        else if (!strcmp(argv[i], "--character-walk") && i + 1 < argc && !charSpawns.empty()) charSpawns.back().walk = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--character-yaw") && i + 1 < argc && !charSpawns.empty()) charSpawns.back().yaw = (float)atof(argv[++i]) * 3.14159265f / 180.f;
        else if (!strcmp(argv[i], "--character-height") && i + 1 < argc && !charSpawns.empty()) charSpawns.back().height = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--ragdoll-at") && i + 1 < argc) ragdollFrame = atoi(argv[++i]);  // test: every character goes limp at that frame
        else if (!strcmp(argv[i], "--dlss") && i + 1 < argc) upscaler = (dlss::Mode)std::clamp(atoi(argv[++i]), 0, (int)dlss::MODE_COUNT - 1);
    }

    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    {   // window: the screen's size (maximised), borderless or fullscreen on request
        GLFWmonitor* mon = glfwGetPrimaryMonitor();
        const GLFWvidmode* vm = mon ? glfwGetVideoMode(mon) : nullptr;
        int ww = vm ? vm->width : 1600, wh = vm ? vm->height : 900;
        if (windowMode == 1) glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
        if (windowMode == 0) glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
        window = glfwCreateWindow(ww, wh, "ptx - path traced grass", windowMode == 2 ? mon : nullptr, nullptr);
        if (shotPath) resFollowWindow = false;  // screenshots / tests: fixed 1080p
        else { OW = (uint32_t)ww; OH = (uint32_t)wh; }
    }
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow*, int, int) { swapDirty = true; });
    glfwSetDropCallback(window, [](GLFWwindow*, int n, const char** paths) { editor::dropFiles(std::vector<std::string>(paths, paths + n)); });

    initVulkan(validate);
    createSwapchain();
    if (resFollowWindow && swapExtent.width >= 64) { OW = swapExtent.width; OH = swapExtent.height; resOptions[4][0] = OW; resOptions[4][1] = OH; }
    loadForest(worldDir);  // before buildScene: the impostor atlas has a layer per tree variant
    buildScene();
    createLayoutAndPipelines();
    createScreenResources();
    createAccels();
    writeDescriptors();
    initTextureSlots();
    unsigned hw = std::max(4u, std::thread::hardware_concurrency());
    jobs::init(hw - 2);            // shared workers: Jolt jobs + parallel CPU work (2 cores left for loaders)
    loadGroundTextures();
    physics::init();
    texcomp::init();
    physics::setSurfaceFn(groundSurfaceAt);
    input::init();
    audio::init();
    stream::init(worldDir, 2);     // loader threads (parse, decode, collision shapes)
    worldmap::init(worldDir);
    if (!replaying) openMoveLog();
    if (mapTest) {
        worldmap::beginStroke();
        for (int k = 0; k < 400; k++) worldmap::brush(worldmap::RAISE, 40.f, 60.f, 45.f, 10.f, 0.05f, 0, 0);  // ~20 m hill
        for (int k = 0; k < 30; k++) worldmap::brush(worldmap::SMOOTH, 40.f, 60.f, 50.f, 5.f, 0.05f, 0, 0);
        worldmap::addRiver({{60.f, 30.f, 3.f, 1.f}, {90.f, 20.f, 4.f, 1.5f}, {120.f, 35.f, 5.f, 2.f}, {160.f, 20.f, 6.f, 2.f}});
    }
    {
        editor::Host eh;
        eh.worldDir = worldDir;
        eh.upload = uploadModel;
        eh.retire = [](std::shared_ptr<Model> m) {
            retireNext.push_back([m]() {
                if (m->gpu) releaseModel(*m);
                for (auto& md : m->data.meshes) if (md.shape) { physics::releaseShape(md.shape); md.shape = nullptr; }
            });
        };
        eh.height = [](float x, float z) { return heightFn(x, z); };
        eh.snowBrush = [](float x, float z, float r, float rate) {
            gSimBrush[0] = x; gSimBrush[1] = z; gSimBrush[2] = r; gSimBrush[3] = rate;
            if (r > 0.f) simActiveFrames = 900;  // keep the terrain mesh following the pile (and its avalanches)
        };
        eh.thumbnail = [](const uint8_t* px, int w, int h) -> uint64_t {
            Image im = mkTexture((uint32_t)w, (uint32_t)h, 1, VK_FORMAT_R8G8B8A8_UNORM);
            Buffer st = mkBuf((VkDeviceSize)w * h * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
            memcpy(st.map, px, (size_t)w * h * 4);
            oneShot([&](VkCommandBuffer cb) { recordTextureUpload(cb, im, (uint32_t)w, (uint32_t)h, 1, st.b, 0); });
            freeBuf(st);
            return (uint64_t)ImGui_ImplVulkan_AddTexture(texSampler, im.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL);
        };
        editor::init(eh);
    }
    charWorldDir = worldDir;
    character::init([](float x, float z) { return heightFn(x, z); });
    bakeDir = worldDir + "bake/";
    initTrees();
    if (walkMode) {
        V3 feet = cam.pos - V3{0, 1.65f, 0};
        feet.y = std::max(feet.y, heightFn(feet.x, feet.z) + 0.05f);
        physics::spawnPlayer(&feet.x);
    }
    prevCam = cam;
    bladeCap = maxBlades;

    // ImGui
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForVulkan(window, true);
    VkDescriptorPoolSize ips{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 1024};  // font + asset browser thumbnails
    VkDescriptorPoolCreateInfo ipci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    ipci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    ipci.maxSets = 1024;
    ipci.poolSizeCount = 1;
    ipci.pPoolSizes = &ips;
    VK_CHECK(vkCreateDescriptorPool(dev, &ipci, nullptr, &imguiPool));
    ImGui_ImplVulkan_InitInfo ii{};
    ii.Instance = instance;
    ii.PhysicalDevice = phys;
    ii.Device = dev;
    ii.QueueFamily = qfam;
    ii.Queue = queue;
    ii.DescriptorPool = imguiPool;
    ii.MinImageCount = 2;
    ii.ImageCount = (uint32_t)swapImages.size();
    ii.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    ii.UseDynamicRendering = true;
    ii.PipelineRenderingCreateInfo = {VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO_KHR};
    ii.PipelineRenderingCreateInfo.colorAttachmentCount = 1;
    ii.PipelineRenderingCreateInfo.pColorAttachmentFormats = &uiFmt;
    ImGui_ImplVulkan_Init(&ii);

    if (benchOnly) startBenchmark();
    auto last = std::chrono::steady_clock::now();
    prof::setThreadName("main");
    bool profOpen = false;
    if (profFrames > 0) prof::setCaptureFrames(profFrames);
    while (!glfwWindowShouldClose(window)) {
        if (profFrames > 0 && frameIndex == 30) prof::toggleCapture();
        prof::frameBegin();
        PROF_ZONE("frame");
        if (float sf = prof::slowFps(); sf > 0.f) {  // slow motion (Profiler window)
            PROF_ZONE("slow motion sleep");
            static auto prevT = std::chrono::steady_clock::now();
            auto target = prevT + std::chrono::microseconds((int64_t)(1e6f / sf));
            std::this_thread::sleep_until(target);
            prevT = std::chrono::steady_clock::now();
        }
        {
            PROF_ZONE("poll events");
            glfwPollEvents();
        }
        input::update();
        int fw, fh;
        glfwGetFramebufferSize(window, &fw, &fh);
        if (fw == 0 || fh == 0) { glfwWaitEvents(); continue; }
        if (pendingWindowMode >= 0) { applyWindowMode(pendingWindowMode); pendingWindowMode = -1; }
        if (swapDirty) {
            vkDeviceWaitIdle(dev);
            createSwapchain();
            ImGui_ImplVulkan_SetMinImageCount(2);
            swapDirty = false;
            // output resolution follows the window
            if (resFollowWindow && swapExtent.width >= 64 && swapExtent.height >= 64 && (swapExtent.width != OW || swapExtent.height != OH)) {
                resOptions[4][0] = swapExtent.width;
                resOptions[4][1] = swapExtent.height;
                pendingRes = 4;
            }
        }

        auto now = std::chrono::steady_clock::now();
        float dt = std::chrono::duration<float>(now - last).count();
        frameDt = dt;
        // snow changes the ground shape: refresh the terrain mesh every few frames while the simulation runs
        if (S.worldSim && S.snowMeshEvery > 0 && frameIndex % (uint32_t)S.snowMeshEvery == 0 && (S.rain > 0.f || gSimBrush[2] > 0.f || simActiveFrames > 0)) {
            terrainDirty = true;
            if (simActiveFrames > 0) simActiveFrames--;
        }
        last = now;
        cpuFrameMs = cpuFrameMs * 0.9f + dt * 1000.f * 0.1f;
        if (replaying && replayAt < replayFrames.size()) dt = replayFrames[replayAt].dt;  // the recorded step: same physics

        {
            PROF_ZONE("wait GPU");
            VK_CHECK(vkWaitForFences(dev, 1, &frameFence, VK_TRUE, UINT64_MAX));
        }
        graph.begin();
        if (shotUIPending && haveTimestamps) {  // last frame's swapchain copy (with the UI) is complete
            std::string p = newShotPath();
            shotDone(p, writePng(p, (const uint8_t*)uiShotBuf.map, uiShotW, uiShotH, swapBGR));
            shotUIPending = false;
        }
        if (S.worldSim && simRB.map && haveTimestamps) {  // last frame's readback is complete: snapshot it for the physics
            const uint16_t* h = (const uint16_t*)simRB.map;
            for (int i = 0; i < SIM_RB * SIM_RB; i++) { simSnap[i][0] = halfToFloat(h[i * 4]); simSnap[i][1] = halfToFloat(h[i * 4 + 2]); }
            simSnapX = simRBx; simSnapZ = simRBz;
            simSnapValid = true;
        } else simSnapValid = false;
        if (haveTimestamps && prof::capturing) {
            std::vector<prof::GpuPass> gp;
            for (auto& t : graph.timings()) gp.push_back({t.name, t.queue == rg::COMPUTE, t.startMs, t.ms});
            prof::gpuFrame(gp);
        }
        if (haveTimestamps) {
            readTimestamps();
            // size this frame's grass BLAS from last frame's blade count, with headroom
            lastBladeCount = *(const uint32_t*)countReadback.map;
            bladeCap = std::min(maxBlades, std::max(8192u, lastBladeCount + lastBladeCount / 4 + 4096));
        }
        // previous frame is done: free its staging/scratch, then stream (uploads recorded into uploadCmd)
        for (auto& f : retireNext) f();
        retireNext.clear();
        uploadsRecorded = false;
        frameUploadBytes = 0;
        if ((int)frameIndex == vegRegenFrame) { trees::treePreset(5, trees::treeParams[0]); vegDirty = true; }
        if (vegDirty) { vegDirty = false; regenVegetation(); }
        if (uint64_t env = bakeEnvHash(); env != bakeEnv) { PROF_ZONE("rebake volumes"); bakeEnv = env; rebuildBakeVolumes(true); }  // sun / sky moved: rebake
        saveFinishedBakes();
        if ((int)frameIndex == editDemoFrame) editor::demoHouse();
        if ((int)frameIndex == editSaveFrame) { std::string err; if (!editor::saveAll(err)) fprintf(stderr, "editor: %s\n", err.c_str()); }
        { PROF_ZONE("editor apply"); editor::apply(); }  // edited cells go live in this upload window
        {
            PROF_ZONE("stream update");
            if (stream::update(cam.pos.x, cam.pos.y, cam.pos.z, loadRadius, uploadModel, releaseModel)) onStreamChanged(editor::takeMoveOnly());
        }
        { PROF_ZONE("fractures"); processFractures(); }  // fragment meshes upload in this frame's upload window
        { PROF_ZONE("world map upload"); uploadWorldMap(); }
        if (!charSpawns.empty()) { PROF_ZONE("spawn characters"); spawnCharacters(); }
        if (uploadsRecorded) vkEndCommandBuffer(uploadCmd);
        for (uint32_t id : audioMeshRetire) audio::meshRelease(id);
        audioMeshRetire.clear();

        {
        PROF_ZONE("UI");
        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        if (mapClickFrame >= 0) {  // synthetic input after the backend's, so it wins this frame
            ImGuiIO& io = ImGui::GetIO();
            int f = (int)frameIndex - mapClickFrame;
            if (f == 0 || f == 1) io.AddKeyEvent(ImGuiKey_M, f == 0);
            if (f >= 0) io.AddMousePosEvent(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.6f);
            if (f == 5 || f == 305) io.AddMouseButtonEvent(0, f == 5);
        }
        ImGui::NewFrame();
        drawUI();
        {
            static bool f9 = false;
            bool k = glfwGetKey(window, GLFW_KEY_F9) == GLFW_PRESS;
            if (k && !f9) { prof::toggleCapture(); profOpen = true; }
            f9 = k;
            if (ImGui::IsKeyPressed(ImGuiKey_F8, false)) profOpen = !profOpen;
            // controls cheat sheet (F1)
            static bool helpOpen = false;
            if (ImGui::IsKeyPressed(ImGuiKey_F1, false)) helpOpen = !helpOpen;
            {
                ImGuiIO& io = ImGui::GetIO();
                const char* hint = "F1: controls";
                ImVec2 ts = ImGui::CalcTextSize(hint);
                ImGui::GetForegroundDrawList()->AddText(ImVec2(io.DisplaySize.x - ts.x - 10, 6), IM_COL32(255, 255, 255, 140), hint);
            }
            if (helpOpen) {
                ImGuiIO& io = ImGui::GetIO();
                ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f), ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
                ImGui::SetNextWindowSize(ImVec2(620, 640), ImGuiCond_FirstUseEver);
                ImGui::SetNextWindowSizeConstraints(ImVec2(300, 200), ImVec2(FLT_MAX, std::max(200.f, io.DisplaySize.y - 20.f)));
                if (ImGui::Begin("Controls (F1)", &helpOpen)) {
                    auto section = [](const char* t) { ImGui::SeparatorText(t); };
                    auto row = [](const char* k, const char* what) {
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn(); ImGui::TextColored(ImVec4(1.f, 0.85f, 0.4f, 1), "%s", k);
                        ImGui::TableNextColumn(); ImGui::TextWrapped("%s", what);
                    };
                    auto table = [](const char* id) {
                        bool ok = ImGui::BeginTable(id, 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp);
                        if (ok) { ImGui::TableSetupColumn("key", ImGuiTableColumnFlags_WidthFixed, 170); ImGui::TableSetupColumn("action"); }
                        return ok;
                    };
                    section("Moving");
                    if (table("move")) {
                        row("Right mouse + move", "Look around");
                        row("W A S D", "Move");
                        row("E / Q", "Fly up / down (E switches lights while walking)");
                        row("Shift / Ctrl", "Faster / turbo (fly)");
                        row("F", "Toggle walking / flying");
                        row("Space", "Jump (walking)");
                        row("Gamepad", "Sticks move / look, RT throw, A jump, Y walk / fly, LB / RB down / up, LT or L3 sprint");
                        ImGui::EndTable();
                    }
                    section("World");
                    if (table("world")) {
                        row("G / Left mouse", "Throw a rock (size and speed: Interaction panel)");
                        row("L", "Lights of the building you're in on / off");
                        row("P", "Primary rays: raster G-buffer / path traced");
                        row("I", "Global illumination on / off");
                        ImGui::EndTable();
                    }
                    section("Editor (Tab)");
                    if (table("edit")) {
                        row("Tab", "Object editor on / off");
                        row("Click / drag", "Select / move on the ground (Alt+drag: up / down)");
                        row("R / Shift+R", "Rotate by the step");
                        row("Arrows, PgUp / PgDn", "Nudge by the grid");
                        row("N", "Grid snap on / off (hold Shift to invert)");
                        row("Ctrl+D / Del", "Duplicate / delete");
                        row("Ctrl+Z / Ctrl+S", "Undo / save (cells and the world map)");
                        row("Drag from Assets", "Place an asset where you release it; drop files on the window to import");
                        ImGui::EndTable();
                    }
                    section("World map (M, inside the editor)");
                    if (table("map")) {
                        row("M", "World map mode on / off: left mouse paints with the tool");
                        row("Left mouse (hold)", "Raise / lower / smooth / flatten / roughen, paint grass, trees, textures, snow");
                        row("Ctrl+click", "Flatten: pick the height under the cursor");
                        row("River: click / drag", "Draw the course (water runs from the first point)");
                        row("River: wheel", "Wider / narrower from here on");
                        row("River: Shift+wheel", "Deeper / shallower");
                        row("River: Enter / Bksp / Esc", "Carve it / remove last point / cancel");
                        row("Ctrl+Z", "Undo the last stroke");
                        ImGui::EndTable();
                    }
                    section("Capture and debug");
                    if (table("dbg")) {
                        row("F12", "Screenshot (scene) -> screenshots/*.png + .txt scene, path copied");
                        row("Shift+F12", "Screenshot with the UI");
                        row("F9", "Profiler: capture frames (flame chart)");
                        row("F8", "Profiler window");
                        row("F1", "This list");
                        ImGui::EndTable();
                    }
                    ImGui::TextDisabled("Reopen a screenshot's scene: ./build/ptx --from-shot <file.png>");
                }
                ImGui::End();
            }
            // screenshots: F12 the scene, Shift+F12 with the UI (falls back to the scene where the swapchain can't be read)
            static bool f12 = false;
            bool k12 = glfwGetKey(window, GLFW_KEY_F12) == GLFW_PRESS;
            if (k12 && !f12) {
                if (ImGui::GetIO().KeyShift && swapCanCopy && !hdrActive) shotUIReq = true;
                else shotSceneReq = true;
            }
            f12 = k12;
            if (glfwGetTime() - shotMsgT < 4.0) {
                ImGuiIO& io = ImGui::GetIO();
                ImGui::GetForegroundDrawList()->AddRectFilled(ImVec2(10, io.DisplaySize.y - 34), ImVec2(io.DisplaySize.x * 0.75f, io.DisplaySize.y - 10),
                                                               IM_COL32(0, 0, 0, 170), 4.f);
                ImGui::GetForegroundDrawList()->AddText(ImVec2(16, io.DisplaySize.y - 30), IM_COL32(255, 255, 255, 255), shotMsg.c_str());
            }
            if (profOpen) prof::ui(&profOpen);
        }
        {   // world editor: Tab toggles
            static bool tabPrev = false;
            bool tab = !ImGui::GetIO().WantCaptureKeyboard && glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS;
            if (tab && !tabPrev) editor::setActive(!editor::active());
            tabPrev = tab;
            V3 f = cam.fwd(), r = cam.right(), u = cam.up();
            editor::View ev{{cam.pos.x, cam.pos.y, cam.pos.z}, {f.x, f.y, f.z}, {r.x, r.y, r.z}, {u.x, u.y, u.z}, tanf(cam.fovDeg * 0.5f * 3.14159265f / 180.f)};
            editor::ui(ev);
        }
        {   // light switch for the building you're in: E while walking (E flies up otherwise), L any time
            static bool prevKey = false;
            bool k = !ImGui::GetIO().WantCaptureKeyboard &&
                     ((walkMode && glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) || glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS);
            if (k && !prevKey)
                if (int v = volumeAt(cam.pos.x, cam.pos.y, cam.pos.z); v >= 0) bakeVols[v].on = !bakeVols[v].on;
            prevKey = k;
        }
        ImGui::Render();
        }

        if (pendingRes >= 0) {
            vkDeviceWaitIdle(dev);
            destroyScreenResources();
            OW = resOptions[pendingRes][0];
            OH = resOptions[pendingRes][1];
            createScreenResources();
            writeDescriptors();
            pendingRes = -1;
        }

        prevCam = cam;
        if (replaying) {  // the camera follows the log; its events fire on their frames
            if (replayAt < replayFrames.size()) {
                const MoveFrame& m = replayFrames[replayAt];
                cam.pos = {m.x, m.y, m.z};
                cam.yaw = m.yaw * 3.14159265f / 180.f;
                cam.pitch = m.pitch * 3.14159265f / 180.f;
                simTime = m.t;
                walkMode = false;  // the log already holds where walking put the camera
                while (replayEvAt < replayEvents.size() && replayEvents[replayEvAt].f <= m.f) {
                    const MoveEvent& e = replayEvents[replayEvAt++];
                    if (e.kind == "throw") { float s0 = S.throwSize, v0 = S.throwSpeed; S.throwSize = e.a; S.throwSpeed = e.b; throwRockImpl(); S.throwSize = s0; S.throwSpeed = v0; }
                    else if (e.kind == "shot" && replayShots) shotSceneReq = true;
                }
                replayAt++;
            } else {
                replaying = false;
                printf("replay finished\n");
                if (replayExit) glfwSetWindowShouldClose(window, 1);
            }
        } else if (!bench.running) {
            updateCamera(std::min(dt, 0.1f));
            simTime += std::min(dt, 0.1f);
        }
        moveLogFrame(dt);
        if (autoThrow > 0 && frameIndex >= 30 && frameIndex % 20 == 0) { throwRock(); autoThrow--; }
        if (treeThrow > 0 && frameIndex >= 60 && frameIndex % 40 == 0 && !treeList.empty()) {
            const TreeInst* tb = nullptr;
            for (const TreeInst& tr : treeList)
                if (tr.v < (uint32_t)trees::VARIANTS && (!tb || dot(tr.p - cam.pos, tr.p - cam.pos) < dot(tb->p - cam.pos, tb->p - cam.pos))) tb = &tr;
            if (tb) {
                // a limb 3-10 cm thick (most of the throws), now and then the trunk
                trees::Wood w;
                trees::woodWhole((int)tb->v, w);
                WoodFrame F = treeFrame(*tb);
                V3 tgt = tb->p + V3{0, 1.2f * tb->scale, 0};
                float a[3], b[3], r;
                for (int k = (int)(hash2(treeThrow, 5) * 400); treeThrow % 4 && k < 4000; k++)
                    if (trees::woodElem(w, k % 1000, a, b, r) && r * tb->scale > 0.03f && r * tb->scale < 0.1f && k % 1000 >= trees::TRUNK_ELEMS) {
                        tgt = woodToWorld(F, (V3{a[0], a[1], a[2]} + V3{b[0], b[1], b[2]}) * 0.5f);
                        break;
                    }
                V3 p = tgt + V3{-6.f, 0.5f, 0}, d = normalize(tgt - p);
                cam.pos = p;
                cam.yaw = atan2f(d.x, d.z);
                cam.pitch = asinf(d.y);
                throwRock();
                treeThrow--;
            }
        }
        // this frame's poses become "previous" before anything moves (per-object motion vectors)
        for (Thrown& t : thrown) { t.prevXf = t.xf; t.hasPrev = true; }
        for (Droplet& d : spray) { d.prevP = d.p; d.hasPrev = true; }
        for (character::Character* c : character::all())
            if (auto* gc = (GpuChar*)c->user) { gc->prevW = character::worldXf(*c); gc->hasPrev = true; }
        if (!bench.running) {
            PROF_ZONE("physics");
            { PROF_ZONE("thrown + water forces"); updateThrown(dt); }
            { PROF_ZONE("spray"); updateSpray(std::min(dt, 0.05f)); }
            { PROF_ZONE("crater water"); updateRiverSwell(std::min(dt, 0.1f)); updateCraterWater(std::min(dt, 0.1f)); }
            { PROF_ZONE("characters: animate"); character::preStep(std::min(dt, 0.1f)); }
            { PROF_ZONE("physics step (Jolt)"); physics::step(std::min(dt, 1.f / 30.f)); }
            { PROF_ZONE("characters: pose"); character::postStep(std::min(dt, 0.1f)); }
            if (ragdollFrame >= 0 && (int)frameIndex == ragdollFrame)
                for (character::Character* c : character::all()) { float j[3] = {0, 0, 0}; character::ragdoll(*c, j, c->pos); }
        }
        benchStep();
        autoScale();
        { PROF_ZONE("frame data + placement"); updateFrameData(); }
        { PROF_ZONE("write thrown"); writeThrown(); }
        { PROF_ZONE("write characters"); writeCharacters(); }
        { PROF_ZONE("audio"); updateAudio(std::min(dt, 0.1f)); }
        { PROF_ZONE("cull models"); cullModels(); }
        { PROF_ZONE("cull trees"); cullTrees(); }
        { PROF_ZONE("tree colliders"); syncTreeBodies(); }
        { PROF_ZONE("rock colliders"); syncRockBodies(); }

        uint32_t si;
        VkResult ar;
        { PROF_ZONE("acquire swapchain"); ar = vkAcquireNextImageKHR(dev, swapchain, UINT64_MAX, acquireSem, VK_NULL_HANDLE, &si); }
        if (ar == VK_ERROR_OUT_OF_DATE_KHR) {
            swapDirty = true;
            if (uploadsRecorded) {  // don't drop recorded uploads: run them now
                VkSubmitInfo us{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                us.commandBufferCount = 1;
                us.pCommandBuffers = &uploadCmd;
                VK_CHECK(vkQueueSubmit(queue, 1, &us, VK_NULL_HANDLE));
                vkQueueWaitIdle(queue);
            }
            continue;
        }
        VK_CHECK(vkResetFences(dev, 1, &frameFence));

        { PROF_ZONE("build frame graph"); buildFrameGraph(si, ImGui::GetDrawData()); }
        std::vector<VkCommandBuffer> first;
        if (uploadsRecorded) first.push_back(uploadCmd);
        {
            PROF_ZONE("record + submit");
            graph.execute(first, acquireSem, VK_PIPELINE_STAGE_2_TRANSFER_BIT | VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT, "present",
                          renderDone[si], frameFence);
        }
        haveTimestamps = true;
        frameIndex++;

        VkPresentInfoKHR pi{VK_STRUCTURE_TYPE_PRESENT_INFO_KHR};
        pi.waitSemaphoreCount = 1;
        pi.pWaitSemaphores = &renderDone[si];
        pi.swapchainCount = 1;
        pi.pSwapchains = &swapchain;
        pi.pImageIndices = &si;
        VkResult pr;
        { PROF_ZONE("present"); pr = vkQueuePresentKHR(queue, &pi); }
        FrameMark;
        if (shotSceneReq) { shotSceneReq = false; saveScenePng(); }
        if (pr == VK_ERROR_OUT_OF_DATE_KHR || pr == VK_SUBOPTIMAL_KHR) swapDirty = true;
        if (benchOnly && bench.done) {
            printf("%s @ %ux%u -> %ux%u (%s), ~%u blades/frame\n", gpuName, W, H, OW, OH, rrActive ? dlss::modeName(upscaler) : "native", lastBladeCount);
            for (int t = 1; t <= 4; t++)
                printf("  %d ray(s)/bounce (%2d rpp): avg %.2f ms  p95 %.2f ms  -> %.0f fps\n", t, raysPerPixel(t, 2), bench.avg[t], bench.p95[t], 1000.f / bench.p95[t]);
            for (int fps : {60, 40, 30, 25}) printf("  target %d fps -> %d ray(s)/bounce\n", fps, benchBestTier(fps));
            break;
        }
        if (shotPath && shotSeq > 0 && (int)frameIndex > shotFrames - shotSeq && (int)frameIndex < shotFrames) {
            vkDeviceWaitIdle(dev);
            saveShot((std::string(shotPath) + "." + std::to_string(shotFrames - (int)frameIndex)).c_str());
        }
        if (shotPath && (int)frameIndex == shotFrames) {
            vkDeviceWaitIdle(dev);
            for (int i = 0; i < std::min(gCraterCount.load(), MAX_CRATERS); i++)
                printf("crater %d: r %.1f m at (%.0f, %.0f)%s fill %.2f\n", i, gCraters[i].r, gCraters[i].x, gCraters[i].z,
                       craterWater[i].pond ? " pond" : "", craterWater[i].fill);
            graph.begin();
            readTimestamps();
            saveShot(shotPath);
            break;
        }

        // frame limiter for the fps target
        if (targetFps > 0 && !bench.running) {
            auto end = now + std::chrono::duration_cast<std::chrono::steady_clock::duration>(std::chrono::duration<double>(1.0 / targetFps));
            while (std::chrono::steady_clock::now() < end) {
                auto left = end - std::chrono::steady_clock::now();
                if (left > std::chrono::milliseconds(2)) std::this_thread::sleep_for(left - std::chrono::milliseconds(1));
            }
        }
    }

    vkDeviceWaitIdle(dev);
    graph.destroy();
    destroyScreenResources();
    dlss::shutdown();
    {
        size_t n = 0;
        vkGetPipelineCacheData(dev, pipeCache, &n, nullptr);
        std::vector<char> blob(n);
        if (n && vkGetPipelineCacheData(dev, pipeCache, &n, blob.data()) == VK_SUCCESS)
            std::ofstream(PIPE_CACHE_FILE, std::ios::binary).write(blob.data(), n);
    }
    if (profFrames > 0 && prof::hasCapture()) printf("trace: %s\n", prof::exportTrace().c_str());
    // Quit now. Everything left is memory and handles the OS reclaims, and an orderly teardown can stall for seconds
    // (a loader thread mid-way through a 400 MB model, the audio simulation, the editor's import worker): the window
    // has to close when it's told to.
    if (moveLog) fclose(moveLog);
    glfwDestroyWindow(window);
    fflush(stdout);
    fflush(stderr);
    _exit(0);
    stream::shutdown();
    audio::shutdown();
    physics::shutdown();
    jobs::shutdown();
    input::shutdown();
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    glfwDestroyWindow(window);
    glfwTerminate();
    return 0;
}

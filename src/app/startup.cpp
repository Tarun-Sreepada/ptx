#include "app/startup.h"

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <thread>
#include <vector>

#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include "app/window.h"
#include "audio/audio.h"
#include "characters/character.h"
#include "core/input.h"
#include "core/jobs.h"
#include "core/math.h"
#include "core/prof.h"
#include "destruction/craters.h"
#include "editor/editor.h"
#include "gpu/context.h"
#include "gpu/descriptors.h"
#include "gpu/memory.h"
#include "gpu/pipelines.h"
#include "gpu/swapchain.h"
#include "gpu/textures.h"
#include "physics/physics.h"
#include "render/bake.h"
#include "render/targets.h"
#include "scene/camera.h"
#include "scene/characters.h"
#include "scene/forest.h"
#include "scene/grass.h"
#include "scene/models.h"
#include "scene/scene_gpu.h"
#include "scene/vegetation.h"
#include "sim/world_sim.h"
#include "tools/bench.h"
#include "tools/movelog.h"
#include "world/ground_surface.h"
#include "world/terrain_query.h"
#include "world/texcomp.h"
#include "world/world.h"
#include "world/worldmap.h"

namespace {
constexpr int MIN_SWAP_EXTENT = 64;          // smaller (minimised) surfaces keep the default output resolution
constexpr float EYE_HEIGHT = 1.65f;          // walking: eye above the feet (m)
constexpr float SPAWN_CLEARANCE = 0.05f;     // feet above the ground when (re)spawning
constexpr uint32_t IMGUI_DESCRIPTORS = 1024; // font + asset browser thumbnails
constexpr int SNOW_BRUSH_SETTLE_FRAMES = 900;  // keep re-meshing the terrain this long after snow was painted

// Wall-clock time of each startup stage, printed as one line at the end.
struct StageTimer {
    using clock = std::chrono::steady_clock;
    clock::time_point t0 = clock::now(), last = t0;
    std::string line;
    void stage(const char* name) {
        auto now = clock::now();
        char buf[64];
        snprintf(buf, sizeof buf, "%s %.2f  ", name, std::chrono::duration<float>(now - last).count());
        line += buf;
        last = now;
    }
    void print() {
        printf("startup %.2f s on %u threads: %s\n", std::chrono::duration<float>(clock::now() - t0).count(), jobs::threadCount() + 1,
               line.c_str());
    }
};

// The editor's view of the renderer: upload / retire models, terrain height, thumbnails, the snow brush.
void initEditor(const std::string& worldDir) {
    editor::Host eh;
    eh.worldDir = worldDir;
    eh.upload = uploadModel;
    eh.retire = [](std::shared_ptr<Model> m) {
        retireNext.push_back([m]() {
            if (m->gpu) releaseModel(*m);
            for (auto& md : m->data.meshes)
                if (md.shape) { physics::releaseShape(md.shape); md.shape = nullptr; }
        });
    };
    eh.height = [](float x, float z) { return heightFn(x, z); };
    eh.snowBrush = [](float x, float z, float r, float rate) {
        gSimBrush[0] = x; gSimBrush[1] = z; gSimBrush[2] = r; gSimBrush[3] = rate;
        if (r > 0.f) simActiveFrames = SNOW_BRUSH_SETTLE_FRAMES;  // keep the terrain mesh following the pile (and its avalanches)
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

void initImGui() {
    ImGui::CreateContext();
    ImGui::StyleColorsDark();
    ImGui_ImplGlfw_InitForVulkan(window, true);
    VkDescriptorPoolSize ips{VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, IMGUI_DESCRIPTORS};
    VkDescriptorPoolCreateInfo ipci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    ipci.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
    ipci.maxSets = IMGUI_DESCRIPTORS;
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
}

// --map-test: a ~20 m hill and a carved river, through the same brushes the editor uses
void mapTestEdits() {
    worldmap::beginStroke();
    for (int k = 0; k < 400; k++) worldmap::brush(worldmap::RAISE, 40.f, 60.f, 45.f, 10.f, 0.05f, 0, 0);
    for (int k = 0; k < 30; k++) worldmap::brush(worldmap::SMOOTH, 40.f, 60.f, 50.f, 5.f, 0.05f, 0, 0);
    worldmap::addRiver({{60.f, 30.f, 3.f, 1.f}, {90.f, 20.f, 4.f, 1.5f}, {120.f, 35.f, 5.f, 2.f}, {160.f, 20.f, 6.f, 2.f}});
}
}  // namespace

void startup(const LaunchOptions& o) {
    StageTimer T;
    // all cores but one: the OS, the audio device and the compositor keep a core to themselves
    unsigned hw = std::max(2u, std::thread::hardware_concurrency());
    jobs::init(hw - 1);

    // ---- CPU work that needs no GPU: runs on the pool while the device comes up
    jobs::Async forest([&] { loadForest(o.worldDir); });                   // forest.txt assets (glTF + textures)
    jobs::Async vegetation([] { generateVegetation(); });                   // procedural trees + plants
    jobs::Async geometry([] { prepareSceneGeometry(); });                   // rock meshes, index lists
    jobs::Async physicsInit([] { physics::init(); texcomp::init(); });       // Jolt, BC7 encoder tables
    jobs::Async audioInit([] { audio::init(); });                           // device, Steam Audio, HRTF, synth sounds
    jobs::Async mapInit([&] { worldmap::init(o.worldDir); });               // terrain/map.bin
    jobs::Async editorInit([&] { initEditor(o.worldDir); });                // material library, asset list

    // ---- main thread: window, device, swapchain; pipelines compile on the pool meanwhile
    createWindow(o.shotPath != nullptr);
    T.stage("window");
    initVulkan(o.validate);
    T.stage("vulkan");
    createSwapchain();
    if (resFollowWindow && swapExtent.width >= MIN_SWAP_EXTENT) {
        OW = swapExtent.width; OH = swapExtent.height;
        resOptions[4][0] = OW; resOptions[4][1] = OH;
    }
    createPipelineLayout();
    jobs::Async pipelines([] { createPipelines(); });
    forest.wait();  // the impostor atlas has a layer per tree variant
    geometry.wait();
    T.stage("loading");
    buildScene();
    createScreenResources();
    createAccels();
    writeDescriptors();
    initTextureSlots();
    T.stage("gpu resources");

    physicsInit.wait();
    physics::setSurfaceFn(groundSurfaceAt);
    input::init();
    mapInit.wait();  // the stream's loader threads sample the world map heights (collision tiles)
    if (mapTest) mapTestEdits();
    settleCraters();  // --crater / saved scenes: their impact planes
    unsigned loaders = std::max(2u, hw / 3);  // stream loader threads: parse, decode, collision shapes
    stream::init(o.worldDir, (int)loaders);
    charWorldDir = o.worldDir;
    character::init([](float x, float z) { return heightFn(x, z); });
    bakeDir = o.worldDir + "bake/";

    vegetation.wait();
    pipelines.wait();  // the impostor bake runs a compute pipeline
    audioInit.wait();  // model uploads create acoustic meshes
    T.stage("pipelines + vegetation");
    initTrees();
    T.stage("trees");
    if (walkMode) {
        V3 feet = cam.pos - V3{0, EYE_HEIGHT, 0};
        feet.y = std::max(feet.y, heightFn(feet.x, feet.z) + SPAWN_CLEARANCE);
        physics::spawnPlayer(&feet.x);
    }
    prevCam = cam;
    bladeCap = maxBlades;
    initImGui();
    editorInit.wait();
    if (!replaying) openMoveLog();
    if (o.benchOnly) startBenchmark();
    T.stage("rest");
    T.print();
}

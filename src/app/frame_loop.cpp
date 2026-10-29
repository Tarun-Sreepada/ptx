#include "app/frame_loop.h"
#include "sim/critters.h"
#include "sim/wildlife.h"
#include "sim/fire.h"
#include "sky/day_cycle.h"
#include "sky/weather_cycle.h"
#include "app/settings_io.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>

#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <imgui_impl_glfw.h>
#include <imgui_impl_vulkan.h>

#include "app/frame_state.h"
#include "app/settings.h"
#include "app/window.h"
#include "audio/audio.h"
#include "characters/character.h"
#include "core/half.h"
#include "core/input.h"
#include "core/prof.h"
#include "destruction/fracture_queue.h"
#include "destruction/throw.h"
#include "destruction/thrown.h"
#include "editor/editor.h"
#include "gpu/context.h"
#include "gpu/descriptors.h"
#include "gpu/memory.h"
#include "gpu/swapchain.h"
#include "render/bake.h"
#include "render/culling.h"
#include "render/culling_vegetation.h"
#include "render/frame_data.h"
#include "render/frame_graph.h"
#include "render/ground_textures.h"
#include "render/targets.h"
#include "scene/audio_sync.h"
#include "scene/camera.h"
#include "scene/characters.h"
#include "scene/colliders.h"
#include "scene/grass.h"
#include "scene/models.h"
#include "scene/scene_gpu.h"
#include "scene/stream_sync.h"
#include "scene/vegetation.h"
#include "sim/debris.h"
#include "sim/water.h"
#include "sim/world_sim.h"
#include "sky/lightning.h"
#include "tools/bench.h"
#include "tools/movelog.h"
#include "tools/screenshot.h"
#include "tools/test_hooks.h"
#include "ui/main_panel.h"
#include "ui/overlays.h"
#include "vegetation/trees.h"
#include "world/terrain.h"
#include "world/world.h"
#include "world/worldmap_gpu.h"
#include "world/worldmap.h"

namespace {
using clock_t_ = std::chrono::steady_clock;
constexpr float MAX_STEP = 0.1f;            // simulation step cap (s): a hitch doesn't launch everything
constexpr float MAX_PHYSICS_STEP = 1.f / 30.f;
constexpr float MAX_SPRAY_STEP = 0.05f;
constexpr int MIN_SWAP_EXTENT = 64;
constexpr uint32_t PROF_START_FRAME = 30;   // --prof: capture from here
constexpr uint32_t MIN_BLADE_SLOTS = 8192, BLADE_HEADROOM = 4096;
constexpr float CPU_FRAME_EMA = 0.1f;       // smoothing of the displayed CPU frame time
constexpr int VEG_REGEN_PRESET = 5;         // --veg-regen: grow species A as this preset

clock_t_::time_point lastFrame = clock_t_::now();

// fence wait, then everything last frame left for the CPU: UI screenshot, sim readback, timings, blade count
void readBackLastFrame() {
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
        bladeCap = std::min(maxBlades, std::max(MIN_BLADE_SLOTS, lastBladeCount + lastBladeCount / 4 + BLADE_HEADROOM));
    }
}

// previous frame is done: free its staging / scratch, then record this frame's uploads into uploadCmd
void uploadWindow() {
    for (auto& f : retireNext) f();
    retireNext.clear();
    uploadsRecorded = false;
    frameUploadBytes = 0;
    if ((int)frameIndex == vegRegenFrame) { trees::treePreset(VEG_REGEN_PRESET, trees::treeParams[0]); vegDirty = true; }
    if (float sd = S.dayCycle ? trees::seasonBucketDay(S.dayOfYear) : -1.f; sd != trees::seasonDay) {
        trees::seasonDay = sd;  // the leaves turned (or the cycle was switched): regrow
        vegDirty = true;
    }
    if (vegDirty) { vegDirty = false; regenVegetation(); }
    if (uint64_t env = bakeEnvHash(); env != bakeEnv) { PROF_ZONE("rebake volumes"); bakeEnv = env; rebuildBakeVolumes(true); }  // sun / sky moved
    saveFinishedBakes();
    testHooksUploadWindow();
    { PROF_ZONE("editor apply"); editor::apply(); }  // edited cells go live in this upload window
    {
        PROF_ZONE("stream update");
        if (stream::update(cam.pos.x, cam.pos.y, cam.pos.z, loadRadius, uploadModel, releaseModel)) onStreamChanged(editor::takeMoveOnly());
    }
    { PROF_ZONE("fractures"); processFractures(); }  // fragment meshes upload in this frame's upload window
    { PROF_ZONE("world map upload"); uploadWorldMap(); }
    { PROF_ZONE("ground textures"); updateGroundTextures(); }
    if (!charSpawns.empty()) { PROF_ZONE("spawn characters"); spawnCharacters(); }
    if (uploadsRecorded) vkEndCommandBuffer(uploadCmd);
    for (uint32_t id : audioMeshRetire) audio::meshRelease(id);
    audioMeshRetire.clear();
}

void buildUI() {
    PROF_ZONE("UI");
    ImGui_ImplVulkan_NewFrame();
    ImGui_ImplGlfw_NewFrame();
    testHooksImGuiInput();
    ImGui::NewFrame();
    drawUI();
    drawOverlays();
    ImGui::Render();
}

// the camera follows the replay log (its events fire on their frames), or the player
void moveCamera(float dt) {
    prevCam = cam;
    if (replaying) {
        if (replayAt < replayFrames.size()) {
            const MoveFrame& m = replayFrames[replayAt];
            cam.pos = {m.x, m.y, m.z};
            cam.yaw = m.yaw * 3.14159265f / 180.f;
            cam.pitch = m.pitch * 3.14159265f / 180.f;
            simTime = m.t;
            walkMode = false;  // the log already holds where walking put the camera
            while (replayEvAt < replayEvents.size() && replayEvents[replayEvAt].f <= m.f) {
                const MoveEvent& e = replayEvents[replayEvAt++];
                if (e.kind == "throw") {
                    float s0 = S.throwSize, v0 = S.throwSpeed;
                    S.throwSize = e.a; S.throwSpeed = e.b;
                    throwRockImpl();
                    S.throwSize = s0; S.throwSpeed = v0;
                } else if (e.kind == "shot" && replayShots) shotSceneReq = true;
            }
            replayAt++;
        } else {
            replaying = false;
            printf("replay finished\n");
            if (replayExit) glfwSetWindowShouldClose(window, 1);
        }
    } else if (!bench.running) {
        updateCamera(std::min(dt, MAX_STEP));
        simTime += std::min(dt, MAX_STEP);
    }
    moveLogFrame(dt);
}

void simulate(float dt) {
    testHooksSimulate();
    // this frame's poses become "previous" before anything moves (per-object motion vectors)
    for (Thrown& t : thrown) { t.prevXf = t.xf; t.hasPrev = true; }
    for (Droplet& d : spray) { d.prevP = d.p; d.hasPrev = true; }
    for (character::Character* c : character::all())
        if (auto* gc = (GpuChar*)c->user) { gc->prevW = character::worldXf(*c); gc->hasPrev = true; }
    if (bench.running) return;
    PROF_ZONE("physics");
    float step = std::min(dt, MAX_STEP);
    { PROF_ZONE("thrown + water forces"); updateThrown(dt); }
    { PROF_ZONE("spray"); updateSpray(std::min(dt, MAX_SPRAY_STEP)); }
    { PROF_ZONE("fires"); updateFires(dt); }
    { PROF_ZONE("critters"); updateCritters(dt); }
    advanceDay(dt);
    advanceWeather(dt);
    updateLightning(step);
    { PROF_ZONE("crater water"); updateRiverSwell(step); updateCraterWater(step); }
    updateWildlife(step);
    { PROF_ZONE("characters: animate"); character::preStep(step); }
    { PROF_ZONE("physics step (Jolt)"); physics::step(std::min(dt, MAX_PHYSICS_STEP)); }
    { PROF_ZONE("characters: pose"); character::postStep(step); }
    characterSteps(step);
    testHooksAfterPhysics();
}

void prepareRender(float dt) {
    benchStep();
    autoScale();
    { PROF_ZONE("frame data + placement"); updateFrameData(); }
    { PROF_ZONE("write thrown"); writeThrown(); }
    { PROF_ZONE("write characters"); writeCharacters(); }
    { PROF_ZONE("audio"); updateAudio(std::min(dt, MAX_STEP)); }
    { PROF_ZONE("cull models"); cullModels(); }
    { PROF_ZONE("cull trees"); cullTrees(); }
    { PROF_ZONE("tree colliders"); syncTreeBodies(); }
    { PROF_ZONE("rock colliders"); syncRockBodies(); }
}

// acquire, record + submit the render graph, present. False: the swapchain is out of date (frame skipped).
bool renderAndPresent() {
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
        return false;
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
    return true;
}

// benchmark results, screenshot sequences, --shot exit. False: done, quit.
bool afterPresent(const LaunchOptions& o) {
    if (o.benchOnly && bench.done) {
        printBenchmark();
        return false;
    }
    if (o.shotPath && o.shotSeq > 0 && (int)frameIndex > o.shotFrames - o.shotSeq && (int)frameIndex < o.shotFrames) {
        vkDeviceWaitIdle(dev);
        saveShot((std::string(o.shotPath) + "." + std::to_string(o.shotFrames - (int)frameIndex)).c_str());
    }
    if (o.shotPath && (int)frameIndex == o.shotFrames) {
        vkDeviceWaitIdle(dev);
        for (int i = 0; i < std::min(gCraterCount.load(), MAX_CRATERS); i++)
            printf("crater %d: r %.1f m at (%.0f, %.0f)%s fill %.2f\n", i, gCraters[i].r, gCraters[i].x, gCraters[i].z,
                   craterWater[i].pond ? " pond" : "", craterWater[i].fill);
        graph.begin();
        readTimestamps();
        saveShot(o.shotPath);
        return false;
    }
    saveSettingsIfChanged(glfwGetTime());  // what was changed in the UI survives a crash
    return true;
}

// frame limiter for the fps target
void limitFrameRate(clock_t_::time_point frameStart) {
    if (targetFps <= 0 || bench.running) return;
    auto end = frameStart + std::chrono::duration_cast<clock_t_::duration>(std::chrono::duration<double>(1.0 / targetFps));
    while (clock_t_::now() < end) {
        auto left = end - clock_t_::now();
        if (left > std::chrono::milliseconds(2)) std::this_thread::sleep_for(left - std::chrono::milliseconds(1));
    }
}

void handleWindowChanges() {
    if (pendingWindowMode >= 0) { applyWindowMode(pendingWindowMode); pendingWindowMode = -1; }
    if (!swapDirty) return;
    vkDeviceWaitIdle(dev);
    createSwapchain();
    ImGui_ImplVulkan_SetMinImageCount(2);
    swapDirty = false;
    // output resolution follows the window
    if (resFollowWindow && swapExtent.width >= MIN_SWAP_EXTENT && swapExtent.height >= MIN_SWAP_EXTENT &&
        (swapExtent.width != OW || swapExtent.height != OH)) {
        resOptions[4][0] = swapExtent.width;
        resOptions[4][1] = swapExtent.height;
        pendingRes = 4;
    }
}

void applyPendingResolution() {
    if (pendingRes < 0) return;
    vkDeviceWaitIdle(dev);
    destroyScreenResources();
    OW = resOptions[pendingRes][0];
    OH = resOptions[pendingRes][1];
    createScreenResources();
    writeDescriptors();
    pendingRes = -1;
}
}  // namespace

bool runFrame(const LaunchOptions& o) {
    if (glfwWindowShouldClose(window)) return false;
    if (profFrames > 0 && frameIndex == PROF_START_FRAME) prof::toggleCapture();
    // --prof: the capture is done, quit (main.cpp writes the trace on the way out)
    if (profFrames > 0 && frameIndex > PROF_START_FRAME + 1 && !prof::capturing && prof::hasCapture()) return false;
    prof::frameBegin();
    PROF_ZONE("frame");
    if (float sf = prof::slowFps(); sf > 0.f) {  // slow motion (Profiler window)
        PROF_ZONE("slow motion sleep");
        static auto prevT = clock_t_::now();
        std::this_thread::sleep_until(prevT + std::chrono::microseconds((int64_t)(1e6f / sf)));
        prevT = clock_t_::now();
    }
    { PROF_ZONE("poll events"); glfwPollEvents(); }
    input::update();
    int fw, fh;
    glfwGetFramebufferSize(window, &fw, &fh);
    if (fw == 0 || fh == 0) { glfwWaitEvents(); return true; }  // minimised
    handleWindowChanges();

    auto now = clock_t_::now();
    float dt = std::chrono::duration<float>(now - lastFrame).count();
    lastFrame = now;
    frameDt = dt;
    keepSnowTerrainFresh();
    cpuFrameMs = cpuFrameMs * (1.f - CPU_FRAME_EMA) + dt * 1000.f * CPU_FRAME_EMA;
    if (replaying && replayAt < replayFrames.size()) dt = replayFrames[replayAt].dt;  // the recorded step: same physics

    readBackLastFrame();
    uploadWindow();
    buildUI();
    applyPendingResolution();
    moveCamera(dt);
    simulate(dt);
    prepareRender(dt);
    if (!renderAndPresent()) return true;
    if (!afterPresent(o)) return false;
    limitFrameRate(now);
    return true;
}

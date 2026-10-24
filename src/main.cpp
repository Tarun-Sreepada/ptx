// ptx: a Vulkan ray-query path tracer with a streamed, destructible world.
//   app/        command line, startup, the frame loop
//   core/       math, limits, jobs (thread pool), profiling, input
//   gpu/        Vulkan device, memory, swapchain, pipelines, descriptors, render graph, DLSS
//   render/     screen targets, frame uniforms, culling, lights, bakes, the frame's render graph
//   scene/      static scene resources, streamed models, vegetation, rocks, characters, camera
//   sim/        water, debris, effects, the GPU world simulation's CPU side
//   sky/        sun / moon / stars, lightning
//   destruction/ thrown rocks, fracture, craters, breaking trees and buildings
//   world/      world data: cells, glTF, terrain, world map, materials, textures
//   physics/ vegetation/ characters/ audio/ editor/ ui/ tools/
// See README.md for building and the folder guide.
#include <cstdio>
#include <unistd.h>

#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "app/cli.h"
#include "app/frame_loop.h"
#include "app/settings_io.h"
#include "app/startup.h"
#include "core/prof.h"
#include "gpu/context.h"
#include "gpu/pipelines.h"
#include "gpu/rendergraph.h"
#include "gpu/dlss.h"
#include "render/frame_graph.h"
#include "render/targets.h"
#include "tools/movelog.h"

int main(int argc, char** argv) {
    LaunchOptions opts;
    if (!parseArgs(argc, argv, opts)) return 0;
    startup(opts);
    while (runFrame(opts)) {}

    vkDeviceWaitIdle(dev);
    graph.destroy();
    destroyScreenResources();
    dlss::shutdown();
    savePipelineCache();
    saveSettings();
    if (profFrames > 0 && prof::hasCapture()) printf("trace: %s\n", prof::exportTrace().c_str());
    // Quit now. Everything left is memory and handles the OS reclaims, and an orderly teardown can stall for seconds
    // (a loader thread mid-way through a 400 MB model, the audio simulation, the editor's import worker): the window
    // has to close when it's told to.
    closeMoveLog();
    glfwDestroyWindow(window);
    fflush(stdout);
    fflush(stderr);
    _exit(0);
}

#include "render/frame_graph.h"
#include "render/graph_internal.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <vulkan/vulkan.h>
#include <imgui.h>
#include <imgui_impl_vulkan.h>
#include "core/prof.h"
#include "gpu/dlss.h"
#include "gpu/rendergraph.h"
#include "world/worldmap.h"
#include "app/frame_state.h"
#include "app/settings.h"
#include "core/limits.h"
#include "core/math.h"
#include "gpu/memory.h"
#include "gpu/pipelines.h"
#include "gpu/swapchain.h"
#include "gpu/vk.h"
#include "render/bake.h"
#include "render/frame_data.h"
#include "render/ground_vt.h"
#include "render/passes.h"
#include "render/targets.h"
#include "scene/camera.h"
#include "scene/characters.h"
#include "scene/grass.h"
#include "scene/scene_gpu.h"
#include "sim/world_sim.h"
#include "tools/screenshot.h"

rg::Graph graph;

bool asyncCompute = true;    // the async compute queue at all (fog below)

// BVH build on the async compute queue: measured 2026-10-28 (RTX 3070) it made frames ~2.7 ms SLOWER (the build is
// only ~0.5 ms and, sharing the GPU, it held up the raster G-buffer), so it runs on the graphics queue by default
bool asyncBVH = false;

bool asyncFog = false;       // volumetric fog on the async compute queue: also ~2.9 ms slower per frame here (same test)

// The frame as a render graph (rendergraph.h): passes declare what they read / write; barriers, queue
// batches and timeline waits are derived. The BVH build runs on the async compute queue and overlaps the
// raster G-buffer, which only needs the grass vertices, not the acceleration structures.
void buildFrameGraph(uint32_t si, ImDrawData* dd) {
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
    rg::Graph& g = graph;
    FrameGraphCtx f{g,
                    g.resource("streamed models"), g.resource("grass counter"), g.resource("grass geometry"), g.resource("terrain"),
                    g.resource("sky LUT"), g.resource("BVH"), g.resource("G-buffer"), g.resource("DI reservoirs"), g.resource("GI reservoirs"),
                    g.resource("lighting"), g.resource("denoise"), g.resource("history"), g.resource("output"), g.resource("swapchain"),
                    g.resource("probes"), g.resource("wind map"), g.resource("HDR scene"), g.resource("exposure"), g.resource("bloom"),
                    g.resource("fog"), g.resource("fog history"),
                    asyncCompute && asyncBVH ? rg::COMPUTE : rg::GFX, asyncCompute && asyncFog ? rg::COMPUTE : rg::GFX,
                    (W + 7) / 8, (H + 7) / 8};
    // the uploads command buffer (stream.cpp models + their BLASes) is submitted first on the graphics queue
    g.add("uploads", rg::GFX, [](VkCommandBuffer) {}).write(f.models, fgc::ASB | fgc::XFER, fgc::ASW | fgc::XW);
    addWorldPasses(f);
    addScenePasses(f);
    addResolvePasses(f);
    addPostPasses(f);
    addPresentPass(f, si, dd, grabUI);
}

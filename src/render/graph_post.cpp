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

void addPostPasses(const FrameGraphCtx& f) {
    using namespace rg;
    using namespace fgc;
    Graph& g = f.g;
    const uint32_t models = f.models, grassCount = f.grassCount, grass = f.grass, terrain = f.terrain, skyR = f.skyR, accel = f.accel,
                   gbuf = f.gbuf, diRes = f.diRes, giRes = f.giRes, light = f.light, denoise = f.denoise, history = f.history,
                   out = f.out, swap = f.swap, probesR = f.probesR, windR = f.windR, hdr = f.hdr, expoR = f.expoR, bloomR = f.bloomR,
                   fogR = f.fogR, fogHistR = f.fogHistR;
    const Queue bvhQ = f.bvhQ, fogQ = f.fogQ;
    const uint32_t gx = f.gx, gy = f.gy;
    auto bind = bindSet;
    (void)models; (void)grassCount; (void)grass; (void)terrain; (void)skyR; (void)accel; (void)gbuf; (void)diRes; (void)giRes;
    (void)light; (void)denoise; (void)history; (void)out; (void)swap; (void)probesR; (void)windR; (void)hdr; (void)expoR;
    (void)bloomR; (void)fogR; (void)fogHistR; (void)bvhQ; (void)fogQ; (void)gx; (void)gy; (void)bind;
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
}

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

void addWorldPasses(const FrameGraphCtx& f) {
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
        g.add("terrain", GFX, [=](VkCommandBuffer cb) { bind(cb); dispatch(cb, P_TERRAIN, (terrainLevels * TERRAIN_LEVEL_VERTS + 63) / 64, 1); }).write(terrain, CS, WR);
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
}

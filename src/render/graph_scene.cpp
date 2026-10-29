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
#include "render/occlusion.h"
#include "render/veg_gpu.h"
#include "render/frame_graph.h"
#include "scene/grass.h"
#include "scene/scene_gpu.h"
#include "sim/world_sim.h"
#include "tools/screenshot.h"

void addScenePasses(const FrameGraphCtx& f) {
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
    if (numCharInst)  // graphics queue: the async BVH build waits on it anyway, the raster G-buffer needn't wait on the BVH
        g.add("skinning", GFX, [=](VkCommandBuffer cb) { bind(cb); recordSkinning(cb); }).write(models, CS | ASB, RW | ASW | ASR);
    // the BVH: names start with "BVH build" (the UI sums them)
    if (terrainDirty)
        g.add("BVH build: terrain", bvhQ, [](VkCommandBuffer cb) { cmdBuildTerrainBLAS(cb); })
            .read(terrain, ASB, R).write(accel, ASB, ASW | ASR);
    g.add("BVH build: grass", bvhQ, [](VkCommandBuffer cb) { cmdBuildGrassBLAS(cb); })
        .read(grass, ASB, R).write(accel, ASB, ASW | ASR);
    g.add("BVH build: TLAS", bvhQ, [](VkCommandBuffer cb) { cmdBuildTLAS(cb); })
        .read(models, ASB, ASR).write(accel, ASB, ASW | ASR);
    terrainDirty = false;

    if (S.rasterPrimary)
        g.add("veg cull", GFX, [=](VkCommandBuffer cb) { bind(cb); recordVegCull(cb); }).write(models, CS, RW);
    if (S.rasterPrimary)
        g.add("raster G-buffer", GFX, [](VkCommandBuffer cb) { rasterGBuffer(cb); })
            .read(grass, RASTER, VTX).read(grassCount, RASTER, VTX).read(terrain, RASTER, VTX).read(models, RASTER, VTX)
            .read(windR, RASTER, R)  // plants sway with the wind map (sway.glsl)
            .write(gbuf, COLOR, CA);
    const uint32_t cloudsR = g.resource("clouds");
    g.add("clouds", GFX, [=](VkCommandBuffer cb) { bind(cb); uint32_t cd = (uint32_t)std::clamp(S.cloudDiv, 1, 16);
        dispatch(cb, P_CLOUD, ((W + cd - 1) / cd + 7) / 8, ((H + cd - 1) / cd + 7) / 8); })
        .read(skyR, CS, R).write(cloudsR, CS, RW);  // primary reads it next
    g.add("primary", GFX, [=](VkCommandBuffer cb) { bind(cb); dispatch(cb, P_PRIMARY, gx, gy, {S.rasterPrimary ? 1u : 0u, 0, 0, 0}); })
        .read(accel, CS, ASR | R).read(skyR, CS, R).read(models, CS, R).read(grass, CS, R).read(terrain, CS, R).read(cloudsR, CS, R).write(gbuf, CS, RW);
    if (S.rasterPrimary && S.occlusionCull)  // next frame's occlusion grid (render/occlusion.h); the CPU reads it after the fence
        g.add("occlusion grid", GFX, [=](VkCommandBuffer cb) {
            bind(cb);
            dispatch(cb, P_OCCLUSION, (W + OCC_TILE - 1) / OCC_TILE, (H + OCC_TILE - 1) / OCC_TILE);
            memBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_HOST_READ_BIT);
        }).read(gbuf, CS, R);
    g.add("surface fx", GFX, [=](VkCommandBuffer cb) { bind(cb); dispatch(cb, P_SURFACE_FX, gx, gy); })
        .read(accel, CS, ASR | R).write(gbuf, CS, RW);
    g.add("volumetric fog", fogQ, [=](VkCommandBuffer cb) { bind(cb); uint32_t fd = (uint32_t)std::clamp(S.fogDiv, 1, 128);
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
        graph.zoneBegin(cb, "trace");
        dispatch(cb, P_GI, gx, gy);
        graph.zoneEnd(cb);
        if (S.restirGI) {
            computeBarrier(cb);
            graph.zoneBegin(cb, "spatial");
            dispatch(cb, P_GI_SPATIAL, gx, gy);
            graph.zoneEnd(cb);
        }
        if (S.giCache) {
            computeBarrier(cb);
            graph.zoneBegin(cb, "radiance cache");
            dispatch(cb, P_RADCACHE, RC_ENTRIES / 64, 1);
            graph.zoneEnd(cb);
        }
    }).read(accel, CS, ASR | R).read(gbuf, CS, R).read(history, CS, R).read(skyR, CS, R).read(probesR, CS, R).write(giRes, CS, RW).write(light, CS, RW);
}

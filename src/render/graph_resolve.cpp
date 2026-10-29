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

void addResolvePasses(const FrameGraphCtx& f) {
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
        vkCmdCopyImage(cb, ximgs[X_FOGAGE].img, VK_IMAGE_LAYOUT_GENERAL, ximgs[X_FOGAGEHIST].img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
    }).read(fogR, XFER, XR).read(hdr, XFER, XR).write(fogHistR, XFER, XW);
}

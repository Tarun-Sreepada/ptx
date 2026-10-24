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

void addPresentPass(const FrameGraphCtx& f, uint32_t si, ImDrawData* dd, bool grabUI) {
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

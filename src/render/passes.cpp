#include "render/passes.h"
#include <array>
#include <cstdint>
#include <vulkan/vulkan.h>
#include "destruction/thrown.h"
#include "gpu/pipelines.h"
#include "render/targets.h"
#include "scene/model_instances.h"
#include "scene/models.h"
#include "scene/scene_gpu.h"
#include "scene/vegetation.h"

void dispatch(VkCommandBuffer cb, int p, uint32_t gx, uint32_t gy, std::array<uint32_t, 4> pc) {
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipes[p]);
    vkCmdPushConstants(cb, pipeLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc.data());
    vkCmdDispatch(cb, gx, gy, 1);
}

void copyImg(VkCommandBuffer cb, int src, int dst) {
    VkImageCopy r{};
    r.srcSubresource = r.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.extent = {W, H, 1};
    vkCmdCopyImage(cb, imgs[src].img, VK_IMAGE_LAYOUT_GENERAL, imgs[dst].img, VK_IMAGE_LAYOUT_GENERAL, 1, &r);
}

void rasterGBuffer(VkCommandBuffer cb) {
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

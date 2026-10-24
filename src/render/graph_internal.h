// Shared by the render graph's stage files (render/graph_*.cpp): the frame's logical resources, its queues and
// dispatch size, stage / access shorthands, binding the one descriptor set.
#pragma once
#include <cstdint>

#include <vulkan/vulkan.h>
#include <imgui.h>

#include "gpu/pipelines.h"
#include "gpu/rendergraph.h"

struct FrameGraphCtx {
    rg::Graph& g;
    uint32_t models, grassCount, grass, terrain, skyR, accel, gbuf, diRes, giRes, light, denoise, history, out, swap, probesR, windR, hdr,
        expoR, bloomR, fogR, fogHistR;
    rg::Queue bvhQ, fogQ;  // BVH build / volumetric fog: async compute when enabled
    uint32_t gx, gy;       // 8 x 8 workgroups over the render resolution
};

namespace fgc {
constexpr VkPipelineStageFlags2 CS = VK_PIPELINE_STAGE_2_COMPUTE_SHADER_BIT, XFER = VK_PIPELINE_STAGE_2_ALL_TRANSFER_BIT,
                                ASB = VK_PIPELINE_STAGE_2_ACCELERATION_STRUCTURE_BUILD_BIT_KHR,
                                RASTER = VK_PIPELINE_STAGE_2_VERTEX_INPUT_BIT | VK_PIPELINE_STAGE_2_VERTEX_SHADER_BIT |
                                         VK_PIPELINE_STAGE_2_DRAW_INDIRECT_BIT | VK_PIPELINE_STAGE_2_FRAGMENT_SHADER_BIT,
                                COLOR = VK_PIPELINE_STAGE_2_COLOR_ATTACHMENT_OUTPUT_BIT;
constexpr VkAccessFlags2 R = VK_ACCESS_2_SHADER_READ_BIT, WR = VK_ACCESS_2_SHADER_WRITE_BIT, RW = R | WR,
                         XR = VK_ACCESS_2_TRANSFER_READ_BIT, XW = VK_ACCESS_2_TRANSFER_WRITE_BIT,
                         ASR = VK_ACCESS_2_ACCELERATION_STRUCTURE_READ_BIT_KHR, ASW = VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
                         VTX = VK_ACCESS_2_VERTEX_ATTRIBUTE_READ_BIT | VK_ACCESS_2_INDEX_READ_BIT | VK_ACCESS_2_INDIRECT_COMMAND_READ_BIT | R,
                         CA = VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_READ_BIT;
}  // namespace fgc

inline void bindSet(VkCommandBuffer cb) { vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipeLayout, 0, 1, &descSet, 0, nullptr); }

// in frame order
void addWorldPasses(const FrameGraphCtx& f);      // history reset, sky LUT, world sim, ground VT, terrain, wind, grass
void addScenePasses(const FrameGraphCtx& f);      // skinning, BVH build, G-buffer (raster + primary), surface fx, fog, ReSTIR DI, probe bake, GI
void addResolvePasses(const FrameGraphCtx& f);    // DLSS-RR or temporal + a-trous + composite, history copies, fog history
void addPostPasses(const FrameGraphCtx& f);       // exposure, bloom, tonemap / post
void addPresentPass(const FrameGraphCtx& f, uint32_t swapIndex, ImDrawData* dd, bool grabUI);

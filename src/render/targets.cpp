#include "render/targets.h"
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <vulkan/vulkan.h>
#include "gpu/dlss.h"
#include "app/frame_state.h"
#include "gpu/context.h"
#include "gpu/memory.h"
#include "gpu/vk.h"

static const VkFormat imgFormats[I_COUNT] = {
    VK_FORMAT_R32_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
    VK_FORMAT_R32_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
    VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R32G32B32A32_SFLOAT,
    VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R8G8B8A8_UNORM};

Image imgs[I_COUNT];

// More screen images, bindings 54 + index (common.glsl and the passes that use them).
enum SizeClass { RENDER_RES, HALF_RES, OUTPUT_RES };

struct XImgDef { VkFormat fmt; SizeClass size; };

static const XImgDef xDefs[X_COUNT] = {
    {VK_FORMAT_R16G16B16A16_SFLOAT, RENDER_RES}, {VK_FORMAT_R16G16B16A16_SFLOAT, RENDER_RES}, {VK_FORMAT_R16G16B16A16_SFLOAT, RENDER_RES},
    {VK_FORMAT_R16G16B16A16_SFLOAT, RENDER_RES},
    {VK_FORMAT_R16G16B16A16_SFLOAT, OUTPUT_RES},   // HDR scene before post (post.glsl hdrTex)
    {VK_FORMAT_R16G16B16A16_SFLOAT, OUTPUT_RES},   // HDR display encoding (PQ / scRGB)
    {VK_FORMAT_R16G16B16A16_SFLOAT, RENDER_RES}, {VK_FORMAT_R16G16B16A16_SFLOAT, RENDER_RES},  // volumetric fog + its history (top-left 1/fogDiv used)
    {VK_FORMAT_R16G16B16A16_SFLOAT, RENDER_RES},   // G-buffer velocity: world-space offset to where the surface was last frame
    {VK_FORMAT_R16_SFLOAT, RENDER_RES}, {VK_FORMAT_R16_SFLOAT, RENDER_RES},  // fog history age (frames) + last frame's
    {VK_FORMAT_R16G16B16A16_SFLOAT, RENDER_RES}, {VK_FORMAT_R16G16B16A16_SFLOAT, RENDER_RES}};  // sky clouds ping-pong: light, transmittance (top-left 1/cloudDiv used)

VkSampler postSampler;          // bilinear, clamp

Image bloomImg;                 // half output resolution, BLOOM_MIPS mips

VkImageView bloomViews[8];

Buffer exposureBuf;             // histogram + adapted exposure (host visible: the UI shows the metered EV)

Image ximgs[X_COUNT];

static int gbufX(int i) { return i == 7 ? X_GVEL : X_GMAT + i - 4; }

VkFormat gbufFormat(int i) { return i < 4 ? imgFormats[I_GPOS + i] : xDefs[gbufX(i)].fmt; }

Image& gbufImage(int i) { return i < 4 ? imgs[I_GPOS + i] : ximgs[gbufX(i)]; }

Buffer resCur, resPrev, giCur, giPrev;

Image depthImg;

uint32_t W = 1920, H = 1080;   // render resolution (what rays are traced at)

uint32_t OW = 1920, OH = 1080; // output resolution (DLSS-RR upscales W x H to this; equal without it)

dlss::Mode upscaler = dlss::QUALITY;

bool rrActive = false;         // upscaler != OFF and the feature was created

Image rrImgs[RR_COUNT];

Buffer radCacheBuf;

void createScreenResources() {
    // render resolution follows the upscaler mode; the output (I_OUT, RR_OUT) stays at the chosen resolution
    float sc = upscaler != dlss::OFF ? dlss::renderScale(upscaler) : 1.f;
    W = std::max(64u, (uint32_t)(OW * sc + 0.5f));
    H = std::max(64u, (uint32_t)(OH * sc + 0.5f));
    for (int i = 0; i < I_COUNT; i++) imgs[i] = i == I_OUT ? mkImage(OW, OH, imgFormats[i]) : mkImage(W, H, imgFormats[i]);
    for (int i = 0; i < RR_COUNT; i++) rrImgs[i] = i == RR_OUT ? mkImage(OW, OH, rrFormats[i]) : mkImage(W, H, rrFormats[i]);
    bloomImg = mkImageUsage(std::max(OW / 2, 1u), std::max(OH / 2, 1u), VK_FORMAT_R16G16B16A16_SFLOAT,
                            VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_SAMPLED_BIT, BLOOM_MIPS);
    for (uint32_t m = 0; m < BLOOM_MIPS; m++) {
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = bloomImg.img;
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = VK_FORMAT_R16G16B16A16_SFLOAT;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, m, 1, 0, 1};
        VK_CHECK(vkCreateImageView(dev, &vi, nullptr, &bloomViews[m]));
    }
    for (int i = 0; i < X_COUNT; i++) {
        SizeClass c = xDefs[i].size;
        ximgs[i] = c == OUTPUT_RES ? mkImage(OW, OH, xDefs[i].fmt) : c == HALF_RES ? mkImage((W + 1) / 2, (H + 1) / 2, xDefs[i].fmt) : mkImage(W, H, xDefs[i].fmt);
    }
    VkDeviceSize rs = (VkDeviceSize)W * H * 16;
    resCur = mkBuf(rs, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    resPrev = mkBuf(rs, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    giCur = mkBuf(rs * 3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    giPrev = mkBuf(rs * 3, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    depthImg = mkImage(W, H, depthFmt);
    oneShot([](VkCommandBuffer cb) {
        VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b.newLayout = VK_IMAGE_LAYOUT_DEPTH_ATTACHMENT_OPTIMAL;
        b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b.image = depthImg.img;
        b.subresourceRange = {VK_IMAGE_ASPECT_DEPTH_BIT, 0, 1, 0, 1};
        b.dstAccessMask = VK_ACCESS_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_EARLY_FRAGMENT_TESTS_BIT, 0, 0, nullptr, 0, nullptr, 1, &b);
        for (auto& im : imgs)
            imgBarrier(cb, im.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        for (auto& im : rrImgs)
            imgBarrier(cb, im.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        for (auto& im : ximgs)
            imgBarrier(cb, im.img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
        VkImageMemoryBarrier bb{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        bb.newLayout = VK_IMAGE_LAYOUT_GENERAL;
        bb.srcQueueFamilyIndex = bb.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        bb.image = bloomImg.img;
        bb.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, BLOOM_MIPS, 0, 1};
        bb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
        vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, 0, 0, nullptr, 0, nullptr, 1, &bb);
    });
    rrActive = false;
    if (upscaler != dlss::OFF)
        oneShot([&](VkCommandBuffer cb) { rrActive = dlss::create(cb, upscaler, W, H, OW, OH); });
    if (upscaler != dlss::OFF && !rrActive) fprintf(stderr, "dlss: falling back to the built-in denoiser\n");
    resetHistory = true;
}

void destroyScreenResources() {
    dlss::release();
    for (auto& im : imgs) freeImage(im);
    for (auto& im : rrImgs) freeImage(im);
    for (auto& im : ximgs) freeImage(im);
    for (uint32_t m = 0; m < BLOOM_MIPS; m++) vkDestroyImageView(dev, bloomViews[m], nullptr);
    freeImage(bloomImg);
    freeBuf(resCur);
    freeBuf(resPrev);
    freeBuf(giCur);
    freeBuf(giPrev);
    freeImage(depthImg);
}

int pendingRes = -1;

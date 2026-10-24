#include "gpu/textures.h"
#include <vector>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <vulkan/vulkan.h>
#include "world/world.h"
#include "core/limits.h"
#include "gpu/context.h"
#include "gpu/memory.h"
#include "gpu/pipelines.h"
#include "gpu/vk.h"

Image defaultTex;

VkSampler texSampler;

// ---------------------------------------------------------------- streamed models: upload / release / instances
uint32_t mipCount(uint32_t w, uint32_t h) { return 32 - (uint32_t)__builtin_clz(std::max(w, h)); }

Image mkTexture(uint32_t w, uint32_t h, uint32_t mips, VkFormat fmt) {
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = fmt;
    ci.extent = {w, h, 1};
    ci.mipLevels = mips;
    ci.arrayLayers = 1;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = VK_IMAGE_USAGE_SAMPLED_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    return mkImageCI(ci, VK_IMAGE_ASPECT_COLOR_BIT);
}

static void mipBarrier(VkCommandBuffer cb, VkImage img, uint32_t base, uint32_t count, VkImageLayout from, VkImageLayout to, VkAccessFlags sa,
                       VkAccessFlags da, VkPipelineStageFlags ds = VK_PIPELINE_STAGE_TRANSFER_BIT) {
    VkImageMemoryBarrier b{VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
    b.oldLayout = from;
    b.newLayout = to;
    b.srcQueueFamilyIndex = b.dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
    b.image = img;
    b.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, base, count, 0, 1};
    b.srcAccessMask = sa;
    b.dstAccessMask = da;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, ds, 0, 0, nullptr, 0, nullptr, 1, &b);
}

// copy level 0 from staging, build the mip chain with blits, leave it shader-readable
void recordTextureUpload(VkCommandBuffer cb, const Image& im, uint32_t w, uint32_t h, uint32_t mips, VkBuffer src, VkDeviceSize off) {
    mipBarrier(cb, im.img, 0, mips, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    VkBufferImageCopy r{};
    r.bufferOffset = off;
    r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
    r.imageExtent = {w, h, 1};
    vkCmdCopyBufferToImage(cb, src, im.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &r);
    for (uint32_t i = 1; i < mips; i++) {
        mipBarrier(cb, im.img, i - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
                   VK_ACCESS_TRANSFER_READ_BIT);
        VkImageBlit bl{};
        bl.srcSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i - 1, 0, 1};
        bl.srcOffsets[1] = {(int32_t)std::max(1u, w >> (i - 1)), (int32_t)std::max(1u, h >> (i - 1)), 1};
        bl.dstSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, i, 0, 1};
        bl.dstOffsets[1] = {(int32_t)std::max(1u, w >> i), (int32_t)std::max(1u, h >> i), 1};
        vkCmdBlitImage(cb, im.img, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, im.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 1, &bl, VK_FILTER_LINEAR);
    }
    const VkPipelineStageFlags shaders = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT;
    if (mips > 1)
        mipBarrier(cb, im.img, 0, mips - 1, VK_IMAGE_LAYOUT_TRANSFER_SRC_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_READ_BIT,
                   VK_ACCESS_SHADER_READ_BIT, shaders);
    mipBarrier(cb, im.img, mips - 1, 1, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
               VK_ACCESS_SHADER_READ_BIT, shaders);
}

// BC7 texture: every mip level comes precomputed (texcomp.h), so the upload is copies only
void recordCompressedUpload(VkCommandBuffer cb, const Image& im, const TextureData& t, VkBuffer src, VkDeviceSize off) {
    mipBarrier(cb, im.img, 0, t.bcMips, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, 0, VK_ACCESS_TRANSFER_WRITE_BIT);
    std::vector<VkBufferImageCopy> rs(t.bcMips);
    for (uint32_t l = 0; l < t.bcMips; l++) {
        rs[l] = {};
        rs[l].bufferOffset = off + t.bcLevels[l];
        rs[l].imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, l, 0, 1};
        rs[l].imageExtent = {std::max(1u, t.w >> l), std::max(1u, t.h >> l), 1};
    }
    vkCmdCopyBufferToImage(cb, src, im.img, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, t.bcMips, rs.data());
    mipBarrier(cb, im.img, 0, t.bcMips, VK_IMAGE_LAYOUT_TRANSFER_DST_OPTIMAL, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL, VK_ACCESS_TRANSFER_WRITE_BIT,
               VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_FRAGMENT_SHADER_BIT);
}

void writeTexDescriptor(uint32_t slot, VkImageView view) {
    VkDescriptorImageInfo ii{texSampler, view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = descSet;
    w.dstBinding = 32;
    w.dstArrayElement = slot;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = &ii;
    vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
}

// 1x1 white texture in every bindless slot, so unused slots are always valid
void initTextureSlots() {
    defaultTex = mkTexture(1, 1, 1);
    Buffer st = mkBuf(4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    memset(st.map, 0xFF, 4);
    oneShot([&](VkCommandBuffer cb) { recordTextureUpload(cb, defaultTex, 1, 1, 1, st.b, 0); });
    freeBuf(st);
    std::vector<VkDescriptorImageInfo> ii(MAX_TEXTURES, {texSampler, defaultTex.view, VK_IMAGE_LAYOUT_SHADER_READ_ONLY_OPTIMAL});
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = descSet;
    w.dstBinding = 32;
    w.descriptorCount = MAX_TEXTURES;
    w.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
    w.pImageInfo = ii.data();
    vkUpdateDescriptorSets(dev, 1, &w, 0, nullptr);
}

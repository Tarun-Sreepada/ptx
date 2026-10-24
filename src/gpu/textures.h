#pragma once
#include <cstdint>
#include <vulkan/vulkan.h>
#include "world/world.h"
#include "gpu/vk.h"

extern Image defaultTex;

extern VkSampler texSampler;

uint32_t mipCount(uint32_t w, uint32_t h);

Image mkTexture(uint32_t w, uint32_t h, uint32_t mips, VkFormat fmt = VK_FORMAT_R8G8B8A8_SRGB);

void recordTextureUpload(VkCommandBuffer cb, const Image& im, uint32_t w, uint32_t h, uint32_t mips, VkBuffer src, VkDeviceSize off);

void recordCompressedUpload(VkCommandBuffer cb, const Image& im, const TextureData& t, VkBuffer src, VkDeviceSize off);

void writeTexDescriptor(uint32_t slot, VkImageView view);

void initTextureSlots();

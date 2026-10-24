#pragma once
#include <vector>
#include <functional>
#include <cstdint>
#include <cstddef>
#include <vulkan/vulkan.h>
#include "gpu/vk.h"

extern std::vector<std::function<void()>> retireNext;

Buffer mkBuf(VkDeviceSize size, VkBufferUsageFlags usage, VkMemoryPropertyFlags req, VkMemoryPropertyFlags pref = 0);

void freeBuf(Buffer& b);

Image mkImageCI(const VkImageCreateInfo& ci, VkImageAspectFlags aspect);

void oneShot(const std::function<void(VkCommandBuffer)>& f);

Buffer uploadBuf(const void* data, size_t size, VkBufferUsageFlags usage);

void memBarrier(VkCommandBuffer cb, VkPipelineStageFlags ss, VkAccessFlags sa, VkPipelineStageFlags ds, VkAccessFlags da);

void imgBarrier(VkCommandBuffer cb, VkImage img, VkImageLayout from, VkImageLayout to, VkPipelineStageFlags ss, VkAccessFlags sa, VkPipelineStageFlags ds, VkAccessFlags da);

void computeBarrier(VkCommandBuffer cb);

Image mkImage(uint32_t w, uint32_t h, VkFormat fmt);

Image mkImageArray(uint32_t w, uint32_t h, uint32_t layers, VkFormat fmt);

void freeImage(Image& im);

Image mkImageUsage(uint32_t w, uint32_t h, VkFormat fmt, VkImageUsageFlags usage, uint32_t mips = 1);

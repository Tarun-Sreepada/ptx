#pragma once
#include <vulkan/vulkan.h>
#include <vk_mem_alloc.h>
#include <cstdio>
#include <cstdlib>

#define VK_CHECK(x)                                                                    \
    do {                                                                               \
        VkResult r_ = (x);                                                             \
        if (r_ != VK_SUCCESS) {                                                        \
            fprintf(stderr, "Vulkan error %d at %s:%d\n", (int)r_, __FILE__, __LINE__); \
            abort();                                                                   \
        }                                                                              \
    } while (0)

struct Buffer { VkBuffer b = VK_NULL_HANDLE; VmaAllocation m = VK_NULL_HANDLE; VkDeviceSize size = 0; void* map = nullptr; VkDeviceAddress addr = 0; };

struct Image { VkImage img = VK_NULL_HANDLE; VmaAllocation mem = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE; };

struct Accel { VkAccelerationStructureKHR as = VK_NULL_HANDLE; Buffer buf, scratch; VkDeviceAddress addr = 0, scratchAddr = 0; VkDeviceSize size = 0; };

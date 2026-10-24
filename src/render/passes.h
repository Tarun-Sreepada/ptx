#pragma once
#include <array>
#include <cstdint>
#include <vulkan/vulkan.h>

void dispatch(VkCommandBuffer cb, int p, uint32_t gx, uint32_t gy, std::array<uint32_t, 4> pc = {0, 0, 0, 0});

void copyImg(VkCommandBuffer cb, int src, int dst);

void rasterGBuffer(VkCommandBuffer cb);

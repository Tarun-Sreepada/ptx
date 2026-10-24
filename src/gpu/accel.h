#pragma once
#include <cstdint>
#include <vulkan/vulkan.h>
#include "gpu/vk.h"

VkAccelerationStructureGeometryKHR triGeom(const Buffer& vb, uint32_t stride, uint32_t nv, const Buffer& ib);

VkAccelerationStructureBuildGeometryInfoKHR buildInfo(VkAccelerationStructureTypeKHR type, VkBuildAccelerationStructureFlagsKHR flags, const VkAccelerationStructureGeometryKHR* g);

Accel mkAccel(VkAccelerationStructureTypeKHR type, VkBuildAccelerationStructureFlagsKHR flags, const VkAccelerationStructureGeometryKHR* g, uint32_t prims);

void cmdBuild(VkCommandBuffer cb, Accel& a, VkAccelerationStructureTypeKHR type, VkBuildAccelerationStructureFlagsKHR flags, const VkAccelerationStructureGeometryKHR* g, uint32_t prims, bool update);

void setInstance(VkAccelerationStructureInstanceKHR& in, uint32_t customIndex, VkDeviceAddress blas, const float m[3][4]);

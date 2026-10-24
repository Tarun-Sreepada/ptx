#include "gpu/accel.h"
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <vulkan/vulkan.h>
#include "gpu/context.h"
#include "gpu/memory.h"
#include "gpu/vk.h"
#include "render/frame_data.h"

// ---------------------------------------------------------------- acceleration structures
VkAccelerationStructureGeometryKHR triGeom(const Buffer& vb, uint32_t stride, uint32_t nv, const Buffer& ib) {
    VkAccelerationStructureGeometryKHR g{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
    g.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
    g.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
    auto& t = g.geometry.triangles;
    t.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
    t.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
    t.vertexData.deviceAddress = vb.addr;
    t.vertexStride = stride;
    t.maxVertex = nv - 1;
    t.indexType = VK_INDEX_TYPE_UINT32;
    t.indexData.deviceAddress = ib.addr;
    return g;
}

VkAccelerationStructureBuildGeometryInfoKHR buildInfo(VkAccelerationStructureTypeKHR type, VkBuildAccelerationStructureFlagsKHR flags,
                                                            const VkAccelerationStructureGeometryKHR* g) {
    VkAccelerationStructureBuildGeometryInfoKHR bi{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR};
    bi.type = type;
    bi.flags = flags;
    bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
    bi.geometryCount = 1;
    bi.pGeometries = g;
    return bi;
}

Accel mkAccel(VkAccelerationStructureTypeKHR type, VkBuildAccelerationStructureFlagsKHR flags, const VkAccelerationStructureGeometryKHR* g, uint32_t prims) {
    auto bi = buildInfo(type, flags, g);
    VkAccelerationStructureBuildSizesInfoKHR sz{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
    pGetASSizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &prims, &sz);
    Accel a;
    a.size = sz.accelerationStructureSize;
    a.buf = mkBuf(sz.accelerationStructureSize, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkAccelerationStructureCreateInfoKHR ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
    ci.buffer = a.buf.b;
    ci.size = sz.accelerationStructureSize;
    ci.type = type;
    VK_CHECK(pCreateAS(dev, &ci, nullptr, &a.as));
    VkAccelerationStructureDeviceAddressInfoKHR ai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
    ai.accelerationStructure = a.as;
    a.addr = pGetASAddr(dev, &ai);
    VkDeviceSize align = asProps.minAccelerationStructureScratchOffsetAlignment;
    a.scratch = mkBuf(std::max(sz.buildScratchSize, sz.updateScratchSize) + align, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    a.scratchAddr = (a.scratch.addr + align - 1) & ~(align - 1);
    return a;
}

void cmdBuild(VkCommandBuffer cb, Accel& a, VkAccelerationStructureTypeKHR type, VkBuildAccelerationStructureFlagsKHR flags,
                     const VkAccelerationStructureGeometryKHR* g, uint32_t prims, bool update) {
    auto bi = buildInfo(type, flags, g);
    if (update) {
        bi.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_UPDATE_KHR;
        bi.srcAccelerationStructure = a.as;
    }
    bi.dstAccelerationStructure = a.as;
    bi.scratchData.deviceAddress = a.scratchAddr;
    VkAccelerationStructureBuildRangeInfoKHR r{prims, 0, 0, 0};
    const VkAccelerationStructureBuildRangeInfoKHR* pr = &r;
    pCmdBuildAS(cb, 1, &bi, &pr);
}

void setInstance(VkAccelerationStructureInstanceKHR& in, uint32_t customIndex, VkDeviceAddress blas, const float m[3][4]) {
    in = {};
    memcpy(in.transform.matrix, m, sizeof in.transform.matrix);
    in.instanceCustomIndex = customIndex; // material id in shaders
    in.mask = customIndex == 0 ? 0x02 : 0x01;  // terrain on its own bit: crater rays can leave it out (common.glsl visibleEx)
    in.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR | VK_GEOMETRY_INSTANCE_FORCE_OPAQUE_BIT_KHR;
    in.accelerationStructureReference = blas;
}

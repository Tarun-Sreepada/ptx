#pragma once
#include <cstdint>
#include <vulkan/vulkan.h>

extern uint32_t maxBlades;

extern uint32_t bladeCap;

extern uint32_t lastBladeCount, grassGridN;

extern float terrainCenter[2];
extern float terrainBuildPos[2];
extern uint32_t terrainLevels;

extern bool terrainDirty;

// grass is regenerated (culled, LOD'd, compacted) every frame, so its BLAS is rebuilt rather than refit
inline const VkBuildAccelerationStructureFlagsKHR grassFlags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_BUILD_BIT_KHR |
                                                               VK_BUILD_ACCELERATION_STRUCTURE_ALLOW_UPDATE_BIT_KHR;

// Grass BLAS refit: while the camera stays put the blade set is kept and only re-animated (grass.comp pc.a = 1),
// and the BLAS is updated in place; a full rebuild every GRASS_REBUILD frames or when the camera moves / turns.
inline constexpr int GRASS_REBUILD = 8;

extern bool grassRefit, grassBuilt;

extern int grassFrames;

extern uint32_t grassBuildCap;

extern float grassBuildPos[3], grassBuildFwd[3];

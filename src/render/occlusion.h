#pragma once
#include <cstdint>
#include "core/math.h"
#include "gpu/vk.h"

// Occlusion culling against last frame's depth: occlusion.comp reduces the G-buffer's hit distances to the farthest
// one per OCC_TILE x OCC_TILE pixel tile (with the camera it was seen from) into a host-visible buffer, and the CPU
// culls tests each instance's bounding sphere against it before drawing. Only the raster G-buffer is culled: ray
// tracing (shadows, GI, reflections) still sees everything through the BVH.
inline constexpr uint32_t OCC_TILE = 16, OCC_MAX_W = 256, OCC_MAX_H = 160, OCC_HEADER = 16;  // floats before the tiles

extern Buffer occBuf;
extern uint32_t occCulled;  // instances hidden this frame (UI)

void createOcclusionBuffer();
// Snapshot last frame's grid (the fence has been waited on) for this frame's tests; call before the cull passes.
void beginOcclusion(V3 camPos);
// True when the sphere is certainly behind what last frame drew (counts it in occCulled).
bool occluded(V3 c, float r);

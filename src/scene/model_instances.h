#pragma once
#include <vector>
#include <cstdint>
#include <utility>
#include "world/world.h"
#include "scene/models.h"

extern std::vector<uint32_t> emisItems;

// placed instances of streamed meshes, rebuilt when the stream changes; bounds in SoA for SIMD culling
struct DrawItem { GpuModel* m; uint32_t mesh; Xf xf; };

extern std::vector<DrawItem> drawItems;

extern std::vector<float> aabb[6];

extern std::vector<uint8_t> visibleItems;

struct DrawRun { GpuModel* m; uint32_t mesh, first, count; };

extern std::vector<DrawRun> drawRuns;

extern uint32_t numModelInst, numModelVisible;

void rebuildModelInstances(const std::vector<std::pair<const MeshData*, DrawItem>>& items);

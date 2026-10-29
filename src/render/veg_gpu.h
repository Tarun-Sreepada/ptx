#pragma once
// GPU-driven vegetation culling: every placed tree and plant lives in a device buffer (re-uploaded only when
// placement changes), and vegcull.comp frustum- and occlusion-tests them each frame, writing the visible ones'
// raster transforms into modelXf (from VEG_XF_BASE on) and impostors into impBuf, and counting them into indirect
// draw commands: the CPU no longer touches a tree per frame. S.gpuCull off: the CPU path (culling_vegetation.cpp).
#include <cstdint>
#include <vulkan/vulkan.h>
#include "core/limits.h"
#include "gpu/vk.h"

inline constexpr uint32_t VEG_SLOTS = MAX_TREES + MAX_PLANTS;  // raster transform slots reserved for vegetation
inline constexpr uint32_t VEG_XF_BASE = MAX_MODEL_INST;        // first of them in modelXf
inline constexpr uint32_t XF_SLOTS = MAX_MODEL_INST + VEG_SLOTS;  // modelXf: XF_SLOTS current, then XF_SLOTS previous

extern Buffer vegInstBuf, vegCmdBuf, rockVisBuf, meshletBuf, meshletDataBuf, vegBucketBuf;  // rockVisBuf: the visible rocks (rockcull.comp)

void createVegGpu();
bool vegGpuOn();
bool vegMeshShading();  // trees and plants through raster_model.task / .mesh (meshlet culling)
// Bindings 107-112 (commands, instances, visible rocks, meshlets, meshlet data, buckets); also after a rebuild.
void writeVegDescriptors();
// After placement rebuilt treeList / plantList: refresh the instance records and the indirect template.
void uploadVegInstances();
// Counts from the last finished frame (UI): trees drawn as meshes, impostors, plants.
void readVegStats();
// The cull dispatches (graph pass "veg cull", graphics queue, before the raster G-buffer): rocks always,
// trees and plants when vegGpuOn().
void recordVegCull(VkCommandBuffer cb);
// The indirect draws inside the raster G-buffer pass.
void drawVegModels(VkCommandBuffer cb);
void drawVegImpostors(VkCommandBuffer cb);
void drawRocks(VkCommandBuffer cb);  // the rocks rockcull.comp kept (indirect)

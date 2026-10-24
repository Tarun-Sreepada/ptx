#include "world/worldmap_gpu.h"
#include <cstring>
#include <chrono>
#include <cstdint>
#include <cstddef>
#include <vulkan/vulkan.h>
#include "world/worldmap.h"
#include "world/world.h"
#include "gpu/context.h"
#include "gpu/memory.h"
#include "gpu/vk.h"
#include "scene/grass.h"
#include "scene/models.h"
#include "scene/vegetation.h"
#include "sim/world_sim.h"

// World map edits -> GPU: the dirty cell rectangle of each layer, through one staging buffer, into this frame's
// upload command buffer. Collision tiles and placement follow (throttled while a brush is held down).
void uploadWorldMap() {
    worldmap::Rect r = worldmap::takeDirty();
    static double lastCollision = 0;
    static uint32_t pendingVer = 0;
    double now = std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
    if (pendingVer && now - lastCollision > 0.3) {  // collision + placement: at most ~3x a second while editing
        stream::terrainChanged();
        placeDirty = true;
        lastCollision = now;
        pendingVer = 0;
    }
    if (r.x0 > r.x1) return;
    pendingVer = worldmap::version();
    terrainDirty = true;
    const int N = worldmap::N, w = r.x1 - r.x0 + 1, h = r.z1 - r.z0 + 1;
    int wx0 = r.x0 / 2, wz0 = r.z0 / 2, wx1 = r.x1 / 2, wz1 = r.z1 / 2, ww = wx1 - wx0 + 1, wh = wz1 - wz0 + 1;
    // each layer starts 16-byte aligned (copy offsets must be a multiple of the texel size: 16 for the water layer)
    size_t bh = ((size_t)w * h * 4 + 15) & ~size_t(15), bp = bh, bw = (size_t)ww * wh * 16;
    Buffer st = mkBuf(bh + bp + bw, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    char* m = (char*)st.map;
    for (int z = 0; z < h; z++) {
        memcpy(m + (size_t)z * w * 4, worldmap::heightData() + (size_t)(r.z0 + z) * N + r.x0, (size_t)w * 4);
        memcpy(m + bh + (size_t)z * w * 4, worldmap::paintData() + (size_t)(r.z0 + z) * N + r.x0, (size_t)w * 4);
    }
    for (int z = 0; z < wh; z++) memcpy(m + bh + bp + (size_t)z * ww * 16, worldmap::waterData() + (size_t)(wz0 + z) * (N / 2) + wx0, (size_t)ww * 16);
    beginUploads();
    auto copy = [&](const Image& im, VkDeviceSize off, int x, int z, int cw, int ch) {
        VkBufferImageCopy c{};
        c.bufferOffset = off;
        c.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        c.imageOffset = {x, z, 0};
        c.imageExtent = {(uint32_t)cw, (uint32_t)ch, 1};
        vkCmdCopyBufferToImage(uploadCmd, st.b, im.img, VK_IMAGE_LAYOUT_GENERAL, 1, &c);
    };
    memBarrier(uploadCmd, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_READ_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT);
    copy(mapHImg, 0, r.x0, r.z0, w, h);
    copy(mapPaintImg, bh, r.x0, r.z0, w, h);
    copy(mapWaterImg, bh + bp, wx0, wz0, ww, wh);
    memBarrier(uploadCmd, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_READ_BIT);
    retireNext.push_back([st]() mutable { freeBuf(st); });
}

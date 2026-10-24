#include "scene/stream_sync.h"
#include <vector>
#include <string>
#include <map>
#include <algorithm>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <utility>
#include <vulkan/vulkan.h>
#include "core/prof.h"
#include "world/matlib.h"
#include "world/world.h"
#include "core/limits.h"
#include "destruction/buildings.h"
#include "gpu/context.h"
#include "gpu/memory.h"
#include "gpu/textures.h"
#include "gpu/vk.h"
#include "render/bake.h"
#include "render/lights.h"
#include "scene/audio_sync.h"
#include "scene/grass.h"
#include "scene/model_instances.h"
#include "scene/models.h"
#include "scene/vegetation.h"

int loadRadius = 12;          // cells (64 m): ~768 m, matching the terrain horizon so buildings don't vanish in open ground

// a flatten zone's ground texture (matlib albedo) in a bindless slot, uploaded once per name; returns slot + 1
uint32_t zoneTexSlot(const std::string& name) {
    static std::map<std::string, uint32_t> slots;
    if (auto it = slots.find(name); it != slots.end()) return it->second;
    const TextureData& t = matlib::texture(name);
    uint32_t r = 0;
    if (t.w && t.h && texSlots.used() < texSlots.cap) {
        uint32_t mips = mipCount(t.w, t.h);
        Image im = mkTexture(t.w, t.h, mips, t.linear ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_R8G8B8A8_SRGB);
        Buffer st = mkBuf(t.rgba.size(), VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        memcpy(st.map, t.rgba.data(), t.rgba.size());
        beginUploads();  // with this frame's uploads, not a stall of its own
        recordTextureUpload(uploadCmd, im, t.w, t.h, mips, st.b, 0);
        retireNext.push_back([st]() mutable { freeBuf(st); });
        uint32_t s = texSlots.alloc();
        writeTexDescriptor(s, im.view);
        r = s + 1;
    }
    return slots[name] = r;
}

void onStreamChanged(bool moveOnly) {
    PROF_FUNC();
    meshPartsCache.clear();
    std::vector<std::pair<const MeshData*, DrawItem>> items;
    stream::forEachPlaced([&](const Placed& p) {
        auto* g = (GpuModel*)p.model->gpu;
        if (g && g->meshes[p.mesh].blas) items.push_back({&p.model->data.meshes[p.mesh], DrawItem{g, (uint32_t)p.mesh, p.xf}});
    });
    std::stable_sort(items.begin(), items.end(), [](auto& a, auto& b) {
        return a.second.m->meshes[a.second.mesh].slot < b.second.m->meshes[b.second.mesh].slot;
    });
    rebuildModelInstances(items);
    rebuildWorldLights();
    if (moveOnly) return;  // editor drag of a mesh / light: terrain, vegetation, bakes unchanged
    placeDirty = true;
    rebuildBakeVolumes(false);
    syncAudioInstances();
    auto& zs = stream::zones();
    float* z = (float*)zoneBuf.map;
    for (size_t i = 0; i < std::min<size_t>(zs.size(), MAX_ZONES); i++) {
        uint32_t slot = zs[i].mat.empty() ? 0 : zoneTexSlot(zs[i].mat);  // 0 = no texture, else slot + 1
        float v[8] = {zs[i].minx, zs[i].minz, zs[i].maxx, zs[i].maxz, zs[i].height, zs[i].falloff, (float)slot, zs[i].tile};
        memcpy(z + i * 8, v, sizeof v);
    }
    terrainDirty = true;
}

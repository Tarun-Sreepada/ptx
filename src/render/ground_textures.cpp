#include "render/ground_textures.h"

#include <chrono>
#include <cstdio>
#include <cstring>
#include <memory>

#include "app/settings.h"
#include "core/jobs.h"
#include "gpu/context.h"
#include "gpu/memory.h"
#include "gpu/textures.h"
#include "scene/models.h"
#include "world/texgen.h"

uint32_t groundSlots[4];
float groundMaxH[2];
bool groundReady = false;
uint32_t groundTexSize = 4096;

namespace {
constexpr float TILE_METRES = 4.f;  // one tile covers 4 m x 4 m (ground.glsl F.groundP.x)
std::unique_ptr<jobs::Async> loading;
texgen::Set sets[texgen::KIND_COUNT];

bool modeWantsTextures() { return S.groundMode == 0 || S.groundMode == 2; }

void upload() {
    for (int k = 0; k < texgen::KIND_COUNT; k++) {
        texgen::Set& set = sets[k];
        groundMaxH[k] = set.maxHeight;
        uint32_t n = set.size, mips = mipCount(n, n);
        size_t bytes = (size_t)n * n * 4;
        Buffer st = mkBuf(bytes * 2, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        memcpy(st.map, set.albedo.data(), bytes);
        memcpy((char*)st.map + bytes, set.detail.data(), bytes);
        Image img[2];
        for (int j = 0; j < 2; j++) {
            img[j] = mkTexture(n, n, mips, j == 0 ? VK_FORMAT_R8G8B8A8_SRGB : VK_FORMAT_R8G8B8A8_UNORM);
            groundSlots[2 * k + j] = texSlots.alloc();
        }
        beginUploads();  // with this frame's uploads: no stall when the background load lands
        for (int j = 0; j < 2; j++) recordTextureUpload(uploadCmd, img[j], n, n, mips, st.b, bytes * j);
        retireNext.push_back([st]() mutable { freeBuf(st); });
        for (int j = 0; j < 2; j++) writeTexDescriptor(groundSlots[2 * k + j], img[j].view);
        set = {};  // the CPU copy isn't needed any more
    }
}
}  // namespace

void updateGroundTextures() {
    if (groundReady || (!modeWantsTextures() && !loading)) return;
    if (!loading) {
        loading = std::make_unique<jobs::Async>([] {
            texgen::Params p;
            p.size = groundTexSize;
            p.tile = TILE_METRES;
            for (int k = 0; k < texgen::KIND_COUNT; k++) {
                auto t0 = std::chrono::steady_clock::now();
                bool cached = texgen::load((texgen::Kind)k, p, "texcache", sets[k]);
                printf("ground textures %d: %ux%u %s (%.1f s)\n", k, sets[k].size, sets[k].size, cached ? "from cache" : "generated",
                       std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count());
            }
        });
        return;
    }
    if (!loading->done()) return;
    loading.reset();
    upload();
    groundReady = true;
}

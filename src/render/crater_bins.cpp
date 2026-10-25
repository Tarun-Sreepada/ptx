#include "render/crater_bins.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include "world/terrain.h"
#include "gpu/vk.h"
#include "core/limits.h"
#include "scene/grass.h"

static constexpr float CB_CELL = 16.f;
static constexpr float HALF0 = 0.5f * TERRAIN_GRID * TERRAIN_SPACING0;  // finest clipmap level's half width

Buffer craterBinBuf;

int cbOrigin[2] = {0, 0};

bool cbValid = false;

void buildCraterBins(float cx, float cz) {
    if (!craterBinBuf.map) return;
    uint32_t* b = (uint32_t*)craterBinBuf.map;
    int ox = (int)floorf(cx / CB_CELL) - CB_N / 2, oz = (int)floorf(cz / CB_CELL) - CB_N / 2;
    int n = std::min(gCraterCount.load(std::memory_order_relaxed), MAX_CRATERS);
    static std::vector<uint32_t> lists[CB_BINS];
    for (auto& l : lists) l.clear();
    for (int i = 0; i < n; i++) {
        const Crater& c = gCraters[i];
        // the ejecta blanket's edge (terrain.h), and past it the mesh pit of a bowl smaller than the mesh cells
        // there (crater_surface.glsl craterLocal: 1.5 x the clipmap spacing at that distance)
        float d = std::max(hypotf(c.x - cx, c.z - cz) - c.r, 1e-3f);
        float level = std::clamp(ceilf(log2f(d / HALF0)), 0.f, (float)terrainLevels - 1.f);
        float reach = CRATER_EJECTA_REACH * c.r + std::max(1.f, 1.5f * TERRAIN_SPACING0 * exp2f(level));
        int x0 = (int)floorf((c.x - reach) / CB_CELL) - ox, x1 = (int)floorf((c.x + reach) / CB_CELL) - ox;
        int z0 = (int)floorf((c.z - reach) / CB_CELL) - oz, z1 = (int)floorf((c.z + reach) / CB_CELL) - oz;
        for (int z = std::max(z0, 0); z <= std::min(z1, CB_N - 1); z++)
            for (int x = std::max(x0, 0); x <= std::min(x1, CB_N - 1); x++) lists[z * CB_N + x].push_back((uint32_t)i);
    }
    uint32_t at = 0;
    bool overflow = false;
    for (int k = 0; k < CB_BINS; k++) {
        if (at + lists[k].size() > CB_LIST) { overflow = true; break; }
        b[2 * k] = at;
        b[2 * k + 1] = (uint32_t)lists[k].size();
        memcpy(b + 2 * CB_BINS + at, lists[k].data(), lists[k].size() * 4);
        at += (uint32_t)lists[k].size();
    }
    cbOrigin[0] = ox; cbOrigin[1] = oz;
    cbValid = !overflow;  // (overflow: the shaders fall back to looping over all craters)
}

#include "scene/placement/placement.h"

#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <vulkan/vulkan.h>
#include "core/prof.h"
#include "core/jobs.h"
#include "core/simd.h"
#include "vegetation/trees.h"
#include "world/worldmap.h"
#include "world/terrain.h"
#include "world/world.h"
#include "app/frame_state.h"
#include "app/settings.h"
#include "core/limits.h"
#include "core/math.h"
#include "destruction/rocks.h"
#include "destruction/thrown.h"
#include "destruction/tree_damage.h"
#include "gpu/accel.h"
#include "gpu/swapchain.h"
#include "render/bake.h"
#include "render/crater_bins.h"
#include "render/frame_ubo.h"
#include "render/ground_textures.h"
#include "render/ground_vt.h"
#include "render/light_build.h"
#include "render/lights.h"
#include "render/targets.h"
#include "scene/camera.h"
#include "scene/grass.h"
#include "scene/ground_mask.h"
#include "scene/model_instances.h"
#include "scene/models.h"
#include "scene/rocks.h"
#include "scene/scene_gpu.h"
#include "scene/stream_sync.h"
#include "scene/vegetation.h"
#include "sim/effects.h"
#include "sim/water.h"
#include "sim/water_splash.h"
#include "sim/world_sim.h"
#include "sky/celestial.h"
#include "sky/lightning.h"
#include "world/terrain_query.h"

// rocks: hashed 7 m grid out to the terrain's edge, kept while they cover ~1.5 px; buried rocks a crater dug up;
// hand-placed rocks; then the river's rocks (their fixed layout, the water shader evaluates the same)
void placeRocks(const PlacementCtx& c) {
    const float RR = c.RR, pixAngle = c.pixAngle, keepPx = c.keepPx;
    auto* inst = c.inst;
    float* xf = c.xf;
    const auto& zs = c.zs;
    (void)RR; (void)pixAngle; (void)keepPx; (void)inst; (void)xf; (void)zs;
    const float RC = 7.f;
    numRocks = 0;
    auto rocks = placeRows(RC, RR, [&](int ix, int iz, PlacedCell& pl) {
        float h = hash2(ix * 31 + 7, iz * 17 + 3);
        if (h > 0.16f) return false;
        float x = (ix + hash2(ix + 911, iz)) * RC, z = (iz + hash2(ix, iz + 577)) * RC;
        float sr = hash2(ix + 3, iz + 5);
        float sc = 0.25f + 0.9f * sr * sr + (h < 0.015f ? 1.8f : 0.f);
        float far = std::min(RR, 2.f * sc / (pixAngle * keepPx));  // rock is ~2*sc across
        if ((x - cam.pos.x) * (x - cam.pos.x) + (z - cam.pos.z) * (z - cam.pos.z) > far * far) return false;
        if (zoneCover(x, z, zs.data(), (int)zs.size()) > 0.f || inCrater(x, z, 0.f)) return false;  // no rocks on city ground / in craters
        if (!brokenRocks.empty() && brokenRocks.count(rockKey(x, z))) return false;  // knocked loose
        float yaw = hash2(ix + 13, iz + 29) * 6.2831853f, c = cosf(yaw) * sc, sn = sinf(yaw) * sc;
        float m[3][4] = {{c, 0, sn, x}, {0, sc, 0, heightFn(x, z) - 0.3f * sc}, {-sn, 0, c, z}};
        memcpy(pl.m, m, sizeof m);
        return true;
    });
    for (auto& row : rocks)
        for (auto& pl : row) {
            if (numRocks >= (uint32_t)MAX_ROCKS) break;
            uint32_t var = rockVarOf(pl.m[0][3], pl.m[2][3]);
            setRockInstance(inst[INST_MODELS + numModelInst + numRocks], var, pl.m);
            writeRockXf(xf, numRocks, pl.m, pl.m, var);
            numRocks++;
        }
    // buried rocks: the ground holds stones everywhere (a hashed 1.5 m grid, each at its own depth below the natural
    // surface), seen only where a crater has dug down to them. Deterministic, so a crater uncovers the same stones
    // wherever it is dug, and a deeper crater uncovers more.
    {
        const float BC = 1.5f;
        int nc = std::min(gCraterCount.load(std::memory_order_relaxed), MAX_CRATERS);
        for (int ci = 0; ci < nc && numRocks < (uint32_t)MAX_ROCKS; ci++) {
            const Crater& cr = gCraters[ci];
            float dx = cr.x - cam.pos.x, dz = cr.z - cam.pos.z;
            if (dx * dx + dz * dz > RR * RR) continue;
            for (int iz = (int)floorf((cr.z - cr.r) / BC); iz <= (int)floorf((cr.z + cr.r) / BC); iz++)
                for (int ix = (int)floorf((cr.x - cr.r) / BC); ix <= (int)floorf((cr.x + cr.r) / BC); ix++) {
                    float h = hash2(ix * 41 + 9, iz * 23 + 17);
                    if (h > 0.6f) continue;
                    float x = (ix + hash2(ix + 101, iz + 7)) * BC, z = (iz + hash2(ix + 13, iz + 211)) * BC;
                    float cx = x - cr.x, cz = z - cr.z;
                    if (cx * cx + cz * cz > cr.r * cr.r) continue;
                    // owned by the first crater that covers it (overlapping craters don't double it)
                    bool earlier = false;
                    for (int k = 0; k < ci && !earlier; k++) {
                        float ex = x - gCraters[k].x, ez = z - gCraters[k].z;
                        earlier = ex * ex + ez * ez < gCraters[k].r * gCraters[k].r;
                    }
                    if (earlier) continue;
                    float sr = hash2(ix + 3, iz + 61);
                    float sc = 0.15f + 0.6f * sr * sr;
                    float burial = 0.25f + 2.2f * hash2(ix + 77, iz + 5);  // top of the stone below the natural ground
                    float ground = heightFn(x, z), natural = terrainBase(x, z, zs.data(), (int)zs.size());
                    float dug = natural - ground;
                    if (dug < burial - 0.2f * sc) continue;  // still buried
                    if (!brokenRocks.empty() && brokenRocks.count(rockKey(x, z))) continue;
                    // bedded in the crater floor where the digging stopped
                    float yaw = hash2(ix + 19, iz + 3) * 6.2831853f, c = cosf(yaw) * sc, sn = sinf(yaw) * sc;
                    float m[3][4] = {{c, 0, sn, x}, {0, sc, 0, ground - 0.35f * sc}, {-sn, 0, c, z}};
                    uint32_t var = rockVarOf(x, z);
                    setRockInstance(inst[INST_MODELS + numModelInst + numRocks], var, m);
                    writeRockXf(xf, numRocks, m, m, var);
                    numRocks++;
                }
        }
    }
    for (auto& pr : stream::props()) {  // hand-placed rocks
        if (pr.kind != PROP_ROCK || numRocks >= (uint32_t)MAX_ROCKS) continue;
        uint32_t var = rockVarOf(pr.xf.m[0][3], pr.xf.m[2][3]);
        setRockInstance(inst[INST_MODELS + numModelInst + numRocks], var, pr.xf.m);
        writeRockXf(xf, numRocks, pr.xf.m, pr.xf.m, var);
        numRocks++;
    }
    numBreakableRocks = numRocks;
    // river rocks (terrain.h riverRock): fixed layout the water shader also evaluates
    for (int k = 0; k < RIVER_ROCK_CHANNELS; k++)  // every channel: main, braids, tributaries
    for (int i = (int)floorf((cam.pos.x - RR) / RIVER_ROCK_L); i <= (int)floorf((cam.pos.x + RR) / RIVER_ROCK_L) && numRocks < (uint32_t)MAX_ROCKS; i++) {
        float x, y, z, sc;
        if (!riverRock(i, k, x, y, z, sc)) continue;
        float far = std::min(RR, 2.f * sc / (pixAngle * keepPx));
        if ((x - cam.pos.x) * (x - cam.pos.x) + (z - cam.pos.z) * (z - cam.pos.z) > far * far) continue;
        float yaw = hash2(i + k * 1000003, 863) * 6.2831853f, c = cosf(yaw) * sc, sn = sinf(yaw) * sc;
        float m[3][4] = {{c, 0, sn, x}, {0, sc, 0, y}, {-sn, 0, c, z}};
        uint32_t var = rockVarOf(x, z);
        setRockInstance(inst[INST_MODELS + numModelInst + numRocks], var, m);
        writeRockXf(xf, numRocks, m, m, var);
        numRocks++;
    }
    // placed rocks don't move: writeRockXf already set previous = current
}

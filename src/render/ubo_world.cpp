#include "render/ubo_fill.h"
#include "render/frame_data.h"

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

void fillWorldState(UBO& u) {
    // terrain grid recentred in 0.5 m steps
    // terrain clipmap: enough levels to reach the view distance; rebuilt when the finest level recentres (every
    // 2 x its spacing) or the level count changes. All levels centre on the camera position of that rebuild.
    uint32_t levels = 1;
    while (levels < TERRAIN_LEVELS_MAX && terrainLevelReach(levels - 1) < S.viewDist) levels++;
    float snap = 2.f * TERRAIN_SPACING0;
    float cx = roundf(cam.pos.x / snap), cz = roundf(cam.pos.z / snap);
    if (cx != terrainCenter[0] || cz != terrainCenter[1] || levels != terrainLevels) {
        terrainCenter[0] = cx; terrainCenter[1] = cz;
        terrainBuildPos[0] = cam.pos.x; terrainBuildPos[1] = cam.pos.z;
        terrainLevels = levels;
        terrainTris = levels * TERRAIN_LEVEL_TRIS;
        terrainDirty = true;
    }
    u.terrainP[0] = terrainBuildPos[0]; u.terrainP[1] = terrainBuildPos[1]; u.terrainP[2] = TERRAIN_SPACING0;
    u.terrainP[3] = terrainLevelReach(terrainLevels - 1);
    u.terrain2[0] = (float)terrainLevels; u.terrain2[1] = (float)TERRAIN_GRID; u.terrain2[2] = TERRAIN_MORPH_BAND; u.terrain2[3] = 0.f;
    u.counts[0] = (uint32_t)std::min<size_t>(stream::zones().size(), MAX_ZONES); u.counts[1] = (uint32_t)S.giSpatialSamples; u.counts[2] = (uint32_t)S.candidates; u.counts[3] = (uint32_t)S.spatialSamples;
    // sub-pixel jitter for DLSS-RR: Halton(2,3), 64 phases (RR wants many), in render pixels
    prevJitter[0] = jitter[0]; prevJitter[1] = jitter[1];
    if (rrActive) {
        auto halton = [](uint32_t i, uint32_t b) { float f = 1, r = 0; for (i++; i; i /= b) { f /= b; r += f * (i % b); } return r; };
        jitter[0] = halton(frameIndex % 64, 2) - 0.5f;
        jitter[1] = halton(frameIndex % 64, 3) - 0.5f;
    } else jitter[0] = jitter[1] = 0;
    u.river[0] = gRiver.width; u.river[1] = gRiver.depth; u.river[2] = gRiver.level; u.river[3] = gRiver.slope;
    u.water[0] = S.flow; u.water[1] = S.clarity; u.water[2] = S.foam; u.water[3] = S.splash;
    u.water2[0] = S.foamTrail; u.water2[1] = S.viewDist; u.water2[2] = S.treeDensity; u.water2[3] = gRiver.bumps;
    u.waterCol[0] = S.murk[0]; u.waterCol[1] = S.murk[1]; u.waterCol[2] = S.murk[2]; u.waterCol[3] = S.absorb;
    u.dynInfo[0] = (uint32_t)thrown.size();
    for (size_t i = 0; i < thrown.size(); i++) {
        const Thrown& t = thrown[i];
        float d[4] = {t.xf.m[0][3], t.xf.m[1][3], t.xf.m[2][3], t.radius / 0.85f};
        memcpy(u.dynRocks[i], d, sizeof d);
        float v[4] = {t.v[0], t.v[1], t.v[2], 0.f};
        memcpy(u.dynRockV[i], v, sizeof v);
    }
    u.dynInfo[1] = (uint32_t)std::min(splashEvents.size(), (size_t)MAX_SPLASH_EV);
    for (uint32_t i = 0; i < u.dynInfo[1]; i++) {
        const SplashEvent& e = splashEvents[i];
        float d[8] = {e.x, e.z, e.t, e.impact, e.sc, e.speed, 0, 0};
        memcpy(u.splashEv[2 * i], d, sizeof d);
    }
    u.dynInfo[2] = (uint32_t)std::min(gCraterCount.load(), MAX_CRATERS);
    memcpy(u.craters, gCraters, sizeof(Crater) * u.dynInfo[2]);
    for (uint32_t i = 0; i < u.dynInfo[2]; i++) {
        const CraterWater& w = craterWater[i];
        u.craterG[i][0] = w.ground[0]; u.craterG[i][1] = gCraters[i].gx; u.craterG[i][2] = gCraters[i].gz; u.craterG[i][3] = gCraters[i].base;
        u.craterW[i][0] = w.pond ? w.level : -1e9f; u.craterW[i][1] = w.fill; u.craterW[i][2] = w.divert; u.craterW[i][3] = w.inflowSpeed;
    }
    u.jitter[0] = jitter[0]; u.jitter[1] = jitter[1]; u.jitter[2] = prevJitter[0]; u.jitter[3] = prevJitter[1];
}

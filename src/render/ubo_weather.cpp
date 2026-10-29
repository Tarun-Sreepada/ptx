#include "render/ubo_fill.h"
#include "sim/fire.h"
#include "sim/trample.h"
#include "sky/day_cycle.h"

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

void fillWeather(UBO& u) {
        // the fog pass also runs (without fog) while there's dust to show
        set4(u.fog0, V3{S.fog ? S.fogDensity : 0.f, S.fogFalloff, S.fogAniso}, S.fog || !dustList.empty() ? 1.f : 0.f);
        set4(u.fog1, V3{S.fogLights ? 1.f : 0.f, S.fogDist, S.fogNoise}, S.fogBlend);
        static double lastWetT = simTime;
        float wdt = std::clamp((float)(simTime - lastWetT), 0.f, 0.25f);
        lastWetT = simTime;
        if (S.rainWets) {
            // rain only below the snow line (above it the precipitation is snow)
            float rainHere = cam.pos.y < S.snowLine - 15.f ? S.rain : 0.f;
            // evaporation: faster in sun and wind
            float dryK = 0.5f + 1.2f * std::max(sinf(S.sunElev * 3.14159265f / 180.f), 0.f) * (S.clouds ? 1.f - 0.6f * S.cloudCoverage : 1.f) +
                         0.8f * S.windStrength;
            auto ease = [&](float& v, float target, float up, float down) {
                v += (target - v) * (1.f - expf(-wdt * (target > v ? up : down)));
            };
            // surface film: wet within ~20 s of rain, runs off / evaporates within a couple of minutes
            ease(S.wetness, std::min(rainHere * 1.5f, 1.f), 0.05f, 0.012f * dryK);
            // soaked into porous material: darkens over a minute, dries over ~10-20 minutes
            ease(S.soak, std::min(rainHere * 2.f, 1.f), 0.015f, 0.0015f * dryK);
            // canopies and eaves hold water and drip for a minute or so after the rain stops
            ease(S.drip, std::min(rainHere * 3.f, 1.f), 0.1f, 0.02f);
        }
        u.weather[0] = S.wetness; u.weather[1] = S.rain; u.weather[2] = S.puddles; u.weather[3] = S.caustics;
        u.wet2[0] = S.soak; u.wet2[1] = S.drip; u.wet2[2] = S.rivulets; u.wet2[3] = 0.f;
        writeLightning(u);
        writeFires(u);
        set4(u.rainbow, V3{S.rainbow * std::min(S.rain * 2.f, 1.f), S.dropMM * 0.5f, 400.f}, S.rainbow > 0.f && S.rain > 0.f ? 1.f : 0.f);
        {   // fog banks drift with the wind (and slowly change shape)
            static double lastBankT = simTime;
            static float drift[2] = {0.f, 0.f};
            float bdt = std::clamp((float)(simTime - lastBankT), 0.f, 0.25f);
            lastBankT = simTime;
            float wd = S.windDirDeg * 3.14159265f / 180.f, ws = S.windStrength * S.windSpeed * 2.5f;  // ~1-4 m/s near the ground
            drift[0] -= cosf(wd) * ws * bdt; drift[1] -= sinf(wd) * ws * bdt;
            set4(u.fogBank, V3{S.bankDensity, S.bankScale, S.bankTop}, S.bankCover);
            set4(u.fogBank2, V3{drift[0], drift[1], S.bankDetail}, S.fogBanks ? 1.f : 0.f);
        }
        // world map + simulation
        for (int i = 0; i < 8; i++) {
            const std::string& nm = worldmap::paletteName(i);
            u.mapPal[i] = nm.empty() ? 0u : zoneTexSlot(nm);
            u.mapPalTile[i] = S.mapTexTile;
        }
        u.mapP[0] = worldmap::ORIGIN; u.mapP[1] = worldmap::CELL; u.mapP[2] = (float)worldmap::N; u.mapP[3] = 0.f;
        if (S.snowFollowsClouds) S.snowLine = S.dayCycle ? std::min(S.cloudBase, seasonSnowLine(S.dayOfYear)) : S.cloudBase;  // winter brings it down
        u.sim0[0] = S.snowLine; u.sim0[1] = S.rain; u.sim0[2] = S.meltRate / 3600.f; u.sim0[3] = S.riverRange;
        u.sim1[0] = tanf(S.reposeDeg * 3.14159265f / 180.f); u.sim1[1] = S.avalanche; u.sim1[2] = std::clamp(frameDt * S.simSpeed, 0.001f, 0.5f);
        u.sim1[3] = S.worldSim ? 1.f : 0.f;
        u.sim2[0] = S.snowfall / 3600.f; u.sim2[1] = S.dryRate; u.sim2[2] = S.waterMobility; u.sim2[3] = riverSwell;
        memcpy(u.simBrush, gSimBrush, 16);
        buildCraterBins(cam.pos.x, cam.pos.z);
        int fd = std::clamp(S.fogDiv, 1, 128);
        u.fogRes[0] = fd; u.fogRes[1] = ((int)W + fd - 1) / fd; u.fogRes[2] = ((int)H + fd - 1) / fd; u.fogRes[3] = S.godRays ? 1 : 0;
        int cd = std::clamp(S.cloudDiv, 1, 16);
        u.cloudRes[0] = cd; u.cloudRes[1] = ((int)W + cd - 1) / cd; u.cloudRes[2] = ((int)H + cd - 1) / cd; static int lastCd = 0;
        static uint32_t cloudFrame = 0;
        u.cloudRes[3] = (int)(++cloudFrame & 1u) | (cd != lastCd ? 2 : 0);
        lastCd = cd;
        u.craterGrid[0] = cbOrigin[0]; u.craterGrid[1] = cbOrigin[1]; u.craterGrid[2] = CB_N; u.craterGrid[3] = cbValid ? 1 : 0;
        u.fractal[0] = (float)S.groundMode; u.fractal[1] = S.fractalRelief; u.fractal[2] = S.fractalPebbles; u.fractal[3] = S.fractalFinest;
        uint32_t nd = 0;
        for (const Decal& d : decalList) {
            float age = std::min((float)(simTime - d.t0), 999.f);
            if (age > 300.f) continue;
            float a[8] = {d.p[0], d.p[1], d.p[2], d.r, d.n[0], d.n[1], d.n[2], (float)d.kind + age / 1000.f};
            memcpy(u.decals[2 * nd], a, sizeof a);
            nd++;
        }
        u.lightInfo[3] = nd;
        writeTrample(u);  // footprints after the decals, grass pushers
        dustList.erase(std::remove_if(dustList.begin(), dustList.end(), [](const Dust& d) { return simTime - d.t0 > 15.0; }), dustList.end());
        uint32_t nu = 0;
        for (const Dust& d : dustList) {
            float a[8] = {d.p[0], d.p[1], d.p[2], d.r, d.density, (float)(simTime - d.t0), d.rise, 0};
            memcpy(u.dust[2 * nu], a, sizeof a);
            nu++;
        }
        u.fxInfo[0] = nu;
}

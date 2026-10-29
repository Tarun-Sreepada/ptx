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
#include "world/biome.h"
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

void placePlants(const PlacementCtx& c) {
    const float RR = c.RR, pixAngle = c.pixAngle, keepPx = c.keepPx;
    auto* inst = c.inst;
    float* xf = c.xf;
    const auto& zs = c.zs;
    (void)RR; (void)pixAngle; (void)keepPx; (void)inst; (void)xf; (void)zs;
    // plants: hashed 2.5 m grid within PLANT_RANGE. Cattail beds in damp patches, dandelion meadows, roses and
    // shrubs scattered everywhere (thicker along forest edges); dropped once under ~3 px like rocks.
    numPlants = 0;
    plantList.clear();
    if (plantModel.gpu) {
        auto* g = (GpuModel*)plantModel.gpu;
        const float PC = 2.5f;
        auto sstepP = [](float a, float b, float x) { float t = std::clamp((x - a) / (b - a), 0.f, 1.f); return t * t * (3 - 2 * t); };
        uint32_t base = INST_MODELS + numModelInst + numRocks + numTrees;
        auto plantsP = placeRows(PC, PLANT_RANGE, [&](int ix, int iz, PlacedCell& pl) {
            float x = (ix + hash2(ix + 401, iz + 13)) * PC, z = (iz + hash2(ix + 17, iz + 409)) * PC;
            float h = hash2(ix * 29 + 11, iz * 23 + 5) / std::max(S.treeDensity, 0.01f);
            float wet = vnoise(x * 0.02f + 90.f, z * 0.02f - 40.f), meadow = vnoise(x * 0.035f - 13.f, z * 0.035f + 77.f);
            float edge = vnoise(x * 0.0045f + 31.f, z * 0.0045f - 7.f);  // same low-frequency forest noise as the trees
            uint32_t v;
            float bank = 1e9f;
            if (riverDry(x, z, 3.5f) == 0.f) {  // riverside: reed beds right at the waterline, nothing in deeper water
                RiverInfo ri = riverInfo(x, z);
                bank = fabsf(ri.d) - ri.w;
                if (bank < -0.8f) return false;
            }
            if (bank < 3.5f && h < 0.6f * sstepP(0.35f, 0.6f, vnoise(x * 0.06f + 7.f, z * 0.06f))) v = trees::PLANT_CATTAIL;  // reed beds in stretches
            else if (bank < 0.5f) return false;
            else if (wet > 0.75f && h < 0.35f) v = trees::PLANT_CATTAIL;
            else if (meadow > 0.58f && h < 0.4f) v = trees::PLANT_DANDELION;
            else if (h < (0.025f + 0.08f * std::clamp((edge - 0.42f) * 6.f, 0.f, 1.f)) * S.bushAmount) {
                float k = hash2(ix + 3, iz + 61);
                v = k < S.berryShare ? trees::PLANT_BERRYBUSH : (k < S.berryShare + (1.f - S.berryShare) * 0.5f ? trees::PLANT_BUSH : trees::PLANT_FLOWERBUSH);
            }
            else if (h < 0.035f + 0.03f * meadow) v = trees::PLANT_ROSE;
            else return false;
            float sc = 0.75f + 0.5f * hash2(ix + 23, iz + 7);
            float far = std::min(PLANT_RANGE, 2.f * plantR[v] * sc / (pixAngle * 3.f));
            if ((x - cam.pos.x) * (x - cam.pos.x) + (z - cam.pos.z) * (z - cam.pos.z) > far * far) return false;
            if (zoneCover(x, z, zs.data(), (int)zs.size()) > 0.f || inCrater(x, z, 0.f)) return false;
            float y;  // (deserts, ice, salt, rock: none; nor above the shrub line or on steep faces)
            if (hash2(ix + 631, iz + 47) >= habitatAt(x, z, false, y) * std::min(biomeGrassMul(x, z), 1.f)) return false;
            float yaw = hash2(ix + 5, iz + 37) * 6.2831853f, c = cosf(yaw) * sc, sn = sinf(yaw) * sc;
            float m[3][4] = {{c, 0, sn, x}, {0, sc, 0, y - 0.02f}, {-sn, 0, c, z}};
            memcpy(pl.m, m, sizeof m);
            pl.t.v = v;
            return true;
        });
        // ground clutter: denser 1.2 m grid close by. Ferns, twigs and leaf litter under the forest (same forest noise as
        // the trees), pebbles on bare ground and along the river banks.
        const float CC = 1.2f;
        auto clutterP = placeRows(CC, S.clutterRange, [&](int ix, int iz, PlacedCell& pl) {
            if (S.clutter <= 0.f) return false;
            float x = (ix + hash2(ix + 911, iz + 37)) * CC, z = (iz + hash2(ix + 41, iz + 977)) * CC;
            float h = hash2(ix * 31 + 7, iz * 17 + 3) / S.clutter;
            float forest = vnoise(x * 0.0045f + 31.f, z * 0.0045f - 7.f);
            float bare = 1.f - grassMaskCPU(x, z);
            uint32_t v;
            float bank = 1e9f;
            if (riverDry(x, z, 2.f) == 0.f) {
                RiverInfo ri = riverInfo(x, z);
                bank = fabsf(ri.d) - ri.w;
                if (bank < -0.3f) return false;
            }
            if (bank < 2.5f && h < 0.5f) v = trees::PLANT_PEBBLES;
            else if (forest > 0.45f && h < 0.22f * (forest - 0.3f) * 4.f) {
                float k = hash2(ix + 71, iz + 5);
                v = k < 0.45f ? trees::PLANT_LITTER : (k < 0.75f ? trees::PLANT_TWIGS : trees::PLANT_FERN);
            } else if (h < 0.06f + 0.25f * bare) v = trees::PLANT_PEBBLES;
            else if (h < 0.09f + 0.25f * bare) v = trees::PLANT_TWIGS;
            else return false;
            float sc = 0.7f + 0.6f * hash2(ix + 29, iz + 11);
            float far = std::min(S.clutterRange, 2.f * plantR[v] * sc / (pixAngle * 2.f));
            if ((x - cam.pos.x) * (x - cam.pos.x) + (z - cam.pos.z) * (z - cam.pos.z) > far * far) return false;
            if (zoneCover(x, z, zs.data(), (int)zs.size()) > 0.f) return false;
            float y;
            if (v != trees::PLANT_PEBBLES) {
                if (hash2(ix + 653, iz + 71) >= habitatAt(x, z, false, y) * std::min(biomeGrassMul(x, z), 1.f)) return false;
            } else y = heightFn(x, z);
            float yaw = hash2(ix + 3, iz + 53) * 6.2831853f, c = cosf(yaw) * sc, sn = sinf(yaw) * sc;
            float m[3][4] = {{c, 0, sn, x}, {0, sc, 0, y - 0.01f}, {-sn, 0, c, z}};
            memcpy(pl.m, m, sizeof m);
            pl.t.v = v;
            return true;
        });
        for (auto* rows : {&plantsP, &clutterP})
            for (auto& row : *rows)
                for (auto& pl : row) {
                    if (numPlants >= (uint32_t)MAX_PLANTS) break;
                    setInstance(inst[base + numPlants], 3 + g->meshes[pl.t.v].slot, g->meshes[pl.t.v].blasAddr, pl.m);
                    PlantInst pi;
                    memcpy(pi.m, pl.m, 48);
                    pi.v = pl.t.v;
                    plantList.push_back(pi);
                    numPlants++;
                }
    }
}

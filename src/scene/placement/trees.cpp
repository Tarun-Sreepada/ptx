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

namespace {
constexpr float CONIFER_STAND_FREQ = 0.004f;   // 1/m: conifer or broadleaf woods, ~250 m across
constexpr float CONIFER_SHARE = 0.45f;         // of the land, where both kinds exist
constexpr float PATCH_FREQ = 0.012f;           // 1/m: patches with one dominant species, ~80 m
constexpr float MIXED_IN = 0.25f;              // share of trees from the rest of the stand

// Species stands like a real forest: big conifer and broadleaf woods; in each, patches dominated by one species
// with the others mixed in. Needle or leaf comes from the species' current parameters (the sliders can change it).
uint32_t pickSpecies(float x, float z, int ix, int iz) {
    uint32_t con[trees::SPECIES], brd[trees::SPECIES];
    int nc = 0, nb = 0;
    for (int s = 0; s < trees::SPECIES; s++) {
        if (trees::conifer(trees::treeParams[s])) con[nc++] = (uint32_t)s;
        else brd[nb++] = (uint32_t)s;
    }
    auto even = [](float n) { return std::clamp((n - 0.5f) * 2.2f + 0.5f, 0.f, 0.999f); };  // value noise -> ~uniform
    bool needles = nc > 0 && (nb == 0 || even(vnoise(x * CONIFER_STAND_FREQ + 50.f, z * CONIFER_STAND_FREQ - 3.f)) < CONIFER_SHARE);
    const uint32_t* set = needles ? con : brd;
    int n = needles ? nc : nb;
    if (hash2(ix + 311, iz + 73) < MIXED_IN) return set[std::min((int)(hash2(ix + 17, iz + 91) * n), n - 1)];
    return set[std::min((int)(even(vnoise(x * PATCH_FREQ - 20.f, z * PATCH_FREQ + 9.f)) * n), n - 1)];
}
}  // namespace

void placeTrees(const PlacementCtx& c) {
    const float RR = c.RR, pixAngle = c.pixAngle, keepPx = c.keepPx;
    auto* inst = c.inst;
    float* xf = c.xf;
    const auto& zs = c.zs;
    (void)RR; (void)pixAngle; (void)keepPx; (void)inst; (void)xf; (void)zs;
    // trees: hashed 9 m grid; forests where a low-frequency noise is high, the odd lone tree elsewhere.
    // Same screen-size rule as rocks; the BVH gets the real mesh at every distance.
    numTrees = 0;
    treeList.clear();
    if (treeModel.gpu) {
        auto* g = (GpuModel*)treeModel.gpu;
        const float TC = 9.f;
        auto sstep = [](float a, float b, float x) { float t = std::clamp((x - a) / (b - a), 0.f, 1.f); return t * t * (3 - 2 * t); };
        // the tree a grid cell would hold, before camera / town / water rules: a pure function of the cell so
        // neighbours can be evaluated for the footprint test
        struct Cand { float x, z, sc; uint32_t v; };
        auto candidate = [&](int ix, int iz, Cand& cd) {
            float x = (ix + 0.15f + 0.7f * hash2(ix + 71, iz + 3)) * TC, z = (iz + 0.15f + 0.7f * hash2(ix + 5, iz + 191)) * TC;
            float forest = vnoise(x * 0.0045f + 31.f, z * 0.0045f - 7.f) * 0.7f + vnoise(x * 0.017f, z * 0.017f + 11.f) * 0.3f;
            float painted = worldmap::treeMul(x, z);  // world map tree brush: 0 = cleared, 2 = double, also outside forests
            if (hash2(ix * 13 + 5, iz * 7 + 1) > ((sstep(0.5f, 0.66f, forest) * 0.8f + 0.025f) * painted + std::max(painted - 1.f, 0.f) * 0.35f) * S.treeDensity * biomeTreeMul(x, z))
                return false;
            float sc = 0.75f + 0.5f * hash2(ix + 19, iz + 41);
            uint32_t v;
            if (numTreeVariants == trees::VARIANTS)
                v = pickSpecies(x, z, ix, iz) * 2u + (hash2(ix + 7, iz + 77) > 0.5f ? 1u : 0u);  // the seed varies tree to tree
            else {  // weighted pick from the forest mix
                float h = hash2(ix + 7, iz + 77) * treeWeightSum;
                v = (uint32_t)numTreeVariants - 1;
                for (int k = 0; k < numTreeVariants; k++) if ((h -= treeWeight[k]) < 0.f && treeWeight[k] > 0.f) { v = (uint32_t)k; break; }
                if (treeWeight[v] <= 0.f) return false;
            }
            cd = {x, z, sc, v};
            return true;
        };
        float maxFoot = 0.f;
        for (int k = 0; k < numTreeVariants; k++) maxFoot = std::max(maxFoot, treeFoot[k] * 1.25f * forestSpacing);
        const int reach = (int)ceilf(2.f * maxFoot / TC);
        auto treesP = placeRows(TC, RR, [&](int ix, int iz, PlacedCell& pl) {
            Cand cd;
            if (!candidate(ix, iz, cd)) return false;
            if (!brokenTrees.empty() && brokenTrees.count(treeKey(ix, iz))) return false;  // knocked down
            float x = cd.x, z = cd.z, sc = cd.sc;
            uint32_t v = cd.v;
            // bounding footprints (x spacing) may not overlap: of two overlapping candidates the higher priority stays
            if (reach > 0) {
                float ra = treeFoot[v] * sc * forestSpacing, pa = hash2(ix + 911, iz + 37);
                for (int dz = -reach; dz <= reach; dz++)
                    for (int dx = -reach; dx <= reach; dx++) {
                        Cand o;
                        if ((!dx && !dz) || !candidate(ix + dx, iz + dz, o)) continue;
                        float rb = treeFoot[o.v] * o.sc * forestSpacing, rr = ra + rb;
                        if (rr <= 0.f || (o.x - x) * (o.x - x) + (o.z - z) * (o.z - z) >= rr * rr) continue;
                        float pb = hash2(ix + dx + 911, iz + dz + 37);
                        if (pb > pa || (pb == pa && (dx > 0 || (dx == 0 && dz > 0)))) return false;
                    }
            }
            float far = std::min(RR, 2.f * treeR[v] * sc / (pixAngle * 2.f));
            if ((x - cam.pos.x) * (x - cam.pos.x) + (z - cam.pos.z) * (z - cam.pos.z) > far * far) return false;
            if (zoneCover(x, z, zs.data(), (int)zs.size()) > 0.f || riverDry(x, z, 2.5f) == 0.f || inCrater(x, z, 0.f)) return false;  // not in town / water / craters
            float y;
            if (hash2(ix + 613, iz + 29) >= habitatAt(x, z, true, y)) return false;  // above the tree line / too steep
            y -= 0.05f;
            float yaw = hash2(ix + 3, iz + 9) * 6.2831853f, c = cosf(yaw) * sc, sn = sinf(yaw) * sc;
            float m[3][4] = {{c, 0, sn, x}, {0, sc, 0, y}, {-sn, 0, c, z}};
            memcpy(pl.m, m, sizeof m);
            pl.t = {{x, y, z}, yaw, sc, v, ix, iz};
            return true;
        });
        for (auto& row : treesP)
            for (auto& pl : row) {
                if (numTrees >= (uint32_t)MAX_TREES) break;
                setInstance(inst[INST_MODELS + numModelInst + numRocks + numTrees], 3 + g->meshes[pl.t.v].slot, g->meshes[pl.t.v].blasAddr, pl.m);
                treeList.push_back(pl.t);
                numTrees++;
            }
        // hand-placed (editor / Blender: extras tree = <variant>); keyed outside the grid so breaking works the same
        for (auto& pr : stream::props()) {
            if (pr.kind != PROP_TREE || numTrees >= (uint32_t)MAX_TREES) continue;
            const Xf& x = pr.xf;
            float sc = sqrtf(x.m[0][0] * x.m[0][0] + x.m[1][0] * x.m[1][0] + x.m[2][0] * x.m[2][0]);
            float yaw = atan2f(x.m[0][2], x.m[2][2]);
            int ix = 0x40000000 | ((int)lroundf(x.m[0][3] * 10.f) & 0x3FFFFFFF), iz = (int)lroundf(x.m[2][3] * 10.f);
            if (!brokenTrees.empty() && brokenTrees.count(treeKey(ix, iz))) continue;
            uint32_t v = (uint32_t)std::clamp(pr.variant, 0, numTreeVariants - 1);
            float c = cosf(yaw) * sc, sn = sinf(yaw) * sc;
            float m[3][4] = {{c, 0, sn, x.m[0][3]}, {0, sc, 0, x.m[1][3]}, {-sn, 0, c, x.m[2][3]}};
            setInstance(inst[INST_MODELS + numModelInst + numRocks + numTrees], 3 + g->meshes[v].slot, g->meshes[v].blasAddr, m);
            treeList.push_back({{x.m[0][3], x.m[1][3], x.m[2][3]}, yaw, sc, v, ix, iz});
            numTrees++;
        }
    }
}

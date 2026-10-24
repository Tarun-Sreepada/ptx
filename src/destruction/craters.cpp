#include "destruction/craters.h"
#include "sim/fire.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <vector>

#include "core/random.h"
#include "destruction/tree_break.h"
#include "destruction/wood.h"
#include "scene/grass.h"
#include "scene/vegetation.h"
#include "sim/debris.h"
#include "sim/effects.h"
#include "vegetation/trees.h"
#include "world/terrain.h"
#include "world/terrain_query.h"
#include "world/world.h"

namespace {
constexpr float SOIL_STRENGTH = 1.2e5f;         // J per m^3 of loosened soil
constexpr float BOWL_VOLUME = 0.55f;             // excavated volume / R^3 for the crater proportions
constexpr float MIN_RADIUS = 0.3f;               // smaller dents aren't craters
constexpr float MAX_RADIUS = 25.f;
constexpr float MERGE_SIZE = 0.6f;               // a hit inside a bowl enlarges it when its own crater would be this big (of R)
constexpr float MERGE_REACH = 0.5f;              // ... and it lands this close to the centre (of R)
constexpr float FRAGMENT_MIN_ENERGY = 5e4f;      // a fragment of what already broke digs only above this (J)
constexpr float MAX_SLOPE = 1.f;                 // impact plane slope cap (cliffs: the bowl doesn't tilt past 45 deg)
constexpr float BLAST_ENERGY = 2e5f;             // above this the shock knocks trees over
constexpr int MAX_BLAST_TREES = 24;
constexpr float FIRE_ENERGY = 3e7f;             // an impact this hot sets the crater burning (J)
constexpr float FIRE_SIZE = 0.25f;              // the fire's radius / the crater's
constexpr float FIRE_BURN = 40.f;               // s

float radiusFor(float volume) { return std::min(cbrtf(volume / BOWL_VOLUME), MAX_RADIUS); }

// The ground the impact digs into: the surface as it is now (older craters, their ejecta, zones), as a plane across
// the crater's radius. h(x, z) = that surface.
template <class H> void settle(Crater& c, H h) {
    float e = std::max(0.5f, 0.5f * c.r);
    c.base = h(c.x, c.z);
    c.gx = std::clamp((h(c.x + e, c.z) - h(c.x - e, c.z)) / (2.f * e), -MAX_SLOPE, MAX_SLOPE);
    c.gz = std::clamp((h(c.x, c.z + e) - h(c.x, c.z - e)) / (2.f * e), -MAX_SLOPE, MAX_SLOPE);
}

// the bowl at (x, z), if any: the biggest crater whose rim is within reach * r
int bowlAt(float x, float z, float reach) {
    int n = std::min(gCraterCount.load(std::memory_order_relaxed), MAX_CRATERS), best = -1;
    for (int i = 0; i < n; i++) {
        float dx = gCraters[i].x - x, dz = gCraters[i].z - z, r = reach * gCraters[i].r;
        if (dx * dx + dz * dz < r * r && (best < 0 || gCraters[i].r > gCraters[best].r)) best = i;
    }
    return best;
}

void ejecta(V3 at, float R) {
    addDecal(at, {0, 1, 0}, R * 2.2f, 1);  // thrown-out soil around the crater
    addDust(at + V3{0, 0.3f * R, 0}, R * 1.1f, 0.35f * std::min(R, 3.f), 0.5f);
    int clods = std::min((int)(25.f * R * R), 300);
    for (int i = 0; i < clods; i++) {
        float a = rnd01() * 6.2831853f, r0 = R * sqrtf(rnd01()) * 0.8f, sp = (2.f + 5.f * rnd01()) * sqrtf(R);
        float elev = 0.6f + 0.6f * rnd01();
        addDebris(DEB_DIRT, at + V3{cosf(a) * r0, 0.1f, sinf(a) * r0}, V3{cosf(a) * cosf(elev), sinf(elev), sinf(a) * cosf(elev)} * sp,
                  (0.015f + 0.07f * rnd01() * rnd01() * rnd01()) * std::min(sqrtf(R), 2.f), 0.05f * rnd01());
    }
}

// blast: trees within ~2 crater radii go over if the shock carries enough energy to them
void blast(V3 at, float E, float R) {
    if (E <= BLAST_ENERGY) return;
    std::vector<TreeInst> hit;
    for (const TreeInst& t : treeList) {
        float dx = t.p.x - at.x, dz = t.p.z - at.z;
        if (dx * dx + dz * dz < 4.f * R * R && (int)hit.size() < MAX_BLAST_TREES) hit.push_back(t);
    }
    for (const TreeInst& t : hit) {
        V3 d = t.p - at;
        d.y = 0;
        float dist = std::max(sqrtf(dot(d, d)), 0.5f);
        V3 dir = d * (1.f / dist);
        float J = std::min(sqrtf(E) * 8.f / dist, 4e4f);
        // the blast snaps trunks low down; closer in it shreds them
        V3 at2 = t.p + V3{0, (0.5f + 0.9f * rnd01()) * t.scale, 0}, Jv = dir * J + V3{0, J * 0.15f, 0};
        if (t.v < (uint32_t)trees::VARIANTS) strikeTree(t, at2, Jv * (1.f / 60.f), 60.f);  // the shock front as a 60 kg body
        else knockTree(t, Jv, at2, 1.f + J / 6e3f);
    }
}
}  // namespace

void settleCraters() {
    int n = std::min(gCraterCount.load(std::memory_order_acquire), MAX_CRATERS);
    const auto& zs = stream::zones();
    for (int i = 0; i < n; i++)
        if (std::isnan(gCraters[i].base))
            settle(gCraters[i], [&](float x, float z) { return craterCarveN(x, z, terrainBase(x, z, zs.data(), (int)zs.size()), i); });
}

void groundImpact(V3 at, float E, bool primary) {
    float V = E / SOIL_STRENGTH, R = radiusFor(V);
    if (R < MIN_RADIUS) return;
    int n = std::min(gCraterCount.load(std::memory_order_relaxed), MAX_CRATERS);
    int into = bowlAt(at.x, at.z, 1.f);
    // debris falling back into a bowl, or a slow fragment anywhere: fallback, not a new impact
    if (!primary && (into >= 0 || E < FRAGMENT_MIN_ENERGY)) return;
    if (into >= 0) {
        Crater& o = gCraters[into];
        float dx = at.x - o.x, dz = at.z - o.z;
        if (R >= MERGE_SIZE * o.r && dx * dx + dz * dz < MERGE_REACH * MERGE_REACH * o.r * o.r) {
            // a comparable blow near the centre: the bowl grows (volumes add) and shifts toward the hit
            float Vo = BOWL_VOLUME * o.r * o.r * o.r, Vt = Vo + V, w = V / Vt;
            float Rn = radiusFor(Vt);
            o.x += dx * w; o.z += dz * w;
            o.r = Rn;
            o.depth = CRATER_DEPTH_RATIO * Rn;
            terrainDirty = placeDirty = true;
            stream::terrainChanged();
            ejecta(at, R);
            blast(at, E, Rn);
            return;
        }
        // smaller: its own crater, dug into the floor (settle() takes the floor as its ground)
    }
    Crater c{at.x, at.z, R, CRATER_DEPTH_RATIO * R};
    settle(c, heightFn);
    if (n < MAX_CRATERS) {
        gCraters[n] = c;
        gCraterCount.store(n + 1, std::memory_order_release);
    } else {  // full: the smallest crater goes (the least visible loss)
        int k = 0;
        for (int i = 1; i < n; i++) if (gCraters[i].r < gCraters[k].r) k = i;
        if (gCraters[k].r >= R) return;
        gCraters[k] = c;
    }
    terrainDirty = placeDirty = true;
    stream::terrainChanged();
    ejecta(at, R);
    blast(at, E, R);
    if (E > FIRE_ENERGY) startFire(V3{c.x, heightFn(c.x, c.z), c.z}, std::min(FIRE_SIZE * R, 4.f), FIRE_BURN);
}

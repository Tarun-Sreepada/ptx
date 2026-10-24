#include "sim/water.h"
#include <algorithm>
#include <cmath>
#include "world/terrain.h"
#include "app/settings.h"
#include "world/terrain_query.h"
#include "world/world.h"

// Rain swells the river: its surface rises toward ~0.35 m x precipitation over a minute or so, and falls back more
// slowly once it stops (the catchment keeps draining). Raises the water, not the bed; the current speeds up with it.
float riverSwell = 0.f;

CraterWater craterWater[MAX_CRATERS];

void updateRiverSwell(float dt) {
    float target = S.riverRainRise * S.rain;
    float tau = target > riverSwell ? 60.f : 180.f;
    riverSwell += (target - riverSwell) * (1.f - expf(-dt * S.simSpeed / tau));
}

// CPU mirror of material.glsl waterLevelAt: inside a crater only its pond counts; -1e9 = dry
float waterLevelAt(float x, float z) {
    int n = std::min(gCraterCount.load(std::memory_order_relaxed), MAX_CRATERS);
    for (int i = 0; i < n; i++) {
        float dx = x - gCraters[i].x, dz = z - gCraters[i].z, r = gCraters[i].r * 1.2f;
        if (dx * dx + dz * dz < r * r) return craterWater[i].pond ? craterWater[i].level : -1e9f;
    }
    if (fabsf(z - riverCenter(x)) > 3.f * gRiver.width + 1.f) return -1e9f;
    RiverInfo ri = riverInfo(x, z);
    return fabsf(ri.d) < ri.w + 1.f + 2.f * riverSwell ? ri.level + riverSwell : -1e9f;
}

void updateCraterWater(float dt) {
    int n = std::min(gCraterCount.load(std::memory_order_relaxed), MAX_CRATERS);
    float U = 1.3f * S.flow, Qr = std::max(U * gRiver.width * gRiver.depth, 0.1f);  // river discharge, m^3/s
    for (int i = 0; i < n; i++) {
        const Crater& c = gCraters[i];
        CraterWater& w = craterWater[i];
        if (w.lastR != c.r) {  // new or grown: the old water spreads over the bigger bowl
            w.fill = w.lastR > 0.f ? w.fill * powf(w.lastR / c.r, 3.f) : 0.f;
            w.lastR = c.r;
            // the ground before craters under it, and its slope: the plane the shaders build the local surface on
            const auto& zs = stream::zones();
            w.ground[0] = terrainBase(c.x, c.z, zs.data(), (int)zs.size());
            w.ground[1] = c.gx;
            w.ground[2] = c.gz;
        }
        RiverInfo ri = riverInfo(c.x, c.z);
        float bottom = heightFn(c.x, c.z);
        bool touches = fabsf(ri.d) - ri.w < c.r * 1.1f;  // the bowl breaks into the channel
        w.pond = touches && bottom < ri.level - 0.02f;
        if (!w.pond) { w.divert = w.inflowSpeed = 0.f; w.fill = 1.f; continue; }
        float depth = ri.level - bottom, V = 0.5f * 3.14159f * c.r * c.r * depth;
        float head = std::max(1.f - w.fill, 0.f);
        float Q = std::min(S.craterInflow * Qr * sqrtf(head), Qr) * (1.f + riverSwell / std::max(gRiver.depth, 0.3f));
        w.fill = std::min(w.fill + Q * dt / std::max(V, 0.01f), 1.f);
        w.level = bottom + (ri.level + riverSwell - bottom) * w.fill;
        w.divert = w.fill < 1.f ? Q / Qr : 0.f;
        w.inflowSpeed = w.fill < 1.f ? std::min(sqrtf(2.f * 9.81f * depth * head) * 0.6f, 4.f) : 0.f;
    }
}

// The terrain height function (mirrors world/terrain.h): natural noise + world map edits, the river channel, flatten
// zones, then craters (they dig into anything). Snow lies on top of the result.
#ifndef TERRAIN_HEIGHT
#define TERRAIN_HEIGHT
#include "core/frame.glsl"
#include "terrain/noise.glsl"
#include "terrain/zones.glsl"
#include "terrain/river.glsl"
#include "terrain/craters.glsl"
#include "terrain/worldmap.glsl"

// hills and mountains toward the horizon (terrain.h farRelief)
const float RELIEF_START = 3000.0, RELIEF_FULL = 5500.0, RELIEF_HEIGHT = 420.0;
const float RELIEF_VALLEY = 60.0, RELIEF_VALLEY_WIDE = 700.0, RELIEF_SCALE = 1800.0;
float farRelief(vec2 p) {
    float r2 = dot(p, p);
    if (r2 <= RELIEF_START * RELIEF_START) return 0.0;
    float k = smoothstep(RELIEF_START, RELIEF_FULL, sqrt(r2)) * smoothstep(RELIEF_VALLEY, RELIEF_VALLEY_WIDE, abs(p.y - riverCenter(p.x)));
    if (k <= 0.0) return 0.0;
    float h = 0.0, a = 0.5, f = 1.0 / RELIEF_SCALE;
    for (int o = 0; o < 5; o++) {
        float n = 1.0 - abs(2.0 * tnoise(p.x * f + 31.7 * float(o), p.y * f - 17.3 * float(o)) - 1.0);
        h += a * n * n;
        a *= 0.5;
        f *= 2.1;
    }
    return k * RELIEF_HEIGHT * h;
}

// the ground before any crater (terrain.h terrainBase)
float groundBase(vec2 p) {
    float h = 2.2 * tnoise(p.x * 0.022 + 10.0, p.y * 0.022) + 0.7 * tnoise(p.x * 0.07, p.y * 0.07 + 5.0) +
              0.12 * tnoise(p.x * 0.35, p.y * 0.35) - 1.5 + farRelief(p) + editHeight(p);
    h = riverCarve(p, h);
    for (uint i = 0u; i < F.counts.x; i++) h = mix(h, zones[2u * i + 1u].x, zoneWeight(i, p));
    return h;
}
// the ground without snow (the simulation's floor)
float groundHeight(vec2 p) { return craterApply(groundBase(p), craterSumAt(p)); }
float terrainHeight(vec2 p) { return groundHeight(p) + snowDepth(p); }  // the visible surface: snow lies on top
#endif

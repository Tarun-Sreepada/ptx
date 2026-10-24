// Water surface height (river, crater ponds, drawn rivers), shared by shading, underwater light and the CPU mirror.
#ifndef WATER_LEVEL
#define WATER_LEVEL
#include "core/frame.glsl"
#include "terrain/height.glsl"

// ---- water surface (river + crater ponds), shared by the water shading, underwater light and the CPU mirror
// Crater ponds (F.craterW, main.cpp updateCraterWater): the crater whose bowl holds p (any crater, -1 = none)
int craterAt(vec2 p) {
    uint cs, cn;
    bool bn = craterRange(p, cs, cn);
    for (uint j = 0u; j < cn; j++) { uint i = craterIdx(bn, cs, j); if (length(p - F.craters[i].xy) < F.craters[i].z * 1.2) return int(i); }
    return -1;
}
int craterPondAt(vec2 p) {
    int i = craterAt(p);
    return i >= 0 && F.craterW[i].x > -1e8 ? i : -1;
}
// Water diverted into filling craters is missing downstream: flow factor at x, and the level drop there
float riverDiversion(float x, out float drop) {
    float s = 0.0;
    drop = 0.0;
    for (uint i = 0u; i < F.dynInfo.z; i++) {
        float q = F.craterW[i].z;
        if (q <= 0.0 || x < F.craters[i].x) continue;
        float k = q * exp(-(x - F.craters[i].x) / 150.0);
        s += k;
        drop += k * 0.3 * F.river.y;
    }
    return clamp(1.0 - s, 0.15, 1.0);
}
// water surface height over p, -1e9 = dry. Inside a crater only its pond counts (a crater by the river is not
// under the river's level until it has filled).
float waterLevelAt(vec2 p) {
    if (F.dynInfo.z > 0u) {
        int c = craterAt(p);
        if (c >= 0) return F.craterW[c].x;
    }
    float drawn = mapWaterAt(p).x;  // a river drawn on the world map
    if (drawn > -999.0) return drawn;
    if (F.river.x <= 0.0 || !riverNear(p, 1.0)) return -1e9;
    vec4 r = riverInfo(p);
    if (abs(r.x) > r.y + 1.0) return -1e9;
    float drop = 0.0;
    if (F.dynInfo.z > 0u) riverDiversion(p.x, drop);
    return r.w - drop + F.sim2.w;
}
#endif

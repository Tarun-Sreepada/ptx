// Impact craters (mirrors world/terrain.h): an absolute paraboloid bowl per crater on the ground plane it was dug
// into, a raised rim and an ejecta blanket thinning as (r / R)^-3. surface = smin(ground + sum of ejecta, lowest bowl):
// overlapping bowls union smoothly, a later crater digs into an older one's floor, the rim is a rounded crest.
// F.craters[i] = (x, z, radius, depth); F.craterG[i] = (ground before craters at the centre, slope x, slope z,
// height of the ground the impact dug into).
#ifndef TERRAIN_CRATERS
#define TERRAIN_CRATERS
#include "core/frame.glsl"
#include "terrain/noise.glsl"

const float CRATER_RIM_RATIO = 0.2;      // rim height / bowl depth (world/terrain.h)
const float CRATER_EJECTA_REACH = 2.5;   // ejecta blanket radius, in crater radii
const float CRATER_SOFT = 0.35;          // rim rounding / rim height
const float CRATER_SOFT_MIN = 0.02;      // m

struct CraterSum { float ejecta, bowl, soft; };
CraterSum craterSumInit() { return CraterSum(0.0, 1e30, CRATER_SOFT_MIN); }
float craterSmin(float a, float b, float k) { float h = max(k - abs(a - b), 0.0) / k; return min(a, b) - h * h * k * 0.25; }

void craterAcc(vec4 c, vec4 g, vec2 p, inout CraterSum s) {
    vec2 d = p - c.xy;
    float t2 = dot(d, d) / (c.z * c.z), reach2 = CRATER_EJECTA_REACH * CRATER_EJECTA_REACH;
    if (t2 > reach2 || isnan(g.w)) return;  // (not settled yet: no shape)
    float rim = CRATER_RIM_RATIO * c.w;
    float tail = 1.0 / (reach2 * CRATER_EJECTA_REACH);
    float e = t2 <= 1.0 ? 1.0 : (1.0 / (t2 * sqrt(t2)) - tail) / (1.0 - tail);
    s.ejecta += rim * e;
    float b = g.w + dot(g.yz, d) - c.w + (c.w + rim) * t2;
    if (b < s.bowl) { s.bowl = b; s.soft = max(CRATER_SOFT * rim, CRATER_SOFT_MIN); }
}
float craterApply(float h, CraterSum s) { return s.bowl >= 1e29 ? h + s.ejecta : craterSmin(h + s.ejecta, s.bowl, s.soft); }

// Crater bins (render/crater_bins.cpp): a 32 x 32 grid of 16 m cells around the camera, each listing the craters whose
// reach (ejecta reach x r + 1 m) touches it. Lookups near the camera loop over those few instead of every crater.
layout(binding = 97, std430) readonly buffer CraterBins { uint craterBins[]; };  // [2b] start, [2b+1] count; list after
const uint CB_BINS = 1024u;
const float CB_CELL = 16.0;
bool craterRange(vec2 p, out uint s, out uint n) {
    ivec2 b = ivec2(floor(p / CB_CELL)) - F.craterGrid.xy;
    if (F.craterGrid.w > 0 && b.x >= 0 && b.y >= 0 && b.x < F.craterGrid.z && b.y < F.craterGrid.z) {
        uint k = uint(b.y * F.craterGrid.z + b.x);
        s = 2u * CB_BINS + craterBins[2u * k];
        n = craterBins[2u * k + 1u];
        return true;
    }
    s = 0u;
    n = F.dynInfo.z;
    return false;
}
uint craterIdx(bool binned, uint s, uint j) { return binned ? craterBins[s + j] : j; }

CraterSum craterSumAt(vec2 p) {
    CraterSum s = craterSumInit();
    if (F.dynInfo.z == 0u) return s;
    uint cs, cn;
    bool bn = craterRange(p, cs, cn);
    for (uint j = 0u; j < cn; j++) { uint i = craterIdx(bn, cs, j); craterAcc(F.craters[i], F.craterG[i], p, s); }
    return s;
}
// How much of the ground at p is crater (1 in a bowl, fading over the rim) or buried under ejecta: no grass there.
float craterDisturbance(vec2 p) {
    if (F.dynInfo.z == 0u) return 0.0;
    float k = 0.0, ej = 0.0;
    uint cs, cn;
    bool bn = craterRange(p, cs, cn);
    for (uint j = 0u; j < cn; j++) {
        uint i = craterIdx(bn, cs, j);
        float t = length(p - F.craters[i].xy) / F.craters[i].z;
        k = max(k, 1.0 - smoothstep(0.95, 1.35, t));
        if (t > 1.0 && t < CRATER_EJECTA_REACH) ej += CRATER_RIM_RATIO * F.craters[i].w / (t * t * t);
    }
    return max(k, smoothstep(0.01, 0.08, ej));
}
float craterCover(vec2 p) {  // 1 inside a crater, fading out over the rim
    if (F.dynInfo.z == 0u) return 0.0;
    float k = 0.0;
    uint cs, cn;
    bool bn = craterRange(p, cs, cn);
    for (uint j = 0u; j < cn; j++) { uint i = craterIdx(bn, cs, j); k = max(k, 1.0 - smoothstep(0.95, 1.35, length(p - F.craters[i].xy) / F.craters[i].z)); }
    return k;
}
#endif

// The river channel and its rocks: mirrors world/terrain.h.
#ifndef TERRAIN_RIVER
#define TERRAIN_RIVER
#include "core/frame.glsl"
#include "terrain/noise.glsl"

// River channel, mirrors terrain.h. x = signed distance across the flow, y = half-width, z = depth, w = water level.
const float RIVER_Z = -58.0, RIVER_BANK = 16.0;
float riverCenter(float x) { return RIVER_Z + 14.0 * sin(0.011 * x) + 7.0 * sin(0.027 * x + 1.3); }
float riverSlope(float x) { return 0.154 * cos(0.011 * x) + 0.189 * cos(0.027 * x + 1.3); }
bool riverNear(vec2 p, float margin) { return abs(p.y - riverCenter(p.x)) <= 3.0 * F.river.x + margin; }
vec4 riverInfo(vec2 p) {
    float dz = riverSlope(p.x);
    return vec4((p.y - riverCenter(p.x)) * inversesqrt(1.0 + dz * dz), F.river.x * (0.8 + 0.45 * tnoise(p.x * 0.008 + 3.0, 7.5)),
                F.river.y * (0.45 + 1.0 * tnoise(p.x * 0.013, 2.5)), F.river.z - F.river.w * p.x);
}
float riverCarve(vec2 p, float h) {
    if (!riverNear(p, RIVER_BANK + 6.0)) return h;
    vec4 r = riverInfo(p);
    float d = abs(r.x), c = d / r.y;
    float prof = d < r.y ? r.w - r.z * (1.0 - c * c) : r.w + 0.35 * (d - r.y);
    if (F.water2.w > 0.0 && d < r.y + 2.0) {  // uneven bed (terrain.h riverCarve, gRiver.bumps)
        float n = tnoise(p.x * 0.45, p.y * 0.45) * 0.6 + tnoise(p.x * 1.3 + 7.0, p.y * 1.3) * 0.4 - 0.5;
        prof += 2.0 * F.water2.w * n * sqrt(max(1.0 - c * c, 0.0));
    }
    return mix(prof, h, smoothstep(r.y + 2.0, r.y + RIVER_BANK, d));
}
const float RIVER_ROCK_L = 2.5;
bool riverRock(int i, out vec4 rk) {  // xyz centre, w scale (terrain.h)
    if (thash(i * 7 + 3, 811) > 0.65) return false;
    bool boulder = thash(i * 5 + 1, 823) < 0.35;
    float x = (float(i) + thash(i, 853)) * RIVER_ROCK_L;
    float hc = thash(i * 3 + 2, 839) * 2.0 - 1.0;
    float c = boulder ? hc * 0.75 : (hc < 0.0 ? -1.0 : 1.0) * (0.45 + 0.5 * abs(hc)), dz = riverSlope(x);
    vec4 r = riverInfo(vec2(x, 0.0));
    float z = riverCenter(x) + c * r.y * sqrt(1.0 + dz * dz);
    float bed = r.w - r.z * (1.0 - c * c);
    float sc = boulder ? min(max(0.9 + 0.8 * thash(i, 857), (r.w - bed) * 0.9), 2.6) : 0.25 + 0.45 * thash(i, 859);
    float y = boulder ? max(bed - 0.3 * sc, r.w - 0.4 * sc) : bed - 0.3 * sc;
    float dc = c * r.y, f = 0.8 * sc, nearest = (dc - f <= 0.0 && dc + f >= 0.0) ? 0.0 : min(abs(dc - f), abs(dc + f));
    float cn = nearest / r.y;
    float minBed = nearest < r.y ? r.w - r.z * (1.0 - cn * cn) : r.w + 0.35 * (nearest - r.y);
    rk = vec4(x, min(y, minBed + 0.4 * sc), z, sc);
    return true;
}
#endif

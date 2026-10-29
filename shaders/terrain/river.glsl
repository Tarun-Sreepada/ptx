// The river channel and its rocks: mirrors world/terrain.h.
#ifndef TERRAIN_RIVER
#define TERRAIN_RIVER
#include "core/frame.glsl"
#include "terrain/noise.glsl"

// River channels, mirrors terrain.h: the main river, braids around islands, tributaries curving in.
// riverInfo: x = signed distance across the flow, y = half-width, z = depth, w = water level (nearest channel).
const float RIVER_Z = -58.0, RIVER_BANK = 16.0, BRAID_CELL = 420.0, TRIB_CELL = 650.0, RIVER_MIN_W = 1.5;
float riverCenter(float x) { vec4 r = F.riverShape; vec2 p = F.riverShape2.xy; return RIVER_Z + r.x * sin(r.z * x + p.x) + r.y * sin(r.w * x + p.y); }
float riverSlope(float x) { vec4 r = F.riverShape; vec2 p = F.riverShape2.xy; return r.x * r.z * cos(r.z * x + p.x) + r.y * r.w * cos(r.w * x + p.y); }
float riverWidthF(float x) {  // terrain.h riverWidthF
    float reach = tnoise(x * 0.0025 + 11.0, 4.5), local = tnoise(x * 0.008 + 3.0, 7.5);
    return clamp(1.0 + F.river2.x * ((reach - 0.5) * 2.2 + (local - 0.5) * 0.9), 0.35, 2.4);
}
float riverHash(int i, int k) { return thash(i * 92821 + k * 131, 7177 + k); }
// channels at x (terrain.h riverChannels): each = centre z, dz/dx, width multiplier, level offset
int riverChannels(float x, out vec4 ch[4]) {
    const float PI_ = 3.14159265;
    float c0 = riverCenter(x), s0 = riverSlope(x), mainW = 1.0;
    int n = 1;
    ch[1] = ch[2] = ch[3] = vec4(0);
    if (F.river2.y > 0.0) {
        int j = int(floor(x / BRAID_CELL));
        if (riverHash(j, 1) < F.river2.y) {
            float st = float(j) * BRAID_CELL + 40.0 + 60.0 * riverHash(j, 2), L = 180.0 + 140.0 * riverHash(j, 3), u = (x - st) / L;
            if (u > 0.0 && u < 1.0) {
                float side = riverHash(j, 4) < 0.5 ? -1.0 : 1.0, sep = F.river.x * (3.5 + 3.0 * riverHash(j, 5)), sn = sin(PI_ * u);
                ch[n++] = vec4(c0 + side * sep * sn * sn, s0 + side * sep * PI_ / L * sin(2.0 * PI_ * u), 1.0 - 0.3 * sn * sn, 0.0);
                mainW = 1.0 - 0.25 * sn * sn;
            }
        }
    }
    ch[0] = vec4(c0, s0, mainW, 0.0);
    if (F.river2.z > 0.0) {
        int k0 = int(floor(x / TRIB_CELL));
        for (int k = k0; k <= k0 + 1 && n < 4; k++) {
            if (riverHash(k, 11) >= F.river2.z) continue;
            float xj = float(k) * TRIB_CELL + TRIB_CELL * (0.55 + 0.35 * riverHash(k, 12)), Lt = 260.0 + 200.0 * riverHash(k, 13), u = (xj - x) / Lt;
            if (u <= 0.0 || u >= 1.0) continue;
            float side = riverHash(k, 14) < 0.5 ? -1.0 : 1.0, D = 120.0 + 120.0 * riverHash(k, 15);
            float wm = 0.45 * (1.0 - smoothstep(0.8, 1.0, u));
            if (wm < 0.03) continue;
            ch[n++] = vec4(c0 + side * D * pow(u, 1.6), s0 - side * D * 1.6 * pow(u, 0.6) / Lt, wm, 0.004 * (xj - x));
        }
    }
    // too small: taper away to a dry stretch (terrain.h); dropped side channels, the main one kept at ~0 width
    float wf = riverWidthF(x);
    int m = 0;
    for (int i = 0; i < n; i++) {
        vec4 c = ch[i];
        c.z *= smoothstep(RIVER_MIN_W, 2.0 * RIVER_MIN_W, F.river.x * wf * c.z);
        if (i == 0 || c.z > 1e-3) ch[m++] = c;
    }
    return m;
}
bool riverWet(vec4 c) { return c.z > 1e-3; }
vec4 riverInfoC(vec2 p, vec4 c) {
    float wf = riverWidthF(p.x);
    return vec4((p.y - c.x) * inversesqrt(1.0 + c.y * c.y), max(F.river.x * wf * c.z, 1e-3),
                F.river.y * (0.45 + 1.0 * tnoise(p.x * 0.013, 2.5)) * inversesqrt(wf) * (0.5 + 0.5 * c.z), F.river.z - F.river.w * p.x + c.w);
}
// the channel whose water is nearest; dz = its direction (dz/dx)
vec4 riverInfoS(vec2 p, out float dz) {
    vec4 ch[4];
    int n = riverChannels(p.x, ch);
    vec4 best = riverInfoC(p, ch[0]);
    dz = ch[0].y;
    for (int i = 1; i < n; i++) {
        vec4 r = riverInfoC(p, ch[i]);
        if (abs(r.x) - r.y < abs(best.x) - best.y) { best = r; dz = ch[i].y; }
    }
    return best;
}
vec4 riverInfo(vec2 p) { float dz; return riverInfoS(p, dz); }
bool riverNear(vec2 p, float margin) {
    vec4 ch[4];
    int n = riverChannels(p.x, ch);
    for (int i = 0; i < n; i++) if (riverWet(ch[i]) && abs(p.y - ch[i].x) <= 3.0 * F.river.x * ch[i].z + margin) return true;
    return false;
}
float riverCarveC(vec2 p, float h, vec4 r) {
    float d = abs(r.x), c = d / r.y;
    if (d > r.y + RIVER_BANK) return h;
    float prof = d < r.y ? r.w - r.z * (1.0 - c * c) : r.w + 0.35 * (d - r.y);
    if (F.water2.w > 0.0 && d < r.y + 2.0) {  // uneven bed (terrain.h riverCarve, gRiver.bumps)
        float n = tnoise(p.x * 0.45, p.y * 0.45) * 0.6 + tnoise(p.x * 1.3 + 7.0, p.y * 1.3) * 0.4 - 0.5;
        prof += 2.0 * F.water2.w * n * sqrt(max(1.0 - c * c, 0.0));
    }
    return mix(prof, h, smoothstep(r.y + 2.0, r.y + RIVER_BANK, d));
}
float riverCarve(vec2 p, float h) {
    vec4 ch[4];
    int n = riverChannels(p.x, ch);
    float o = h;
    for (int i = 0; i < n; i++) {
        if (!riverWet(ch[i]) || abs(p.y - ch[i].x) > 3.0 * F.river.x * ch[i].z + RIVER_BANK + 6.0) continue;
        o = min(o, riverCarveC(p, h, riverInfoC(p, ch[i])));
    }
    return o;
}
// rolling hills from the seed, rising away from the river (terrain.h hills)
const float HILL_HEIGHT = 45.0;
float hills(vec2 p) {
    if (F.river2.w <= 0.0) return 0.0;
    float n = tnoise(p.x * 0.0035 + 101.0, p.y * 0.0035 - 57.0) * 0.6 + tnoise(p.x * 0.009 - 13.0, p.y * 0.009 + 29.0) * 0.3 +
              tnoise(p.x * 0.024 + 7.0, p.y * 0.024 + 3.0) * 0.1;
    vec4 r = riverInfo(p);
    float t = clamp((abs(r.x) - r.y - 6.0) / 60.0, 0.0, 1.0);
    return F.river2.w * HILL_HEIGHT * n * n * t * t * (3.0 - 2.0 * t);
}
const float RIVER_ROCK_L = 2.5;
const int RIVER_ROCK_CHANNELS = 4;
// rock i of channel k (terrain.h riverRock): xyz centre, w scale; dz = its channel's direction
bool riverRock(int i, int k, out vec4 rk, out float dz) {
    int hi = i + k * 1000003;
    float x = (float(i) + thash(hi, 853)) * RIVER_ROCK_L;
    vec4 ch[4];
    if (k >= riverChannels(x, ch) || !riverWet(ch[k])) return false;
    if (thash(hi * 7 + 3, 811) > min(0.65 * riverWidthF(x) * ch[k].z, 0.97)) return false;  // more where it's wide
    bool boulder = thash(hi * 5 + 1, 823) < 0.35;
    float hc = thash(hi * 3 + 2, 839) * 2.0 - 1.0;
    dz = ch[k].y;
    float c = boulder ? hc * 0.75 : (hc < 0.0 ? -1.0 : 1.0) * (0.45 + 0.5 * abs(hc));
    vec4 r = riverInfoC(vec2(x, 0.0), ch[k]);
    float z = ch[k].x + c * r.y * sqrt(1.0 + dz * dz);
    float bed = r.w - r.z * (1.0 - c * c);
    float sc = boulder ? min(max(0.9 + 0.8 * thash(hi, 857), (r.w - bed) * 0.9), 2.6) : 0.25 + 0.45 * thash(hi, 859);
    float y = boulder ? max(bed - 0.3 * sc, r.w - 0.4 * sc) : bed - 0.3 * sc;
    float dc = c * r.y, f = 0.8 * sc, nearest = (dc - f <= 0.0 && dc + f >= 0.0) ? 0.0 : min(abs(dc - f), abs(dc + f));
    float cn = nearest / r.y;
    float minBed = nearest < r.y ? r.w - r.z * (1.0 - cn * cn) : r.w + 0.35 * (nearest - r.y);
    rk = vec4(x, min(y, minBed + 0.4 * sc), z, sc);
    return true;
}
#endif

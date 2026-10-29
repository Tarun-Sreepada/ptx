// Terrain height shared by the renderer (light/rock placement), the physics heightfields and the streamer.
// Mirrors shaders/terrain/*.glsl: value noise + the river channel + flatten zones (city plots, roads) + impact craters.
// Order: natural ground + world map edits -> river carve -> flatten zones -> craters (they dig into anything).
#pragma once
#include <immintrin.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <string>

struct Zone {
    float minx, minz, maxx, maxz, height, falloff;
    std::string mat;    // ground texture on the flattened area (matlib name), empty = plain paving colour
    float tile = 2.f;   // metres per texture repeat
};

// The world seed: mixed into the base hash, so the ground, hills, rivers, trees, plants, rocks and dirt all follow it
// (seed 0 = mix 0 = the original world). Change it through setWorldSeed (terrain_query / settings), never directly.
struct WorldGen { uint32_t seed = 0; float hilliness = 0.35f; };
inline WorldGen gWorld;
inline uint32_t gSeedMix = 0;
inline float hash2(int x, int z) {
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)z * 668265263u + gSeedMix;
    h = (h ^ (h >> 13)) * 1274126177u;
    return ((h ^ (h >> 16)) & 0xFFFFFF) / float(0xFFFFFF);
}
inline float vnoise(float x, float z) {
    int ix = (int)floorf(x), iz = (int)floorf(z);
    float fx = x - ix, fz = z - iz;
    fx = fx * fx * (3 - 2 * fx);
    fz = fz * fz * (3 - 2 * fz);
    float a = hash2(ix, iz), b = hash2(ix + 1, iz), c = hash2(ix, iz + 1), d = hash2(ix + 1, iz + 1);
    return (a + (b - a) * fx) + ((c + (d - c) * fx) - (a + (b - a) * fx)) * fz;
}
// World map edits (worldmap.cpp): metres added to the natural ground (brushes, carved rivers)
float editHeight(float x, float z);
inline float naturalHeight(float x, float z) {
    return 2.2f * vnoise(x * 0.022f + 10, z * 0.022f) + 0.7f * vnoise(x * 0.07f, z * 0.07f + 5) + 0.12f * vnoise(x * 0.35f, z * 0.35f) - 1.5f;
}
// River: a meandering channel running along +x (downstream), carved into the natural terrain before the flatten
// zones. Mirrored in shaders/terrain/river.glsl, parameters in the UBO's river vectors. Along it the seed places braids
// (the river splits around an island and joins again) and tributaries (side streams that curve in and converge).
struct RiverParams {
    float width = 6.f, depth = 1.8f, level = -1.4f, slope = 0.0003f, bumps = 0.f;  // half-width, depth (m), bed roughness (m)
    float widthVar = 0.6f;                   // how much the width wanders along the river
    float braids = 0.45f, tributaries = 0.5f;  // share of the stretches that split / that a side stream joins
};
inline RiverParams gRiver;
// the main channel's meander, from the seed (setWorldSeed); seed 0 keeps the original curve
struct RiverShape { float a1 = 14.f, a2 = 7.f, f1 = 0.011f, f2 = 0.027f, p1 = 0.f, p2 = 1.3f; };
inline RiverShape gRiverShape;
inline constexpr float RIVER_Z = -58.f, RIVER_BANK = 16.f;  // centreline offset, bank blend width past the water edge
inline constexpr float BRAID_CELL = 420.f, TRIB_CELL = 650.f;  // m of river per braid / tributary candidate
inline constexpr float RIVER_MIN_W = 1.5f;  // m half-width: below this a channel dries up (full width at twice this)
inline float riverCenter(float x) { const RiverShape& r = gRiverShape; return RIVER_Z + r.a1 * sinf(r.f1 * x + r.p1) + r.a2 * sinf(r.f2 * x + r.p2); }
inline float riverSlope(float x) { const RiverShape& r = gRiverShape; return r.a1 * r.f1 * cosf(r.f1 * x + r.p1) + r.a2 * r.f2 * cosf(r.f2 * x + r.p2); }
// Width along the river (x half-width): long reaches where it narrows into a run or spreads over a flat, plus the
// shorter wobble; a narrow stretch is deeper (about the same water gets through), a wide one shallower.
inline float riverWidthF(float x) {
    float reach = vnoise(x * 0.0025f + 11.f, 4.5f), local = vnoise(x * 0.008f + 3.f, 7.5f);
    return std::clamp(1.f + gRiver.widthVar * ((reach - 0.5f) * 2.2f + (local - 0.5f) * 0.9f), 0.35f, 2.4f);
}
// One channel at a given x: centre z, its dz/dx, width multiplier, water level above the main river's (m)
struct RiverChan { float c, dz, wm, dl; };
inline float riverHash(int i, int k) { return hash2(i * 92821 + k * 131, 7177 + k); }
// Every channel present at x (main first): the main river, a braid around an island, tributaries curving in.
inline int riverChannels(float x, RiverChan out[4]) {
    constexpr float PI_ = 3.14159265f;
    float c0 = riverCenter(x), s0 = riverSlope(x), mainW = 1.f;
    int n = 1;
    if (gRiver.braids > 0.f) {  // split -> island -> rejoin, the two arms sharing the water
        int j = (int)floorf(x / BRAID_CELL);
        if (riverHash(j, 1) < gRiver.braids) {
            float st = j * BRAID_CELL + 40.f + 60.f * riverHash(j, 2), L = 180.f + 140.f * riverHash(j, 3), u = (x - st) / L;
            if (u > 0.f && u < 1.f) {
                float side = riverHash(j, 4) < 0.5f ? -1.f : 1.f, sep = gRiver.width * (3.5f + 3.f * riverHash(j, 5)), sn = sinf(PI_ * u);
                out[n++] = {c0 + side * sep * sn * sn, s0 + side * sep * PI_ / L * sinf(2.f * PI_ * u), 1.f - 0.3f * sn * sn, 0.f};
                mainW = 1.f - 0.25f * sn * sn;
            }
        }
    }
    out[0] = {c0, s0, mainW, 0.f};
    if (gRiver.tributaries > 0.f) {  // a side stream from upstream, curving in to join at xj (this cell's or the next)
        int k0 = (int)floorf(x / TRIB_CELL);
        for (int k = k0; k <= k0 + 1 && n < 4; k++) {
            if (riverHash(k, 11) >= gRiver.tributaries) continue;
            float xj = k * TRIB_CELL + TRIB_CELL * (0.55f + 0.35f * riverHash(k, 12)), Lt = 260.f + 200.f * riverHash(k, 13), u = (xj - x) / Lt;
            if (u <= 0.f || u >= 1.f) continue;
            float side = riverHash(k, 14) < 0.5f ? -1.f : 1.f, D = 120.f + 120.f * riverHash(k, 15);
            float t = std::clamp((u - 0.8f) / 0.2f, 0.f, 1.f), wm = 0.45f * (1.f - t * t * (3.f - 2.f * t));
            if (wm < 0.03f) continue;
            out[n++] = {c0 + side * D * powf(u, 1.6f), s0 - side * D * 1.6f * powf(u, 0.6f) / Lt, wm, 0.004f * (xj - x)};
        }
    }
    // too small to be a river: narrower than RIVER_MIN_W it tapers away to nothing (a dry stretch) until the water
    // is wide enough again; a side channel that vanishes is dropped, the main one stays as a zero-width marker
    float wf = riverWidthF(x);
    int m = 0;
    for (int i = 0; i < n; i++) {
        float w = gRiver.width * wf * out[i].wm, t = std::clamp((w - RIVER_MIN_W) / RIVER_MIN_W, 0.f, 1.f);
        out[i].wm *= t * t * (3.f - 2.f * t);
        if (i == 0 || out[i].wm > 1e-3f) out[m++] = out[i];
    }
    return m;
}
inline bool riverWet(const RiverChan& c) { return c.wm > 1e-3f; }
struct RiverInfo { float d, w, depth, level, dz; };  // signed distance across the flow (+ = left bank), half-width, depth, water level, channel dz/dx
inline RiverInfo riverInfoC(float x, float z, const RiverChan& ch) {
    float wf = riverWidthF(x);
    return {(z - ch.c) / sqrtf(1.f + ch.dz * ch.dz), std::max(gRiver.width * wf * ch.wm, 1e-3f),
            gRiver.depth * (0.45f + 1.f * vnoise(x * 0.013f, 2.5f)) / sqrtf(wf) * (0.5f + 0.5f * ch.wm), gRiver.level - gRiver.slope * x + ch.dl, ch.dz};
}
// the channel whose water is nearest (or that the point is in)
inline RiverInfo riverInfo(float x, float z) {
    RiverChan ch[4];
    int n = riverChannels(x, ch);
    RiverInfo best = riverInfoC(x, z, ch[0]);
    for (int i = 1; i < n; i++) {  // (side channels are always wet; a dry main one has ~0 width, so it loses)
        RiverInfo r = riverInfoC(x, z, ch[i]);
        if (fabsf(r.d) - r.w < fabsf(best.d) - best.w) best = r;
    }
    return best;
}
// how far outside every channel's water x, z is (m; <= 0 in the water): hills keep away from the river
inline float riverEdgeDist(float x, float z) {
    RiverInfo r = riverInfo(x, z);
    return fabsf(r.d) - r.w;
}
// parabolic bed inside the water, gentle bank rising out of it, blended back into the natural ground; the deepest
// of the channels wins where they meet
inline float riverCarveC(float x, float z, float h, const RiverInfo& r) {
    float d = fabsf(r.d), c = d / r.w;
    if (d > r.w + RIVER_BANK) return h;
    float prof = d < r.w ? r.level - r.depth * (1.f - c * c) : r.level + 0.35f * (d - r.w);
    if (gRiver.bumps > 0.f && d < r.w + 2.f) {  // uneven bed: humps, scours and stony ridges, strongest mid-channel
        float n = vnoise(x * 0.45f, z * 0.45f) * 0.6f + vnoise(x * 1.3f + 7.f, z * 1.3f) * 0.4f - 0.5f;
        prof += 2.f * gRiver.bumps * n * sqrtf(std::max(1.f - c * c, 0.f));
    }
    float t = std::clamp((d - r.w - 2.f) / (RIVER_BANK - 2.f), 0.f, 1.f);
    return prof + (h - prof) * t * t * (3 - 2 * t);
}
inline float riverCarve(float x, float z, float h) {
    RiverChan ch[4];
    int n = riverChannels(x, ch);
    float out = h;
    for (int i = 0; i < n; i++) {
        if (!riverWet(ch[i]) || fabsf(z - ch[i].c) > 3.f * gRiver.width * ch[i].wm + RIVER_BANK + 6.f) continue;  // dry / cheap reject (w <= 2.4 width, |slope| < 0.35)
        out = std::min(out, riverCarveC(x, z, h, riverInfoC(x, z, ch[i])));
    }
    return out;
}
// Rolling hills from the seed (gWorld.hilliness: 0 flat .. 1 big hills), rising away from the river so the valley
// stays where the water is. Mirrored in height.glsl.
inline constexpr float HILL_HEIGHT = 45.f;
inline float hills(float x, float z) {
    if (gWorld.hilliness <= 0.f) return 0.f;
    float n = vnoise(x * 0.0035f + 101.f, z * 0.0035f - 57.f) * 0.6f + vnoise(x * 0.009f - 13.f, z * 0.009f + 29.f) * 0.3f +
              vnoise(x * 0.024f + 7.f, z * 0.024f + 3.f) * 0.1f;
    float e = riverEdgeDist(x, z), t = std::clamp((e - 20.f) / 220.f, 0.f, 1.f);
    return gWorld.hilliness * HILL_HEIGHT * n * n * t * t * (3.f - 2.f * t);
}
// Rocks in the channel, one candidate per RIVER_ROCK_L stretch. Deterministic so the water shader (riverRock) knows where
// they are and shapes the surface around them. Rock mesh: radius ~1, y in [-0.52, 0.72] (buildRockMesh).
// Boulders are sized and lifted so they break the surface; small stones sit on the bed.
inline constexpr float RIVER_ROCK_L = 2.5f;
// k: the channel (riverChannels order: main, braid, tributaries); every channel gets its own rocks.
inline constexpr int RIVER_ROCK_CHANNELS = 4;
inline bool riverRock(int i, int k, float& x, float& y, float& z, float& sc) {
    int hi = i + k * 1000003;  // each channel its own layout
    x = (i + hash2(hi, 853)) * RIVER_ROCK_L;
    RiverChan ch[4];
    if (k >= riverChannels(x, ch) || !riverWet(ch[k])) return false;
    // more rocks where the river spreads wide, fewer where it narrows (riverWidthF, and narrower side channels)
    if (hash2(hi * 7 + 3, 811) > std::min(0.65f * riverWidthF(x) * ch[k].wm, 0.97f)) return false;
    bool boulder = hash2(hi * 5 + 1, 823) < 0.35f;
    float hc = hash2(hi * 3 + 2, 839) * 2.f - 1.f;  // small stones mostly in the shallows
    float c = boulder ? hc * 0.75f : (hc < 0 ? -1.f : 1.f) * (0.45f + 0.5f * fabsf(hc)), dz = ch[k].dz;
    RiverInfo r = riverInfoC(x, 0.f, ch[k]);
    z = ch[k].c + c * r.w * sqrtf(1.f + dz * dz);
    float bed = r.level - r.depth * (1.f - c * c);
    sc = boulder ? std::min(std::max(0.9f + 0.8f * hash2(hi, 857), (r.level - bed) * 0.9f), 2.6f) : 0.25f + 0.45f * hash2(hi, 859);
    y = boulder ? std::max(bed - 0.3f * sc, r.level - 0.4f * sc) : bed - 0.3f * sc;
    // the bed slopes across the channel: find its lowest point under the rock's footprint and keep the
    // mesh bottom (y - 0.52 sc) dug into it, so no side floats (mirrored in river.glsl)
    float dc = c * r.w, f = 0.8f * sc, nearest = (dc - f <= 0.f && dc + f >= 0.f) ? 0.f : std::min(fabsf(dc - f), fabsf(dc + f));
    float cn = nearest / r.w;
    float minBed = nearest < r.w ? r.level - r.depth * (1.f - cn * cn) : r.level + 0.35f * (nearest - r.w);
    y = std::min(y, minBed + 0.4f * sc);
    return true;
}

// Surface current (m/s, world xz) for a given flow strength; mirrors riverFlow() in water/flow.glsl.
inline void riverVelocity(float x, float z, float strength, float v[2]) {
    v[0] = v[1] = 0.f;
    RiverInfo r = riverInfo(x, z);
    float c = fabsf(r.d) / r.w;
    if (c >= 1.f) return;
    float dz = r.dz, inv = 1.f / sqrtf(1.f + dz * dz);
    float U = strength * 1.3f * (gRiver.width * gRiver.depth) / (r.w * r.depth) * powf(std::max(1.f - c * c, 0.02f), 0.35f);
    v[0] = U * inv;
    v[1] = U * dz * inv;
}

// 0 in the water, 1 on dry ground (placement)
inline float riverDry(float x, float z, float margin) {
    RiverInfo r = riverInfo(x, z);
    return fabsf(r.d) > r.w + margin ? 1.f : 0.f;
}

// 0 outside, 1 inside the rectangle, smoothstep across the falloff band
inline float zoneWeight(const Zone& zn, float x, float z) {
    float dx = std::max(std::max(zn.minx - x, x - zn.maxx), 0.f), dz = std::max(std::max(zn.minz - z, z - zn.maxz), 0.f);
    float t = std::min(sqrtf(dx * dx + dz * dz) / std::max(zn.falloff, 1e-3f), 1.f);
    return 1.f - t * t * (3 - 2 * t);
}
inline float zoneCover(float x, float z, const Zone* zones, int nz) {
    float w = 0;
    for (int i = 0; i < nz; i++) w = std::max(w, zoneWeight(zones[i], x, z));
    return w;
}
inline float applyZones(float x, float z, float h, const Zone* zones, int nz) {
    for (int i = 0; i < nz; i++) h += (zones[i].height - h) * zoneWeight(zones[i], x, z);
    return h;
}
// The ground before any crater: what an impact digs into.

#include "world/biome.h"

// Hills and mountain ranges toward the horizon: ridged noise rising past RELIEF_START from the origin (outside the
// world map's 4 km square and the world's cells, which stay as they are), held down near the river so it runs
// along a valley floor. Sharp crests, broad valleys. Mirrored in shaders/terrain/height.glsl farRelief.
inline constexpr float RELIEF_START = 3000.f, RELIEF_FULL = 5500.f, RELIEF_HEIGHT = 420.f;
inline constexpr float RELIEF_VALLEY = 60.f, RELIEF_VALLEY_WIDE = 700.f;  // m from the river: the floor, full height
inline constexpr float RELIEF_SCALE = 1800.f;                             // m: the largest ridges
inline float farRelief(float x, float z) {
    float r2 = x * x + z * z;
    if (r2 <= RELIEF_START * RELIEF_START) return 0.f;
    auto sstep = [](float a, float b, float v) { float t = std::clamp((v - a) / (b - a), 0.f, 1.f); return t * t * (3.f - 2.f * t); };
    float k = sstep(RELIEF_START, RELIEF_FULL, sqrtf(r2)) * sstep(RELIEF_VALLEY, RELIEF_VALLEY_WIDE, fabsf(z - riverCenter(x)));
    if (k <= 0.f) return 0.f;
    float h = 0.f, a = 0.5f, f = 1.f / RELIEF_SCALE;
    for (int o = 0; o < 5; o++) {
        float n = 1.f - fabsf(2.f * vnoise(x * f + 31.7f * o, z * f - 17.3f * o) - 1.f);
        h += a * n * n;
        a *= 0.5f;
        f *= 2.1f;
    }
    return k * RELIEF_HEIGHT * h;
}
inline float terrainBase(float x, float z, const Zone* zones, int nz) {
    return applyZones(x, z, lakeCarve(x, z, riverCarve(x, z, biomeShape(x, z, naturalHeight(x, z) + hills(x, z) + farRelief(x, z)) + editHeight(x, z))), zones, nz);
}

// ---------------------------------------------------------------- impact craters
// A simple crater (Pike 1977, Melosh 1989): a bowl about a third as deep as its radius, a raised rim (~0.2 x the
// depth) of uplifted and thrown-out soil, and an ejecta blanket thinning as (r / R)^-3 out to ~2.5 radii
// (McGetchin et al. 1973). Each crater keeps the ground plane it was dug into (base height + slope, at the time of the
// impact), so a later, smaller crater on the floor of an older one digs into that floor.
// The surface is smin(ground + every crater's ejecta, the lowest bowl): bowls are absolute paraboloids that rise past
// the rim and lose to the ejecta outside, overlapping bowls union smoothly, and nothing has a cliff at the rim.
// Mirrored in shaders/terrain/craters.glsl (craterAcc / craterApply).
struct Crater {
    float x, z, r, depth;
    float base = NAN, gx = 0.f, gz = 0.f;  // ground plane at the impact: height at (x, z), slope dh/dx, dh/dz (NaN: not settled yet)
};
inline constexpr int MAX_CRATERS = 256;
inline constexpr float CRATER_DEPTH_RATIO = 0.33f;   // bowl depth / radius
inline constexpr float CRATER_RIM_RATIO = 0.2f;      // rim height / bowl depth
inline constexpr float CRATER_EJECTA_REACH = 2.5f;   // ejecta blanket radius, in crater radii
inline constexpr float CRATER_SOFT = 0.35f;          // rim rounding (smooth-min width) / rim height
inline constexpr float CRATER_SOFT_MIN = 0.02f;      // m
// fixed slots + release/acquire count so a loader thread never sees a half-written crater. A new impact inside an
// existing crater grows that one (or digs a smaller one into its floor) instead of taking a slot; when all slots
// are used the smallest crater is recycled (either can briefly tear a tile being rebuilt; it's rebuilt right after)
inline Crater gCraters[MAX_CRATERS];
inline std::atomic<int> gCraterCount{0};

struct CraterSum { float ejecta = 0.f, bowl = 1e30f, soft = CRATER_SOFT_MIN; };
inline float craterSmin(float a, float b, float k) {
    float h = std::max(k - fabsf(a - b), 0.f) / k;
    return std::min(a, b) - h * h * k * 0.25f;
}
inline void craterAcc(const Crater& c, float x, float z, CraterSum& s) {
    float dx = x - c.x, dz = z - c.z, t2 = (dx * dx + dz * dz) / (c.r * c.r);
    float reach2 = CRATER_EJECTA_REACH * CRATER_EJECTA_REACH;
    if (t2 > reach2 || c.base != c.base) return;  // (not settled yet: no shape)
    float rim = CRATER_RIM_RATIO * c.depth;
    // ejecta: the rim height inside (cut by the bowl), r^-3 outside, reaching 0 at the blanket's edge
    float tail = 1.f / (reach2 * CRATER_EJECTA_REACH);
    float e = t2 <= 1.f ? 1.f : (1.f / (t2 * sqrtf(t2)) - tail) / (1.f - tail);
    s.ejecta += rim * e;
    // the bowl: an absolute paraboloid on the impact's ground plane, reaching the rim crest at t = 1
    float b = c.base + c.gx * dx + c.gz * dz - c.depth + (c.depth + rim) * t2;
    if (b < s.bowl) { s.bowl = b; s.soft = std::max(CRATER_SOFT * rim, CRATER_SOFT_MIN); }
}
inline float craterApply(float h, const CraterSum& s) { return s.bowl >= 1e29f ? h + s.ejecta : craterSmin(h + s.ejecta, s.bowl, s.soft); }
// the first n craters (n = all for the live surface)
inline float craterCarveN(float x, float z, float h, int n) {
    CraterSum s;
    for (int i = 0; i < n; i++) craterAcc(gCraters[i], x, z, s);
    return craterApply(h, s);
}
inline float craterCarve(float x, float z, float h) {
    return craterCarveN(x, z, h, std::min(gCraterCount.load(std::memory_order_acquire), MAX_CRATERS));
}

inline bool inCrater(float x, float z, float margin) {
    int n = std::min(gCraterCount.load(std::memory_order_acquire), MAX_CRATERS);
    for (int i = 0; i < n; i++) {
        float dx = x - gCraters[i].x, dz = z - gCraters[i].z, r = gCraters[i].r * 1.15f + margin;
        if (dx * dx + dz * dz < r * r) return true;
    }
    return false;
}

inline float terrainHeight(float x, float z, const Zone* zones, int nz) { return craterCarve(x, z, terrainBase(x, z, zones, nz)); }

#ifdef __AVX2__
// 8 lanes of the same math. Used to fill physics heightfields (129 x 129 samples per tile).
inline __m256 hash2x8(__m256i x, __m256i z) {
    __m256i h = _mm256_add_epi32(_mm256_mullo_epi32(x, _mm256_set1_epi32(374761393)), _mm256_mullo_epi32(z, _mm256_set1_epi32(668265263)));
    h = _mm256_add_epi32(h, _mm256_set1_epi32((int)gSeedMix));
    h = _mm256_mullo_epi32(_mm256_xor_si256(h, _mm256_srli_epi32(h, 13)), _mm256_set1_epi32(1274126177));
    h = _mm256_and_si256(_mm256_xor_si256(h, _mm256_srli_epi32(h, 16)), _mm256_set1_epi32(0xFFFFFF));
    return _mm256_div_ps(_mm256_cvtepi32_ps(h), _mm256_set1_ps(float(0xFFFFFF)));
}
inline __m256 vnoise8(__m256 x, __m256 z) {
    __m256 flx = _mm256_floor_ps(x), flz = _mm256_floor_ps(z);
    __m256i ix = _mm256_cvttps_epi32(flx), iz = _mm256_cvttps_epi32(flz), one = _mm256_set1_epi32(1);
    __m256 fx = _mm256_sub_ps(x, flx), fz = _mm256_sub_ps(z, flz), three = _mm256_set1_ps(3), two = _mm256_set1_ps(2);
    fx = _mm256_mul_ps(_mm256_mul_ps(fx, fx), _mm256_sub_ps(three, _mm256_mul_ps(two, fx)));
    fz = _mm256_mul_ps(_mm256_mul_ps(fz, fz), _mm256_sub_ps(three, _mm256_mul_ps(two, fz)));
    __m256i ix1 = _mm256_add_epi32(ix, one), iz1 = _mm256_add_epi32(iz, one);
    __m256 a = hash2x8(ix, iz), b = hash2x8(ix1, iz), c = hash2x8(ix, iz1), d = hash2x8(ix1, iz1);
    __m256 ab = _mm256_fmadd_ps(_mm256_sub_ps(b, a), fx, a), cd = _mm256_fmadd_ps(_mm256_sub_ps(d, c), fx, c);
    return _mm256_fmadd_ps(_mm256_sub_ps(cd, ab), fz, ab);
}
inline __m256 terrainHeight8(__m256 x, __m256 z, const Zone* zones, int nz) {
    auto s = [](float v) { return _mm256_set1_ps(v); };
    __m256 h = _mm256_mul_ps(s(2.2f), vnoise8(_mm256_fmadd_ps(x, s(0.022f), s(10)), _mm256_mul_ps(z, s(0.022f))));
    h = _mm256_fmadd_ps(s(0.7f), vnoise8(_mm256_mul_ps(x, s(0.07f)), _mm256_fmadd_ps(z, s(0.07f), s(5))), h);
    h = _mm256_fmadd_ps(s(0.12f), vnoise8(_mm256_mul_ps(x, s(0.35f)), _mm256_mul_ps(z, s(0.35f))), h);
    h = _mm256_sub_ps(h, s(1.5f));
    alignas(32) float hx[8], xx[8], zz[8];
    _mm256_store_ps(hx, h); _mm256_store_ps(xx, x); _mm256_store_ps(zz, z);
    // scalar per lane, as terrainBase: the horizon mountains (0 inside 3 km), world map edits, the river channel
    for (int i = 0; i < 8; i++) hx[i] = lakeCarve(xx[i], zz[i], riverCarve(xx[i], zz[i], biomeShape(xx[i], zz[i], hx[i] + hills(xx[i], zz[i]) + farRelief(xx[i], zz[i])) + editHeight(xx[i], zz[i])));
    h = _mm256_load_ps(hx);
    __m256 zero = _mm256_setzero_ps();
    for (int i = 0; i < nz; i++) {
        const Zone& zn = zones[i];
        __m256 dx = _mm256_max_ps(_mm256_max_ps(_mm256_sub_ps(s(zn.minx), x), _mm256_sub_ps(x, s(zn.maxx))), zero);
        __m256 dz = _mm256_max_ps(_mm256_max_ps(_mm256_sub_ps(s(zn.minz), z), _mm256_sub_ps(z, s(zn.maxz))), zero);
        __m256 t = _mm256_min_ps(_mm256_div_ps(_mm256_sqrt_ps(_mm256_fmadd_ps(dx, dx, _mm256_mul_ps(dz, dz))), s(std::max(zn.falloff, 1e-3f))), s(1));
        __m256 w = _mm256_sub_ps(s(1), _mm256_mul_ps(_mm256_mul_ps(t, t), _mm256_fnmadd_ps(s(2), t, s(3))));
        h = _mm256_fmadd_ps(_mm256_sub_ps(s(zn.height), h), w, h);
    }
    if (gCraterCount.load(std::memory_order_acquire) > 0) {  // craters: scalar per lane
        _mm256_store_ps(hx, h);
        for (int i = 0; i < 8; i++) hx[i] = craterCarve(xx[i], zz[i], hx[i]);
        h = _mm256_load_ps(hx);
    }
    return h;
}
#endif

// n x n samples starting at (x0, z0) with the given spacing, row-major by z (out[iz * n + ix]).
inline void terrainGrid(float* out, int n, float x0, float z0, float spacing, const Zone* zones, int nz) {
    for (int iz = 0; iz < n; iz++) {
        float z = z0 + iz * spacing;
        int ix = 0;
#ifdef __AVX2__
        const __m256 lane = _mm256_setr_ps(0, 1, 2, 3, 4, 5, 6, 7);
        for (; ix + 8 <= n; ix += 8) {
            __m256 x = _mm256_fmadd_ps(_mm256_add_ps(lane, _mm256_set1_ps((float)ix)), _mm256_set1_ps(spacing), _mm256_set1_ps(x0));
            _mm256_storeu_ps(out + iz * n + ix, terrainHeight8(x, _mm256_set1_ps(z), zones, nz));
        }
#endif
        for (; ix < n; ix++) out[iz * n + ix] = terrainHeight(x0 + ix * spacing, z, zones, nz);
    }
}

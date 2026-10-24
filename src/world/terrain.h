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

inline float hash2(int x, int z) {
    uint32_t h = (uint32_t)x * 374761393u + (uint32_t)z * 668265263u;
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
// zones. Mirrored in shaders/terrain/river.glsl, parameters in the UBO's river vector.
struct RiverParams { float width = 6.f, depth = 1.8f, level = -1.4f, slope = 0.0003f, bumps = 0.f; };  // half-width, depth (m), bed roughness (m)
inline RiverParams gRiver;
inline constexpr float RIVER_Z = -58.f, RIVER_BANK = 16.f;  // centreline offset, bank blend width past the water edge
inline float riverCenter(float x) { return RIVER_Z + 14.f * sinf(0.011f * x) + 7.f * sinf(0.027f * x + 1.3f); }
inline float riverSlope(float x) { return 0.154f * cosf(0.011f * x) + 0.189f * cosf(0.027f * x + 1.3f); }
struct RiverInfo { float d, w, depth, level; };  // signed distance across the flow (+ = left bank), half-width, depth, water level
inline RiverInfo riverInfo(float x, float z) {
    float dz = riverSlope(x);
    return {(z - riverCenter(x)) / sqrtf(1.f + dz * dz), gRiver.width * (0.8f + 0.45f * vnoise(x * 0.008f + 3.f, 7.5f)),
            gRiver.depth * (0.45f + 1.f * vnoise(x * 0.013f, 2.5f)), gRiver.level - gRiver.slope * x};
}
// parabolic bed inside the water, gentle bank rising out of it, blended back into the natural ground
inline float riverCarve(float x, float z, float h) {
    if (fabsf(z - riverCenter(x)) > 3.f * gRiver.width + RIVER_BANK + 6.f) return h;  // cheap reject (w <= 1.25 width, |slope| < 0.35)
    RiverInfo r = riverInfo(x, z);
    float d = fabsf(r.d), c = d / r.w;
    float prof = d < r.w ? r.level - r.depth * (1.f - c * c) : r.level + 0.35f * (d - r.w);
    if (gRiver.bumps > 0.f && d < r.w + 2.f) {  // uneven bed: humps, scours and stony ridges, strongest mid-channel
        float n = vnoise(x * 0.45f, z * 0.45f) * 0.6f + vnoise(x * 1.3f + 7.f, z * 1.3f) * 0.4f - 0.5f;
        prof += 2.f * gRiver.bumps * n * sqrtf(std::max(1.f - c * c, 0.f));
    }
    float t = std::clamp((d - r.w - 2.f) / (RIVER_BANK - 2.f), 0.f, 1.f);
    return prof + (h - prof) * t * t * (3 - 2 * t);
}
// Rocks in the channel, one candidate per RIVER_ROCK_L stretch. Deterministic so the water shader (riverRock) knows where
// they are and shapes the surface around them. Rock mesh: radius ~1, y in [-0.52, 0.72] (buildRockMesh).
// Boulders are sized and lifted so they break the surface; small stones sit on the bed.
inline constexpr float RIVER_ROCK_L = 2.5f;
inline bool riverRock(int i, float& x, float& y, float& z, float& sc) {
    if (hash2(i * 7 + 3, 811) > 0.65f) return false;
    bool boulder = hash2(i * 5 + 1, 823) < 0.35f;
    x = (i + hash2(i, 853)) * RIVER_ROCK_L;
    float hc = hash2(i * 3 + 2, 839) * 2.f - 1.f;  // small stones mostly in the shallows
    float c = boulder ? hc * 0.75f : (hc < 0 ? -1.f : 1.f) * (0.45f + 0.5f * fabsf(hc)), dz = riverSlope(x);
    RiverInfo r = riverInfo(x, 0.f);
    z = riverCenter(x) + c * r.w * sqrtf(1.f + dz * dz);
    float bed = r.level - r.depth * (1.f - c * c);
    sc = boulder ? std::min(std::max(0.9f + 0.8f * hash2(i, 857), (r.level - bed) * 0.9f), 2.6f) : 0.25f + 0.45f * hash2(i, 859);
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
    if (fabsf(z - riverCenter(x)) > 3.f * gRiver.width + 2.f) return;
    RiverInfo r = riverInfo(x, z);
    float c = fabsf(r.d) / r.w;
    if (c >= 1.f) return;
    float dz = riverSlope(x), inv = 1.f / sqrtf(1.f + dz * dz);
    float U = strength * 1.3f * (gRiver.width * gRiver.depth) / (r.w * r.depth) * powf(std::max(1.f - c * c, 0.02f), 0.35f);
    v[0] = U * inv;
    v[1] = U * dz * inv;
}

// 0 in the water, 1 on dry ground (placement)
inline float riverDry(float x, float z, float margin) {
    if (fabsf(z - riverCenter(x)) > 3.f * gRiver.width + margin + 2.f) return 1.f;
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
    return applyZones(x, z, riverCarve(x, z, naturalHeight(x, z) + farRelief(x, z) + editHeight(x, z)), zones, nz);
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
    for (int i = 0; i < 8; i++) hx[i] = riverCarve(xx[i], zz[i], hx[i] + editHeight(xx[i], zz[i]));  // scalar: only near the channel
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

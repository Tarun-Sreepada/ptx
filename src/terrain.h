// Terrain height shared by the renderer (light/rock placement), the physics heightfields and the streamer.
// Mirrors terrainHeight() in shaders/material.glsl: value noise + flatten zones (city plots, roads) that
// pull the ground to a fixed height inside a rectangle with a smooth falloff band around it.
#pragma once
#include <immintrin.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <atomic>
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
// zones. Mirrored in material.glsl (riverInfo / riverCarve), parameters in the UBO's river vector.
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
// Rocks in the channel, one candidate per RIVER_ROCK_L stretch. Deterministic so water.glsl (riverRock) knows where
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
    // mesh bottom (y - 0.52 sc) dug into it, so no side floats (mirrored in material.glsl)
    float dc = c * r.w, f = 0.8f * sc, nearest = (dc - f <= 0.f && dc + f >= 0.f) ? 0.f : std::min(fabsf(dc - f), fabsf(dc + f));
    float cn = nearest / r.w;
    float minBed = nearest < r.w ? r.level - r.depth * (1.f - cn * cn) : r.level + 0.35f * (nearest - r.w);
    y = std::min(y, minBed + 0.4f * sc);
    return true;
}

// Surface current (m/s, world xz) for a given flow strength; mirrors riverFlow() in water.glsl.
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

// Impact craters, mirrored in material.glsl (craterCarve). Written by the main thread when something hits the
// ground hard enough; read by the loader threads that rebuild collision tiles (stream::terrainChanged()).
struct Crater { float x, z, r, depth; };
inline constexpr int MAX_CRATERS = 256;
// fixed slots + release/acquire count so a loader thread never sees a half-written crater. A new impact inside an
// existing crater grows that one instead of taking a slot; when all slots are used the smallest crater is
// recycled (either can briefly tear a tile being rebuilt; it's rebuilt again right after)
inline Crater gCraters[MAX_CRATERS];
inline std::atomic<int> gCraterCount{0};
// Overlapping craters don't stack: the ground is as deep as the deepest bowl there (smooth max, no crease), and a
// rim only stands where no other bowl has dug (it fades out inside one). Mirrored in material.glsl (craterAcc).
inline float craterSmax(float a, float b, float k) { float h = std::max(k - fabsf(a - b), 0.f) / k; return std::max(a, b) + h * h * k * 0.25f; }
inline void craterAcc(const Crater& c, float x, float z, float& D, float& R) {
    float dx = x - c.x, dz = z - c.z, t2 = (dx * dx + dz * dz) / (c.r * c.r);
    if (t2 > 3.24f) return;
    float t = sqrtf(t2), rim = (t - 1.f) / 0.28f;
    float d = c.depth * std::max(1.f - t2, 0.f);
    if (d > 0.f) D = craterSmax(D, d, 0.15f * c.depth + 0.02f);
    R = std::max(R, 0.3f * c.depth * expf(-rim * rim));
}
inline float craterApply(float h, float D, float R) {
    float s = std::clamp((D - 0.02f) / 0.28f, 0.f, 1.f);
    return h - D + R * (1.f - s * s * (3.f - 2.f * s));
}
inline float craterCarve(float x, float z, float h) {
    int n = std::min(gCraterCount.load(std::memory_order_acquire), MAX_CRATERS);
    float D = 0.f, R = 0.f;
    for (int i = 0; i < n; i++) craterAcc(gCraters[i], x, z, D, R);
    return craterApply(h, D, R);
}

inline bool inCrater(float x, float z, float margin) {
    int n = std::min(gCraterCount.load(std::memory_order_acquire), MAX_CRATERS);
    for (int i = 0; i < n; i++) {
        float dx = x - gCraters[i].x, dz = z - gCraters[i].z, r = gCraters[i].r * 1.15f + margin;
        if (dx * dx + dz * dz < r * r) return true;
    }
    return false;
}

// 0 outside, 1 inside the rectangle, smoothstep across the falloff band
inline float zoneWeight(const Zone& zn, float x, float z) {
    float dx = std::max(std::max(zn.minx - x, x - zn.maxx), 0.f), dz = std::max(std::max(zn.minz - z, z - zn.maxz), 0.f);
    float t = std::min(sqrtf(dx * dx + dz * dz) / std::max(zn.falloff, 1e-3f), 1.f);
    return 1.f - t * t * (3 - 2 * t);
}
inline float terrainHeight(float x, float z, const Zone* zones, int nz) {
    float h = craterCarve(x, z, riverCarve(x, z, naturalHeight(x, z) + editHeight(x, z)));
    for (int i = 0; i < nz; i++) h += (zones[i].height - h) * zoneWeight(zones[i], x, z);
    return h;
}
inline float zoneCover(float x, float z, const Zone* zones, int nz) {
    float w = 0;
    for (int i = 0; i < nz; i++) w = std::max(w, zoneWeight(zones[i], x, z));
    return w;
}

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
    {  // river: scalar per lane, only where a lane is near the channel
        alignas(32) float hx[8], xx[8], zz[8];
        _mm256_store_ps(hx, h); _mm256_store_ps(xx, x); _mm256_store_ps(zz, z);
        for (int i = 0; i < 8; i++) hx[i] = craterCarve(xx[i], zz[i], riverCarve(xx[i], zz[i], hx[i] + editHeight(xx[i], zz[i])));
        h = _mm256_load_ps(hx);
    }
    __m256 zero = _mm256_setzero_ps();
    for (int i = 0; i < nz; i++) {
        const Zone& zn = zones[i];
        __m256 dx = _mm256_max_ps(_mm256_max_ps(_mm256_sub_ps(s(zn.minx), x), _mm256_sub_ps(x, s(zn.maxx))), zero);
        __m256 dz = _mm256_max_ps(_mm256_max_ps(_mm256_sub_ps(s(zn.minz), z), _mm256_sub_ps(z, s(zn.maxz))), zero);
        __m256 t = _mm256_min_ps(_mm256_div_ps(_mm256_sqrt_ps(_mm256_fmadd_ps(dx, dx, _mm256_mul_ps(dz, dz))), s(std::max(zn.falloff, 1e-3f))), s(1));
        __m256 w = _mm256_sub_ps(s(1), _mm256_mul_ps(_mm256_mul_ps(t, t), _mm256_fnmadd_ps(s(2), t, s(3))));
        h = _mm256_fmadd_ps(_mm256_sub_ps(s(zn.height), h), w, h);
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

#include "scene/rocks.h"
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <random>
#include <cstdint>
#include <cstddef>
#include <vulkan/vulkan.h>
#include "world/terrain.h"
#include "core/limits.h"
#include "core/math.h"
#include "gpu/accel.h"
#include "render/frame_data.h"
#include "scene/scene_gpu.h"

std::vector<float> rockPts;
std::vector<uint32_t> rockTriIdx;

void* rockHull[16] = {};

// Procedural boulder, instanced as rocks. Real stones are broken blocks worn round, not lumpy spheres:
//  1. fracture: the radius in each direction is a soft minimum over a dozen random cleavage planes (plus a bounding
//     sphere), so the stone has flat-ish faces; the softness is the weathering that rounds the edges and corners
//  2. bedding: shallow ledges along one tilted axis (layered rock splits along its strata)
//  3. cracks: a few joints cut in as narrow grooves
//  4. erosion: ridged 3D noise (sharp crests, soft hollows) plus broad lumps; the base is flattened where it sits
static float rockHash(int x, int y, int z) {
    uint32_t h = (uint32_t)x * 73856093u ^ (uint32_t)y * 19349663u ^ (uint32_t)z * 83492791u;
    h ^= h >> 13; h *= 0x5bd1e995u; h ^= h >> 15;
    return (h & 0xFFFFFF) / 16777216.f;
}

static float rockNoise(V3 p) {
    int ix = (int)floorf(p.x), iy = (int)floorf(p.y), iz = (int)floorf(p.z);
    float fx = p.x - ix, fy = p.y - iy, fz = p.z - iz;
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy); fz = fz * fz * (3 - 2 * fz);
    auto L = [](float a, float b, float t) { return a + (b - a) * t; };
    float c[2][2];
    for (int j = 0; j < 2; j++)
        for (int k = 0; k < 2; k++) c[j][k] = L(rockHash(ix, iy + j, iz + k), rockHash(ix + 1, iy + j, iz + k), fx);
    return L(L(c[0][0], c[1][0], fy), L(c[0][1], c[1][1], fy), fz);
}

static float rockSmooth(float e0, float e1, float x) { x = std::clamp((x - e0) / (e1 - e0), 0.f, 1.f); return x * x * (3 - 2 * x); }

void buildRockMesh(uint32_t seed, std::vector<float>& verts, std::vector<uint32_t>& idx) {
    const float t = 1.618034f;
    std::vector<V3> v = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t},
                         {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
    idx = {0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
           3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1};
    for (int sub = 0; sub < 4; sub++) {
        std::vector<uint32_t> out;
        std::unordered_map<uint64_t, uint32_t> cache;
        auto mid = [&](uint32_t a, uint32_t b) {
            uint64_t key = ((uint64_t)std::min(a, b) << 32) | std::max(a, b);
            auto it = cache.find(key);
            if (it != cache.end()) return it->second;
            v.push_back((v[a] + v[b]) * 0.5f);
            return cache[key] = (uint32_t)v.size() - 1;
        };
        for (size_t i = 0; i < idx.size(); i += 3) {
            uint32_t a = idx[i], b = idx[i + 1], c = idx[i + 2], ab = mid(a, b), bc = mid(b, c), ca = mid(c, a);
            out.insert(out.end(), {a, ab, ca, b, bc, ab, c, ca, bc, ab, bc, ca});
        }
        idx = out;
    }
    std::mt19937 rng(seed * 2654435761u + 20261023u);
    std::uniform_real_distribution<float> U(0.f, 1.f);
    // per-variant character: few big facets (blocky) .. many (rounded), sharp .. worn, flat slab .. tall, elongated
    const int NP = 6 + (int)(U(rng) * 11);
    const float k = 8.f + 22.f * U(rng);                     // edge softness (lower = more worn)
    const float jitter = 0.3f + 0.6f * U(rng), depthVar = 0.15f + 0.3f * U(rng);
    const float bedAmt = U(rng) < 0.5f ? 0.f : 0.01f + 0.03f * U(rng), bedFreq = 4.f + 6.f * U(rng);
    const float ridgeAmt = 0.02f + 0.07f * U(rng), lumpAmt = 0.05f + 0.15f * U(rng);
    const float sy = 0.45f + 0.45f * U(rng), sx = 0.8f + 0.5f * U(rng), sz = 0.8f + 0.3f * U(rng);
    const V3 noiseOff{U(rng) * 100.f, U(rng) * 100.f, U(rng) * 100.f};
    // cleavage planes: spread over the sphere (golden spiral), jittered, at varied depths
    V3 pn[16]; float pd[16];
    for (int i = 0; i < NP; i++) {
        float y = 1.f - 2.f * (i + 0.5f) / NP, r = sqrtf(1.f - y * y), a = i * 2.39996f;
        pn[i] = normalize(V3{cosf(a) * r + (U(rng) - 0.5f) * jitter, y + (U(rng) - 0.5f) * jitter, sinf(a) * r + (U(rng) - 0.5f) * jitter});
        pd[i] = 1.f - depthVar * U(rng);
    }
    V3 bed = normalize(V3{U(rng) - 0.5f, 1.f, U(rng) - 0.5f});  // strata axis
    V3 jn[3]; float jd[3];
    for (int i = 0; i < 3; i++) { jn[i] = normalize(V3{U(rng) - 0.5f, U(rng) - 0.5f, U(rng) - 0.5f}); jd[i] = (U(rng) - 0.5f) * 0.9f; }
    for (auto& p : v) {
        V3 d = normalize(p);
        float sum = expf(-k * 1.2f);
        for (int i = 0; i < NP; i++) sum += expf(-k * pd[i] / std::max(dot(d, pn[i]), 0.06f));
        float r = -logf(sum) / k;
        V3 q = d * r;
        q = q + noiseOff;
        float s = dot(q, bed) * bedFreq + 0.6f * rockNoise(q * 2.f);
        float f = s - floorf(s);
        r -= bedAmt * rockSmooth(0.75f, 1.f, f);  // a step at each bedding plane
        for (int i = 0; i < 3; i++) {
            float w = fabsf(dot(q - noiseOff, jn[i]) - jd[i] - 0.04f * (rockNoise(q * 5.f + V3{3, 1, 7}) - 0.5f));
            r -= 0.015f * std::max(0.f, 1.f - w / 0.015f) * rockSmooth(0.2f, 0.6f, rockNoise(q * 1.5f + V3{(float)i * 9, 0, 0}));
        }
        float ridged = 0.f, amp = 0.5f, fr = 3.f;
        for (int o = 0; o < 4; o++, amp *= 0.5f, fr *= 2.1f) ridged += amp * (1.f - fabsf(2.f * rockNoise(q * fr + V3{(float)o * 17, 5, 3}) - 1.f));
        r += ridgeAmt * (ridged - 0.5f) + lumpAmt * (rockNoise(q * 1.2f + V3{11, 3, 5}) - 0.5f);
        p = d * r;
        p.x *= sx; p.z *= sz;
        p.y = p.y * sy + 0.1f;
        if (p.y < -0.3f) p.y = -0.3f + (p.y + 0.3f) * 0.35f;  // sat flat in the ground
    }
    std::vector<V3> n(v.size());
    for (size_t i = 0; i < idx.size(); i += 3) {
        V3 fn = cross(v[idx[i + 1]] - v[idx[i]], v[idx[i + 2]] - v[idx[i]]);
        for (int k = 0; k < 3; k++) n[idx[i + k]] = n[idx[i + k]] + fn;
    }
    for (size_t i = 0; i < v.size(); i++) {
        V3 nn = normalize(n[i]);
        verts.insert(verts.end(), {v[i].x, v[i].y, v[i].z, nn.x, nn.y, nn.z});
    }
}

// A rock instance: TLAS entry (the variant's BLAS; its base vertex in the SBT record offset, which ray queries
// don't use otherwise, so hit shading finds its normals in rv) and the raster transform (rockXf slot, 4 rows).
uint32_t rockVarOf(float x, float z) { return (uint32_t)(hash2((int)lroundf(x * 8.f) + 71, (int)lroundf(z * 8.f) + 313) * ROCK_VARIANTS) % ROCK_VARIANTS; }

void setRockInstance(VkAccelerationStructureInstanceKHR& in, uint32_t var, const float m[3][4]) {
    setInstance(in, 2, rockAS[var].addr, m);
    in.instanceShaderBindingTableRecordOffset = var * rockVerts;
}

void writeRockXf(float* xf, uint32_t slot, const float m[3][4], const float prev[3][4], uint32_t var) {
    for (int half = 0; half < 2; half++) {
        float* o = xf + (half ? (MAX_ROCKS + MAX_THROWN) * 16 : 0) + slot * 16;
        memcpy(o, half ? prev : m, 48);
        uint32_t base = var * rockVerts;
        memcpy(o + 12, &base, 4);
        o[13] = o[14] = o[15] = 0.f;
    }
}

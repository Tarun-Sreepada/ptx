#include "scene/ground_mask.h"
#include <algorithm>
#include <cmath>
#include <cstdint>

// CPU mirror of material.glsl vnoise / grassMask (without river, zones, craters): bare ground for clutter placement.
static float vnoiseG(float x, float y) {
    auto hu = [](uint32_t v) { v ^= v >> 16; v *= 0x7feb352du; v ^= v >> 15; v *= 0x846ca68bu; v ^= v >> 16; return v; };
    float ix = floorf(x), iy = floorf(y), fx = x - ix, fy = y - iy;
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy);
    auto h = [&](float a, float b) { return hu((uint32_t)(int)a * 7919u + (uint32_t)(int)b * 104729u) / 4294967295.f; };
    float a = h(ix, iy), b = h(ix + 1, iy), c = h(ix, iy + 1), d = h(ix + 1, iy + 1);
    return a + (b - a) * fx + (c - a) * fy + (a - b - c + d) * fx * fy;
}

float grassMaskCPU(float x, float z) {
    auto ss = [](float a, float b, float v) { float t = std::clamp((v - a) / (b - a), 0.f, 1.f); return t * t * (3 - 2 * t); };
    float n = vnoiseG(x * 0.06f, z * 0.06f) * 0.65f + vnoiseG(x * 0.21f, z * 0.21f) * 0.35f;
    float path = fabsf(z - 9.f * sinf(x * 0.045f) - 4.f * sinf(x * 0.11f + 1.f));
    return ss(0.26f, 0.42f, n) * ss(0.9f, 1.7f, path);
}

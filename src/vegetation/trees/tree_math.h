// Small vector math, a hashing RNG and 3D value noise shared by the vegetation generators.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

#include "world/world.h"

namespace trees {
inline constexpr float PI = 3.14159265f, TAU = 6.2831853f;
inline constexpr float GOLDEN_ANGLE = 2.3999632f;  // radians between successive buds (phyllotaxis)

struct V { float x, y, z; };
inline V operator+(V a, V b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
inline V operator-(V a, V b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
inline V operator*(V a, float s) { return {a.x * s, a.y * s, a.z * s}; }
inline float dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
inline V cross(V a, V b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
inline float length(V a) { return sqrtf(dot(a, a)); }
inline V norm(V a) { float l = length(a); return l > 0 ? a * (1.f / l) : V{0, 1, 0}; }
// v rotated about the unit axis k by angle a (Rodrigues)
inline V rot(V v, V k, float a) {
    float c = cosf(a), s = sinf(a);
    return v * c + cross(k, v) * s + k * (dot(k, v) * (1.f - c));
}
// some unit vector perpendicular to d
inline V perpTo(V d) { return norm(cross(d, fabsf(d.y) < 0.95f ? V{0, 1, 0} : V{1, 0, 0})); }

struct Rng {
    uint32_t s;
    float operator()() {
        s = s * 747796405u + 2891336453u;
        uint32_t w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
        return ((w >> 22u) ^ w) / 4294967296.f;
    }
    float in(float a, float b) { return a + (b - a) * (*this)(); }
};
inline V randUnit(Rng& r) {
    float z = r.in(-1.f, 1.f), a = r.in(0.f, TAU), s = sqrtf(std::max(0.f, 1.f - z * z));
    return {s * cosf(a), z, s * sinf(a)};
}

// 3D value noise, 0..1 (canopy lumps, bark flutes)
inline float noise3(V p) {
    auto h3 = [](int x, int y, int z) {  // wrapping (unsigned) multiplies
        return hash2((int)((uint32_t)x * 73856093u ^ (uint32_t)z * 19349663u), (int)((uint32_t)y * 83492791u));
    };
    int ix = (int)floorf(p.x), iy = (int)floorf(p.y), iz = (int)floorf(p.z);
    float fx = p.x - ix, fy = p.y - iy, fz = p.z - iz;
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy); fz = fz * fz * (3 - 2 * fz);
    auto L = [](float a, float b, float t) { return a + (b - a) * t; };
    float a = L(L(h3(ix, iy, iz), h3(ix + 1, iy, iz), fx), L(h3(ix, iy + 1, iz), h3(ix + 1, iy + 1, iz), fx), fy);
    float b = L(L(h3(ix, iy, iz + 1), h3(ix + 1, iy, iz + 1), fx), L(h3(ix, iy + 1, iz + 1), h3(ix + 1, iy + 1, iz + 1), fx), fy);
    return L(a, b, fz);
}
}  // namespace trees

#pragma once
#include <cmath>

// ---------------------------------------------------------------- math
struct V3 { float x = 0, y = 0, z = 0; };

inline V3 operator+(V3 a, V3 b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }

inline V3 operator-(V3 a, V3 b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }

inline V3 operator*(V3 a, float s) { return {a.x * s, a.y * s, a.z * s}; }

inline float dot(V3 a, V3 b) { return a.x * b.x + a.y * b.y + a.z * b.z; }

inline V3 cross(V3 a, V3 b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }

inline V3 normalize(V3 a) { return a * (1.f / sqrtf(dot(a, a))); }

inline void set4(float* d, V3 v, float w) { d[0] = v.x; d[1] = v.y; d[2] = v.z; d[3] = w; }

// Rotation of v about unit axis k by angle a (Rodrigues).
inline V3 rotateAbout(V3 v, V3 k, float a) {
    float c = cosf(a), s = sinf(a);
    return v * c + cross(k, v) * s + k * (dot(k, v) * (1.f - c));
}

inline float vlen(V3 a) { return sqrtf(dot(a, a)); }

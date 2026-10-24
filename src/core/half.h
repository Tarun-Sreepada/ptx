#pragma once
#include <cstring>
#include <cstdint>

inline float halfToFloat(uint16_t h) {
    uint32_t s = (h & 0x8000u) << 16, e = (h >> 10) & 0x1f, m = h & 0x3ff, f;
    if (e == 0) { if (!m) f = s; else { e = 1; while (!(m & 0x400)) { m <<= 1; e--; } m &= 0x3ff; f = s | ((e + 112) << 23) | (m << 13); } }
    else if (e == 31) f = s | 0x7f800000u | (m << 13);
    else f = s | ((e + 112) << 23) | (m << 13);
    float r;
    memcpy(&r, &f, 4);
    return r;
}  // keep refreshing the terrain mesh a while after snow was added (avalanches settle)         // snow depth / water depth / wetness / avalanche; water + snow fluxes  // wind map ping-pong (wind.glsl)

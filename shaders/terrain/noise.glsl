// Hashes and value noise shared by the terrain, materials and effects. Terrain noise mirrors world/terrain.h.
#ifndef TERRAIN_NOISE
#define TERRAIN_NOISE
#include "core/frame.glsl"

float sq(float x) { return x * x; }
uint hashu(uint x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; }

// (+ the world seed's mix: ground detail, dirt and anything else built on it follow the seed)
float vnoise(vec2 p) {
    vec2 i = floor(p), f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    uint sm = F.worldSeed.x;
    float a = float(hashu(uint(int(i.x)) * 7919u + uint(int(i.y)) * 104729u + sm)) / 4294967295.0;
    float b = float(hashu(uint(int(i.x + 1.0)) * 7919u + uint(int(i.y)) * 104729u + sm)) / 4294967295.0;
    float c = float(hashu(uint(int(i.x)) * 7919u + uint(int(i.y + 1.0)) * 104729u + sm)) / 4294967295.0;
    float d = float(hashu(uint(int(i.x + 1.0)) * 7919u + uint(int(i.y + 1.0)) * 104729u + sm)) / 4294967295.0;
    return mix(mix(a, b, f.x), mix(c, d, f.x), f.y);
}

// Infinite terrain height. Mirrors terrain.h (CPU: light/rock placement, physics heightfields).
float thash(int x, int z) {
    uint h = uint(x) * 374761393u + uint(z) * 668265263u + F.worldSeed.x;  // terrain.h hash2 (+ gSeedMix)
    h = (h ^ (h >> 13)) * 1274126177u;
    return float((h ^ (h >> 16)) & 0xFFFFFFu) / float(0xFFFFFF);
}
float tnoise(float x, float z) {
    int ix = int(floor(x)), iz = int(floor(z));
    float fx = x - float(ix), fz = z - float(iz);
    fx = fx * fx * (3.0 - 2.0 * fx);
    fz = fz * fz * (3.0 - 2.0 * fz);
    float a = thash(ix, iz), b = thash(ix + 1, iz), c = thash(ix, iz + 1), d = thash(ix + 1, iz + 1);
    return mix(mix(a, b, fx), mix(c, d, fx), fz);
}
#endif

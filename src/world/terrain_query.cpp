#include "world/terrain_query.h"
#include "world/terrain.h"
#include "world/world.h"
#include "app/settings.h"

// ---------------------------------------------------------------- scene generation
// terrain height including the currently streamed flatten zones
float heightFn(float x, float z) {
    auto& zs = stream::zones();
    return terrainHeight(x, z, zs.data(), (int)zs.size());
}

void setWorldSeed(uint32_t seed) {
    gWorld.seed = seed;
    auto mix = [](uint32_t x) { x ^= x >> 16; x *= 0x7feb352du; x ^= x >> 15; x *= 0x846ca68bu; x ^= x >> 16; return x; };
    gSeedMix = seed ? mix(seed) | 1u : 0u;
    RiverShape r;
    if (seed) {  // a different meander per seed: amplitudes, wavelengths, phases
        auto u = [&](uint32_t k) { return (mix(seed * 2654435761u + k * 40503u) & 0xFFFFFF) / float(0xFFFFFF); };
        r.a1 = 9.f + 14.f * u(1); r.a2 = 3.f + 8.f * u(2);
        r.f1 = 0.007f + 0.007f * u(3); r.f2 = 0.019f + 0.014f * u(4);
        r.p1 = 6.2831853f * u(5); r.p2 = 6.2831853f * u(6);
    }
    gRiverShape = r;
}

void applyWorldGen() {
    if ((uint32_t)S.worldSeed != gWorld.seed) setWorldSeed((uint32_t)S.worldSeed);
    gWorld.hilliness = S.hilliness;
    gRiver.widthVar = S.riverWidthVar;
    gRiver.braids = S.riverBraids;
    gRiver.tributaries = S.riverTribs;
}

// the ground before lakes (biome.h lakeLevel: a lake fills to its centre's height)
float lakeFreeHeight(float x, float z) {
    return riverCarve(x, z, biomeShape(x, z, naturalHeight(x, z) + hills(x, z) + farRelief(x, z)) + editHeight(x, z));
}

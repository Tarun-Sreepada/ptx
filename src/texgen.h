// Procedural ground texture generator: tileable, high resolution albedo + normal/height/AO sets built from
// scattered stamps of individually shaped stones (rounded pebbles, angular rock, flat chips, lumpy cobbles),
// grit and soil. Results are cached on disk (texcache/) keyed by kind, size, seed and generator version.
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace texgen {
enum Kind { GROUND_BARE, GROUND_SOIL, KIND_COUNT };  // stony bare ground, organic soil under grass

struct Params {
    uint32_t size = 4096;  // texels per side
    float tile = 4.f;      // metres covered by one tile
    uint32_t seed = 1;
    float stones = 1.f;    // stone count multiplier
};

struct Set {
    uint32_t size = 0;
    std::vector<uint8_t> albedo;  // RGBA8 sRGB, a = stone mask
    std::vector<uint8_t> detail;  // RGBA8 linear: normal x, normal z (0.5 bias), height (0..1 of maxHeight), AO
    float maxHeight = 0.f;        // metres represented by height = 1
};

Set generate(Kind k, const Params& p);
// generate() with a disk cache under dir; returns true if it came from the cache
bool load(Kind k, const Params& p, const std::string& dir, Set& out);
}  // namespace texgen

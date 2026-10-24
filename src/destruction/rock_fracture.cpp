#include "destruction/rock_fracture.h"
#include <vector>
#include <cstdint>
#include "physics/fracture.h"
#include "world/terrain.h"
#include "world/world.h"

std::vector<FractureReq> pendingFractures;

// Rock solid (body space at scale 1) and its surface textures for fragments.
fracture::Poly rockPoly[16];

TextureData rockTexture() {
    TextureData t;
    t.w = t.h = 64;
    t.rgba.resize(64 * 64 * 4);
    for (uint32_t y = 0; y < 64; y++)
        for (uint32_t x = 0; x < 64; x++) {
            float k = vnoise(x * 0.25f, y * 0.25f) * 0.6f + vnoise(x * 0.9f + 7.f, y * 0.9f) * 0.4f;
            uint8_t v = (uint8_t)(150.f + 90.f * k), *p = &t.rgba[(y * 64 + x) * 4];
            p[0] = v; p[1] = (uint8_t)(v * 0.97f); p[2] = (uint8_t)(v * 0.92f); p[3] = 255;
        }
    return t;
}

// Fracture: a blow carrying normal energy En into a solid of cross-section A cracks it once En exceeds Gc * A
// (Gc = effective fracture energy per m^2 of crack, incl. what goes into heat and sound; tuned so a fist-sized
// stone survives a throw and a boulder breaks on a hard landing, big things being weaker for their size).
// Energy beyond that opens more crack surface: more, smaller pieces, the pattern spreading further from the
// contact as the ratio grows. Pieces under ~4 cm go to gravel particles.
static constexpr float FRACTURE_GC = 3e4f;

float fractureThreshold(float radius) { return FRACTURE_GC * 3.14159f * radius * radius; }

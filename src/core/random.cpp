#include "core/random.h"
#include <cstdint>
#include "core/math.h"

// ---------------------------------------------------------------- input / UI
static uint32_t debrisSeed = 1;

float rnd01() {
    debrisSeed = debrisSeed * 747796405u + 2891336453u;
    return (float)((debrisSeed >> 8) & 0xFFFFFF) / 16777215.f;
}

V3 rndDir() {
    V3 d{rnd01() * 2.f - 1.f, rnd01() * 2.f - 1.f, rnd01() * 2.f - 1.f};
    return dot(d, d) > 1e-4f ? normalize(d) : V3{0, 1, 0};
}

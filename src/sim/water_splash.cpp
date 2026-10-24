#include "sim/water_splash.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include "app/frame_state.h"
#include "core/math.h"
#include "core/random.h"
#include "sim/debris.h"
#include "audio/audio.h"

std::vector<SplashEvent> splashEvents;

namespace {
constexpr float SPLASH_SOUND_MIN = 0.08f;   // splash strength below which it's silent
constexpr float SPLASH_BIG_SOUND = 0.9f;    // the heavy splash sound from here
constexpr double SPLASH_SOUND_GAP = 0.06;   // s between small splash sounds (a skimming stone's wake)
}  // namespace

// Something of size sc (m) hit the water at velocity v: a splash event for the surface (water.glsl) plus spray,
// a crown sheet breaking into droplets that fly up and out, then a delayed Worthington jet from the cavity.
// Impact weighs the vertical speed most (a flat skip throws a small splash, a plunge a big one).
void waterImpact(V3 at, float sc, const float v[3], float scale) {
    float vh = sqrtf(v[0] * v[0] + v[2] * v[2]), vy = fabsf(v[1]);
    float e = std::min(sqrtf(sc) * (0.3f + (0.8f * vy + 0.25f * vh) / 8.f), 3.f) * scale;
    // full: replace the event with the least left of it (impact x foam remaining), never a big fresh splash for
    // the plop of a clod -- FIFO eviction let a burst of debris re-entries wipe the rock's own splash at once
    if (splashEvents.size() >= (size_t)MAX_SPLASH_EV) {
        size_t w = 0;
        float wv = 1e30f;
        for (size_t i = 0; i < splashEvents.size(); i++) {
            float left = splashEvents[i].impact * expf(-(float)(simTime - splashEvents[i].t) / SPLASH_FOAM_TAU);
            if (left < wv) { wv = left; w = i; }
        }
        if (wv > e) return;
        splashEvents.erase(splashEvents.begin() + w);
    }
    splashEvents.push_back({at.x, at.z, (float)simTime, e, sc, sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])});
    {   // the sound: a plop for small things, a heavy splash for big ones; wakes only when they're loud enough
        static double lastSound = -1e9;
        bool big = e > SPLASH_BIG_SOUND;
        if (e > SPLASH_SOUND_MIN && (big || simTime - lastSound > SPLASH_SOUND_GAP)) {
            lastSound = simTime;
            float pos[3] = {at.x, at.y, at.z};
            audio::emitterPlay(audio::SND_SPLASH + (big ? 1u : 0u), pos, std::clamp(0.3f * e, 0.05f, 1.2f), false,
                               std::clamp(1.25f - 0.2f * sc, 0.6f, 1.4f) * (0.9f + 0.2f * rnd01()));
        }
    }
    int n = std::min((int)(60.f * e * std::max(sc, 0.5f)), 500);
    V3 carry = V3{v[0], 0, v[2]} * 0.25f;  // a skipping stone throws its spray forward
    for (int i = 0; i < n; i++) {
        bool jet = i < n / 5 && scale >= 1.f;
        float a = rnd01() * 6.2831853f, se = sqrtf(e);
        float out = jet ? rnd01() * 0.4f : (1.2f + 2.5f * rnd01()) * se, up = jet ? (3.f + 3.f * rnd01()) * se : (1.5f + 3.5f * rnd01()) * se;
        float r0 = jet ? 0.1f * rnd01() * sc : (0.5f + 0.4f * rnd01()) * sc;
        float size = (0.015f + 0.05f * rnd01() * rnd01()) * (0.6f + 0.9f * std::min(sc, 2.f)) * (jet ? 1.6f : 1.f);
        addDebris(DEB_WATER, at + V3{cosf(a) * r0, 0.02f, sinf(a) * r0}, V3{cosf(a) * out, up, sinf(a) * out} + (jet ? V3{} : carry), size,
                  jet ? 0.18f + 0.1f * rnd01() : 0.03f * rnd01());
    }
}

#include "sim/debris.h"
#include "sim/fire.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include "world/terrain.h"
#include "app/frame_state.h"
#include "app/settings.h"
#include "core/limits.h"
#include "core/math.h"
#include "core/random.h"
#include "scene/camera.h"
#include "sim/water.h"
#include "sim/water_splash.h"
#include "world/terrain_query.h"

std::vector<Droplet> spray;

uint32_t numSprayInst = 0;

namespace {
constexpr float EMBER_LIFE = 3.f;      // s until an ember has cooled to nothing
constexpr float EMBER_LIFT = 1.4f;     // hot air lifts it this many g at first ...
constexpr float EMBER_COOL = 1.2f;     // ... fading over this many seconds
constexpr float EMBER_DRAG = 1.8f;     // 1/s: light and fluffy, it goes with the air
}  // namespace

// Particle transform. Water stretches along its velocity (reads as a streak of spray); clods and splinters
// tumble about their spin axis; embers shrink as they cool.
void dropletXf(const Droplet& d, float m[3][4]) {
    float sp = sqrtf(dot(d.v, d.v));
    V3 a;
    float s = d.kind == DEB_EMBER ? d.size * std::max(1.f - d.life / EMBER_LIFE, 0.05f) : d.size, sa = s;
    if (d.kind == DEB_WATER) {
        a = sp > 1e-3f ? d.v * (1.f / sp) : V3{0, 1, 0};
        sa = s * (1.f + 0.06f * sp);
    } else {
        V3 q = normalize(cross(d.spin, fabsf(d.spin.y) < 0.9f ? V3{0, 1, 0} : V3{1, 0, 0}));
        float ang = d.life * 9.f;
        a = normalize(q * cosf(ang) + cross(d.spin, q) * sinf(ang));
    }
    V3 t = normalize(cross(fabsf(a.y) < 0.99f ? V3{0, 1, 0} : V3{1, 0, 0}, a)), b = cross(a, t);
    V3 cols[3] = {t * s, a * sa, b * s};  // local x, y (along velocity / tumble axis), z
    for (int r = 0; r < 3; r++) {
        m[r][0] = (&cols[0].x)[r]; m[r][1] = (&cols[1].x)[r]; m[r][2] = (&cols[2].x)[r];
        m[r][3] = (&d.p.x)[r];
    }
}

void addDebris(uint8_t kind, V3 p, V3 v, float size, float delay) {
    if (spray.size() >= (size_t)MAX_SPRAY) return;
    Droplet d;
    d.p = p; d.v = v; d.size = size; d.life = 0.f; d.delay = delay; d.kind = kind; d.spin = rndDir();
    spray.push_back(d);
}

// The current slapping the upstream face of boulders that break the surface: now and then a small burst of
// droplets thrown up and back (only near the camera, where you can see them).
static void boulderSpray(float dt) {
    if (S.splash <= 0.f || S.flow <= 0.f) return;
    for (int i = (int)floorf((cam.pos.x - 50.f) / RIVER_ROCK_L); i <= (int)floorf((cam.pos.x + 50.f) / RIVER_ROCK_L); i++) {
        float x, y, z, sc;
        if (!riverRock(i, x, y, z, sc)) continue;
        RiverInfo ri = riverInfo(x, z);
        float top = y + 0.72f * sc;
        if (top < ri.level + 0.05f || (x - cam.pos.x) * (x - cam.pos.x) + (z - cam.pos.z) * (z - cam.pos.z) > 2500.f) continue;
        float cur[2];
        riverVelocity(x - 0.9f * sc, z, S.flow, cur);
        float U = sqrtf(cur[0] * cur[0] + cur[1] * cur[1]);
        if (rnd01() > dt * 1.5f * S.splash * U) continue;  // ~1.5 bursts/s per m/s of current
        V3 dir = V3{cur[0], 0, cur[1]} * (1.f / std::max(U, 1e-3f));
        float hh = (ri.level - y) / (0.72f * sc), hR = 0.95f * sc * sqrtf(std::clamp(1.f - hh * hh, 0.05f, 1.f));
        V3 face = V3{x, ri.level + 0.03f, z} - dir * hR;
        int n = 6 + (int)(rnd01() * 10.f * std::min(U, 2.f));
        for (int k = 0; k < n; k++) {
            V3 side = V3{-dir.z, 0, dir.x} * ((rnd01() - 0.5f) * 1.6f * hR);
            addDebris(DEB_WATER, face + side,
                      dir * (-0.6f * U * rnd01()) + side * (1.5f / std::max(hR, 0.3f)) + V3{0, (1.f + 1.5f * rnd01()) * std::min(U, 2.f), 0},
                      0.012f + 0.03f * rnd01() * rnd01());
        }
    }
}

// Ballistic debris with a little air drag. Water is gone when it falls back into the river or lands; clods and
// splinters bounce once off the ground, then settle and vanish.
void updateSpray(float dt) {
    boulderSpray(dt);
    for (size_t i = 0; i < spray.size();) {
        Droplet& d = spray[i];
        if (d.delay > 0.f) { d.delay -= dt; i++; continue; }
        d.life += dt;
        if (d.kind == DEB_EMBER) {  // rides the hot air up, then drifts with the wind as it cools and sinks
            float wd = S.windDirDeg * 3.14159265f / 180.f, ws = S.windStrength * S.windSpeed * 2.5f;
            V3 air{cosf(wd) * ws, 0.f, sinf(wd) * ws};
            d.v.y += 9.8f * (EMBER_LIFT * expf(-d.life / EMBER_COOL) - 1.f) * dt;
            d.v = d.v + (air - d.v) * std::min(EMBER_DRAG * dt, 1.f);
        } else {
            d.v.y -= 9.8f * dt;
            d.v = d.v * (1.f - std::min((d.kind == DEB_WATER ? 0.6f : d.kind == DEB_LEAF ? 3.f : 0.15f) * dt, 0.5f));
        }
        d.p = d.p + d.v * dt;
        float ground = heightFn(d.p.x, d.p.z);
        float wl = waterLevelAt(d.p.x, d.p.z);
        bool inWater = d.v.y < 0.f && d.p.y < wl;
        bool dead = d.life > (d.kind == DEB_WATER ? 4.f : d.kind == DEB_LEAF ? 10.f : d.kind == DEB_EMBER ? EMBER_LIFE : 6.f) ||
                    ((d.kind == DEB_WATER || d.kind == DEB_LEAF || d.kind == DEB_EMBER) && inWater);
        if (!dead && d.kind == DEB_EMBER && d.p.y < ground) {  // still glowing where it lands: maybe a new fire
            emberLanded(d.p);
            dead = true;
        }
        if (!dead && d.p.y < ground) {
            if (d.kind == DEB_WATER || dot(d.v, d.v) < 1.f) dead = true;
            else { d.p.y = ground; d.v = V3{d.v.x * 0.4f, -d.v.y * 0.25f, d.v.z * 0.4f}; }
        }
        if (!dead && d.kind != DEB_WATER && inWater && d.size > 0.05f && d.v.y < -2.f) {  // clods plopping in
            float v[3] = {d.v.x, d.v.y, d.v.z};
            waterImpact({d.p.x, wl, d.p.z}, d.size * 2.f, v, 0.5f);
            dead = true;
        }
        if (dead) { spray[i] = spray.back(); spray.pop_back(); continue; }
        i++;
    }
    while (!splashEvents.empty() && simTime - splashEvents.front().t > SPLASH_LIFE) splashEvents.erase(splashEvents.begin());
}

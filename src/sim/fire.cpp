#include "sim/fire.h"

#include <algorithm>
#include <cmath>

#include "app/frame_state.h"
#include "app/settings.h"
#include "audio/audio.h"
#include "core/limits.h"
#include "core/random.h"
#include "render/lights.h"
#include "scene/camera.h"
#include "sim/debris.h"
#include "world/terrain_query.h"

std::vector<Fire> fires;
static std::vector<float> requested;  // x z radius burn, lit on the next update

namespace {
constexpr float GROW_TIME = 6.f;          // s from a spark to full flames
constexpr float DIE_TIME = 25.f;          // s for the flames to die down once the fuel is gone
constexpr float RAIN_BURN = 7.f;          // full rain uses the fuel this much faster (it's being put out)
constexpr float FLAME_HEIGHT = 2.2f;      // flame height / radius at full heat
constexpr float SMOKE_HEIGHT = 14.f;      // m of smoke column above a 1 m fire at full heat
constexpr float FIRE_CANDELA = 250.f;     // luminous intensity of a 0.5 m campfire (cd)
constexpr float FIRE_LIGHT_COLOR[3] = {1.f, 0.48f, 0.16f};
constexpr float EMBERS_PER_S = 5.f;       // per metre of fire radius at full heat
constexpr float EMBER_IGNITE = 0.003f;    // chance a landing ember lights dry ground
constexpr float SPREAD_RADIUS = 0.35f, SPREAD_BURN = 20.f;  // the small fire it lights
constexpr float CRACKLE_GAIN = 0.7f;

float flameHeight(const Fire& f) { return f.radius * FLAME_HEIGHT * (0.4f + 0.6f * f.heat); }

float flicker(const Fire& f, float t) {  // a few incommensurate sines: no visible period
    return 0.78f + 0.12f * sinf(t * 9.1f + f.seed) + 0.07f * sinf(t * 15.7f + f.seed * 2.3f) + 0.05f * sinf(t * 23.3f + f.seed * 0.7f);
}

V3 windNearGround() {
    float wd = S.windDirDeg * 3.14159265f / 180.f, ws = S.windStrength * S.windSpeed * 2.5f;
    return V3{cosf(wd) * ws, 0.f, sinf(wd) * ws};
}
}  // namespace

void startFire(V3 at, float radius, float burnSeconds) {
    for (Fire& f : fires)  // feeding a fire that's already there
        if (dot(f.p - at, f.p - at) < (f.radius + radius) * (f.radius + radius) * 0.25f) {
            f.fuel += burnSeconds;
            f.radius = std::max(f.radius, radius);
            return;
        }
    if ((int)fires.size() >= MAX_FIRES) {  // full: the weakest goes
        auto weakest = std::min_element(fires.begin(), fires.end(), [](const Fire& a, const Fire& b) { return a.heat * a.radius < b.heat * b.radius; });
        if (weakest->sound != UINT32_MAX) audio::emitterStop(weakest->sound);
        fires.erase(weakest);
    }
    Fire f;
    f.p = at;
    f.radius = radius;
    f.fuel = burnSeconds;
    f.seed = rnd01() * 100.f;
    fires.push_back(f);
}

void requestFire(float x, float z, float radius, float burnSeconds) { requested.insert(requested.end(), {x, z, radius, burnSeconds}); }

void putOutFires() {
    for (Fire& f : fires) if (f.sound != UINT32_MAX) audio::emitterStop(f.sound);
    fires.clear();
}

bool fireInView(float radius, float burnSeconds, float reach) {
    V3 d = cam.fwd(), p = cam.pos;
    for (float t = 0.5f; t < reach; t += 0.25f + t * 0.02f) {
        V3 q = p + d * t;
        float g = heightFn(q.x, q.z);
        if (q.y <= g) {
            if (riverDry(q.x, q.z, 0.f) <= 0.f) return false;  // water doesn't burn
            startFire(V3{q.x, g, q.z}, radius, burnSeconds);
            return true;
        }
    }
    return false;
}

void emberLanded(V3 at) {
    if (S.rain > 0.05f || S.wetness > 0.2f || rnd01() > EMBER_IGNITE) return;
    if (riverDry(at.x, at.z, 0.f) <= 0.f) return;
    startFire(V3{at.x, heightFn(at.x, at.z), at.z}, SPREAD_RADIUS, SPREAD_BURN);
}

void updateFires(float dt) {
    for (size_t i = 0; i + 4 <= requested.size(); i += 4)
        startFire(V3{requested[i], heightFn(requested[i], requested[i + 1]), requested[i + 1]}, requested[i + 2], requested[i + 3]);
    requested.clear();
    float t = (float)simTime;
    V3 wind = windNearGround();
    for (size_t i = 0; i < fires.size();) {
        Fire& f = fires[i];
        f.fuel -= dt * (1.f + RAIN_BURN * S.rain);
        if (f.fuel > 0.f) f.heat = std::min(f.heat + dt / GROW_TIME, 1.f);
        else f.heat -= dt / DIE_TIME;
        if (f.heat <= 0.f && f.fuel <= 0.f) {
            if (f.sound != UINT32_MAX) audio::emitterStop(f.sound);
            fires.erase(fires.begin() + i);
            continue;
        }
        // embers: thrown up by the flames, carried off by the wind (debris.cpp flies them)
        f.emberDue += dt * EMBERS_PER_S * f.radius * f.heat * f.heat;
        float fh = flameHeight(f);
        for (; f.emberDue >= 1.f; f.emberDue -= 1.f) {
            float a = rnd01() * 6.2831853f, r = sqrtf(rnd01()) * f.radius * 0.7f;
            V3 at = f.p + V3{cosf(a) * r, fh * (0.3f + 0.5f * rnd01()), sinf(a) * r};
            V3 v = V3{(rnd01() - 0.5f) * 1.5f, 1.5f + 2.5f * rnd01(), (rnd01() - 0.5f) * 1.5f} + wind * 0.3f;
            addDebris(DEB_EMBER, at, v, 0.008f + 0.012f * rnd01());
        }
        // the crackle, louder with the flames
        float pos[3] = {f.p.x, f.p.y + 0.5f * fh, f.p.z};
        float gain = CRACKLE_GAIN * f.heat * sqrtf(f.radius / 0.5f) * flicker(f, t);
        if (f.sound == UINT32_MAX || !audio::emitterAlive(f.sound)) f.sound = audio::emitterPlay(audio::SND_FIRE, pos, gain, true, 0.9f + 0.2f * fmodf(f.seed, 1.f));
        else { audio::emitterSetPos(f.sound, pos); audio::emitterSetGain(f.sound, gain); }
        i++;
    }
}

void writeFires(UBO& u) {
    int n = std::min((int)fires.size(), MAX_FIRES);
    u.fireInfo[0] = (float)n;
    for (int i = 0; i < n; i++) {
        const Fire& f = fires[i];
        float fh = flameHeight(f), sh = SMOKE_HEIGHT * sqrtf(std::max(f.radius, 0.1f)) * sqrtf(f.heat);
        float a[4] = {f.p.x, f.p.y, f.p.z, f.radius}, b[4] = {f.heat, f.seed, fh, sh};
        std::copy(a, a + 4, u.fires[2 * i]);
        std::copy(b, b + 4, u.fires[2 * i + 1]);
    }
}

int writeFireLights(float* L, int room) {
    float t = (float)simTime;
    int n = 0;
    for (const Fire& f : fires) {
        if (n >= room) break;
        if (f.heat <= 0.f) continue;
        float I = FIRE_CANDELA * lightScale * (f.radius / 0.5f) * (f.radius / 0.5f) * f.heat * flicker(f, t);
        float fh = flameHeight(f);
        // the light sits in the flames, its sampling sphere about their size; w = 1: not drawn as an orb
        float v[8] = {f.p.x, f.p.y + 0.4f * fh, f.p.z, 0.4f * f.radius, FIRE_LIGHT_COLOR[0] * I, FIRE_LIGHT_COLOR[1] * I, FIRE_LIGHT_COLOR[2] * I, 1.f};
        std::copy(v, v + 8, L + n * 8);
        n++;
    }
    return n;
}

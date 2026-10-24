#include "scene/ambience.h"

#include <algorithm>
#include <cmath>
#include <cstdint>

#include "app/settings.h"
#include "audio/audio.h"
#include "core/math.h"
#include "core/random.h"
#include "scene/camera.h"
#include "scene/vegetation.h"
#include "world/terrain.h"
#include "world/terrain_query.h"

namespace {
constexpr float BIRD_RANGE = 45.f;                      // m: singers in the trees around
constexpr float BIRD_GAP_MIN = 1.5f, BIRD_GAP_MAX = 7.f;  // s between songs at a full chorus
constexpr float DAWN_HOUR = 6.f, DAWN_WIDTH = 1.5f;     // the dawn chorus
constexpr float OWL_GAP_MIN = 25.f, OWL_GAP_MAX = 80.f;
constexpr float OWL_RANGE_MIN = 30.f, OWL_RANGE_MAX = 70.f;
constexpr int CRICKETS = 3;
constexpr float CRICKET_RANGE_MIN = 6.f, CRICKET_RANGE_MAX = 22.f, CRICKET_MOVE = 18.f;
constexpr float RIVER_HEAR = 140.f;                     // m: the river is silent beyond this
constexpr float LEAF_RANGE = 25.f;
constexpr float CALM_WIND = 0.15f;                      // wind strength the leaves start to rustle at

float daylight() { return std::clamp((S.sunElev + 6.f) / 10.f, 0.f, 1.f); }
// warm enough for crickets: summer half of the year when the seasons run, else always
bool warmNight() { return !S.dayCycle || (S.dayOfYear > 120.f && S.dayOfYear < 280.f); }

const TreeInst* randomTreeNear(V3 at, float rmin, float rmax) {
    if (treeList.empty()) return nullptr;
    for (int tries = 0; tries < 24; tries++) {
        const TreeInst& t = treeList[(size_t)(rnd01() * (float)treeList.size()) % treeList.size()];
        float dx = t.p.x - at.x, dz = t.p.z - at.z, d2 = dx * dx + dz * dz;
        if (d2 >= rmin * rmin && d2 <= rmax * rmax) return &t;
    }
    return nullptr;
}
V3 canopyPoint(const TreeInst& t) {
    float h = treeTop[t.v] * t.scale * (0.5f + 0.35f * rnd01()), a = rnd01() * 6.2831853f, r = 1.5f * t.scale * rnd01();
    return t.p + V3{cosf(a) * r, h, sinf(a) * r};
}

void stop(uint32_t& e) { if (e != UINT32_MAX) audio::emitterStop(e); e = UINT32_MAX; }
// keep a looping emitter playing sound at pos with gain (stopped when silent)
void loop(uint32_t& e, uint32_t sound, V3 pos, float gain, float pitch = 1.f) {
    if (gain <= 0.005f) { stop(e); return; }
    if (e == UINT32_MAX || !audio::emitterAlive(e)) e = audio::emitterPlay(sound, &pos.x, gain, true, pitch);
    else { audio::emitterSetPos(e, &pos.x); audio::emitterSetGain(e, gain); }
}

void birds(float dt) {
    static float wait = 2.f;
    float day = daylight(), quiet = std::clamp(1.f - S.rain * 3.f, 0.f, 1.f) * std::clamp(1.3f - S.windStrength, 0.2f, 1.f);
    float dawn = S.dayCycle ? expf(-(S.timeOfDay - DAWN_HOUR) * (S.timeOfDay - DAWN_HOUR) / (2.f * DAWN_WIDTH * DAWN_WIDTH)) : 0.f;
    float chorus = day * quiet * (1.f + 1.5f * dawn);
    if ((wait -= dt * chorus) > 0.f || chorus <= 0.01f) return;
    wait = BIRD_GAP_MIN + (BIRD_GAP_MAX - BIRD_GAP_MIN) * rnd01();
    V3 at;
    if (const TreeInst* t = randomTreeNear(cam.pos, 3.f, BIRD_RANGE)) at = canopyPoint(*t);
    else if (rnd01() < 0.4f) {  // open country: a lark singing high overhead
        float a = rnd01() * 6.2831853f, r = 10.f + 25.f * rnd01();
        at = V3{cam.pos.x + cosf(a) * r, 0.f, cam.pos.z + sinf(a) * r};
        at.y = heightFn(at.x, at.z) + 15.f + 20.f * rnd01();
    } else return;
    audio::emitterPlay(audio::SND_BIRD + (uint32_t)(rnd01() * 3.99f), &at.x, 0.25f + 0.25f * rnd01(), false, 0.88f + 0.27f * rnd01());
}

void owl(float dt) {
    static float wait = 30.f;
    if (daylight() > 0.1f || S.rain > 0.2f || (wait -= dt) > 0.f) return;
    wait = OWL_GAP_MIN + (OWL_GAP_MAX - OWL_GAP_MIN) * rnd01();
    if (const TreeInst* t = randomTreeNear(cam.pos, OWL_RANGE_MIN, OWL_RANGE_MAX)) {
        V3 at = canopyPoint(*t);
        audio::emitterPlay(audio::SND_OWL, &at.x, 0.55f, false, 0.95f + 0.1f * rnd01());
    }
}

void crickets() {
    static uint32_t em[CRICKETS] = {UINT32_MAX, UINT32_MAX, UINT32_MAX};
    static V3 at[CRICKETS];
    float gain = S.ambience ? (1.f - daylight()) * std::clamp(1.f - S.rain * 4.f, 0.f, 1.f) * (warmNight() ? 1.f : 0.f) : 0.f;
    for (int i = 0; i < CRICKETS; i++) {
        V3 d = at[i] - cam.pos;
        if (em[i] == UINT32_MAX || d.x * d.x + d.z * d.z > CRICKET_MOVE * CRICKET_MOVE) {  // a cricket near where we are now
            stop(em[i]);
            float a = rnd01() * 6.2831853f, r = CRICKET_RANGE_MIN + (CRICKET_RANGE_MAX - CRICKET_RANGE_MIN) * rnd01();
            at[i] = V3{cam.pos.x + cosf(a) * r, 0.f, cam.pos.z + sinf(a) * r};
            at[i].y = heightFn(at[i].x, at[i].z) + 0.1f;
        }
        loop(em[i], audio::SND_CRICKETS, at[i], gain * (0.25f + 0.1f * i), 0.94f + 0.05f * i);
    }
}

void river() {
    static uint32_t em = UINT32_MAX;
    // nearest point of the river's centre line (it runs along x)
    float bx = cam.pos.x, bd = 1e30f;
    for (int k = -8; k <= 8; k++) {
        float x = cam.pos.x + k * 10.f, dz = riverCenter(x) - cam.pos.z, d2 = (x - cam.pos.x) * (x - cam.pos.x) + dz * dz;
        if (d2 < bd) { bd = d2; bx = x; }
    }
    float z = riverCenter(bx), dist = sqrtf(bd);
    V3 at{bx, riverInfo(bx, z).level, z};
    float gain = S.ambience ? std::clamp(1.f - dist / RIVER_HEAR, 0.f, 1.f) * (0.3f + 0.7f * std::min(S.flow, 2.f)) * 0.7f : 0.f;
    loop(em, audio::SND_RIVER, at, gain);
}

void leaves() {
    static uint32_t em = UINT32_MAX;
    const TreeInst* best = nullptr;
    float bd = LEAF_RANGE * LEAF_RANGE;
    for (const TreeInst& t : treeList) {
        float dx = t.p.x - cam.pos.x, dz = t.p.z - cam.pos.z, d2 = dx * dx + dz * dz;
        if (d2 < bd) { bd = d2; best = &t; }
    }
    float wind = std::max(S.windStrength * (0.5f + S.windSpeed / 4.f) - CALM_WIND, 0.f);
    float gain = best && S.ambience ? wind * (1.f - sqrtf(bd) / LEAF_RANGE) * 1.2f : 0.f;
    V3 at = best ? best->p + V3{0.f, treeTop[best->v] * best->scale * 0.6f, 0.f} : cam.pos;
    loop(em, audio::SND_LEAVES, at, gain);
}
}  // namespace

void updateAmbience(float dt) {
    if (S.ambience) { birds(dt); owl(dt); }
    crickets();  // (the loops fade out when it's switched off)
    river();
    leaves();
}

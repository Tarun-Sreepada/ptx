#include "sim/critters.h"

#include <algorithm>
#include <cmath>

#include "app/frame_state.h"
#include "app/settings.h"
#include "core/random.h"
#include "scene/camera.h"
#include "scene/vegetation.h"
#include "vegetation/trees.h"
#include "world/terrain_query.h"

namespace {
// birds
constexpr int FLOCKS = 3, FLOCK_BIRDS = 9;
constexpr float FLOCK_HEIGHT_MIN = 25.f, FLOCK_HEIGHT_MAX = 70.f;  // m above the ground
constexpr float FLOCK_RADIUS_MIN = 30.f, FLOCK_RADIUS_MAX = 90.f;  // m: the circles they wheel in
constexpr float FLOCK_RANGE = 180.f;                               // m: flocks keep within this of the camera
constexpr float BIRD_SPEED = 11.f, BIRD_SPEED_MIN = 6.f, BIRD_SPEED_MAX = 18.f;  // m/s
constexpr float BIRD_SPAN = 0.35f;                                 // m wingspan
constexpr float BIRD_FLAP_HZ = 4.5f, BIRD_FLAP_AMP = 0.9f;
constexpr float GLIDE_DIHEDRAL = 0.3f;                             // radians: wings held up a little in a glide
constexpr float ALARM_TIME = 8.f, ALARM_REACH = 200.f;
// butterflies
constexpr int BUTTERFLIES = 10;
constexpr float BUTTERFLY_RANGE = 18.f, BUTTERFLY_SPAN = 0.07f;
constexpr float BUTTERFLY_FLAP_HZ = 9.f, BUTTERFLY_FLAP_AMP = 1.25f, BUTTERFLY_SPEED = 1.4f;
// fireflies
constexpr int FIREFLIES = 70;
constexpr float FIREFLY_RANGE = 24.f, FIREFLY_SIZE = 0.012f, FIREFLY_SPEED = 0.5f;
constexpr float FIREFLY_FLASH = 0.35f;                             // s a flash lasts

float daylight() { return std::clamp((S.sunElev + 6.f) / 10.f, 0.f, 1.f); }
bool warmSeason() { return !S.dayCycle || (S.dayOfYear > 110.f && S.dayOfYear < 290.f); }

// a body frame: forward z (along v), wings x, up y banked by bank (radians), scaled
void frame(V3 p, V3 fwd, float bank, float scale, float m[3][4]) {
    V3 z = normalize(fwd), x = cross(V3{0, 1, 0}, z);
    x = dot(x, x) > 1e-8f ? normalize(x) : V3{1, 0, 0};  // straight up or down: any wing axis
    V3 y = cross(z, x);
    V3 xb = x * cosf(bank) + y * sinf(bank), yb = y * cosf(bank) - x * sinf(bank);
    V3 cols[3] = {xb * scale, yb * scale, z * scale};
    for (int r = 0; r < 3; r++) {
        m[r][0] = (&cols[0].x)[r]; m[r][1] = (&cols[1].x)[r]; m[r][2] = (&cols[2].x)[r];
        m[r][3] = (&p.x)[r];
    }
}

// last frame's transform of each critter (motion vectors)
struct Last { float m[3][4]; bool valid = false; };

struct Bird { V3 p, v, slot; float phase, stateT; bool flapping; float bank; Last last; };
struct Flock { V3 anchor; float ang = 0.f, radius = 50.f, height = 40.f, dir = 1.f, alarm = 0.f; bool live = false; Bird birds[FLOCK_BIRDS]; };
Flock flocks[FLOCKS];

struct Butterfly { V3 p, v, home; float phase, rest; int mesh; bool live = false; Last last; };
Butterfly butterflies[BUTTERFLIES];

struct Firefly { V3 p, v; float height, period, offset; bool live = false; Last last; };
Firefly fireflies[FIREFLIES];

std::vector<CritterDraw> draws;

// queue cd for drawing, with last frame's transform (or this one's when it just appeared)
void emit(CritterDraw& cd, Last& last) {
    const float(*prev)[4] = last.valid ? last.m : cd.m;
    std::copy(&prev[0][0], &prev[0][0] + 12, &cd.prev[0][0]);
    std::copy(&cd.m[0][0], &cd.m[0][0] + 12, &last.m[0][0]);
    last.valid = true;
    draws.push_back(cd);
}

V3 randomAround(V3 c, float rmin, float rmax) {
    float a = rnd01() * 6.2831853f, r = rmin + (rmax - rmin) * sqrtf(rnd01());
    return V3{c.x + cosf(a) * r, 0.f, c.z + sinf(a) * r};
}

void spawnFlock(Flock& f) {
    f.live = true;
    f.anchor = randomAround(cam.pos, 0.f, FLOCK_RANGE * 0.5f);
    f.radius = FLOCK_RADIUS_MIN + (FLOCK_RADIUS_MAX - FLOCK_RADIUS_MIN) * rnd01();
    f.height = FLOCK_HEIGHT_MIN + (FLOCK_HEIGHT_MAX - FLOCK_HEIGHT_MIN) * rnd01();
    f.dir = rnd01() < 0.5f ? 1.f : -1.f;
    f.ang = rnd01() * 6.2831853f;
    V3 c = f.anchor + V3{cosf(f.ang), 0.f, sinf(f.ang)} * f.radius;
    c.y = heightFn(c.x, c.z) + f.height;
    for (Bird& b : f.birds) {
        b.slot = V3{(rnd01() - 0.5f) * 10.f, (rnd01() - 0.5f) * 3.f, (rnd01() - 0.5f) * 10.f};
        b.p = c + b.slot;
        b.v = V3{-sinf(f.ang), 0.f, cosf(f.ang)} * (BIRD_SPEED * f.dir);
        b.phase = rnd01();
        b.stateT = rnd01() * 2.f;
        b.flapping = rnd01() < 0.5f;
        b.bank = 0.f;
        b.last.valid = false;
    }
}

void updateBirds(float dt) {
    bool out = daylight() > 0.3f && S.rain < 0.5f;
    for (Flock& f : flocks) {
        if (!out) { f.live = false; continue; }
        V3 d = f.anchor - cam.pos;
        if (!f.live || d.x * d.x + d.z * d.z > FLOCK_RANGE * FLOCK_RANGE * 1.5f) spawnFlock(f);
        float alarmK = f.alarm > 0.f ? 1.f + 0.6f * (f.alarm / ALARM_TIME) : 1.f;
        f.alarm = std::max(f.alarm - dt, 0.f);
        f.ang += f.dir * BIRD_SPEED * alarmK / f.radius * dt;
        V3 c = f.anchor + V3{cosf(f.ang), 0.f, sinf(f.ang)} * (f.radius * alarmK);
        c.y = heightFn(c.x, c.z) + f.height + (alarmK - 1.f) * 30.f;
        V3 heading = V3{-sinf(f.ang), 0.f, cosf(f.ang)} * f.dir;
        for (Bird& b : f.birds) {
            V3 target = c + b.slot + V3{sinf(simTime * 0.7f + b.phase * 9.f), 0.5f * sinf(simTime * 0.5f + b.phase * 5.f), cosf(simTime * 0.6f + b.phase * 7.f)} * 2.f;
            V3 want = heading * (BIRD_SPEED * alarmK) + (target - b.p) * 0.6f;
            V3 v0 = b.v;
            b.v = b.v + (want - b.v) * std::min(1.5f * dt, 1.f);
            float sp = std::max(vlen(b.v), 1e-3f);
            b.v = b.v * (std::clamp(sp, BIRD_SPEED_MIN, BIRD_SPEED_MAX * alarmK) / sp);
            b.p = b.p + b.v * dt;
            // bank into the turn: sideways acceleration over gravity
            V3 acc = (b.v - v0) * (1.f / std::max(dt, 1e-4f));
            V3 side = cross(V3{0, 1, 0}, b.v);
            side = dot(side, side) > 1e-8f ? normalize(side) : V3{1, 0, 0};
            b.bank += (std::clamp(atanf(dot(acc, side) / 9.8f), -0.8f, 0.8f) - b.bank) * std::min(3.f * dt, 1.f);
            // flap a while, glide a while; always flap climbing or alarmed
            if ((b.stateT -= dt) <= 0.f) { b.flapping = !b.flapping; b.stateT = b.flapping ? 0.8f + 1.5f * rnd01() : 1.f + 3.f * rnd01(); }
            bool flap = b.flapping || b.v.y > 1.f || f.alarm > 0.f;
            CritterDraw cd;
            cd.mesh = trees::PLANT_BIRD;
            frame(b.p, b.v, -b.bank, BIRD_SPAN, cd.m);
            cd.flapHz = flap ? BIRD_FLAP_HZ * (f.alarm > 0.f ? 1.3f : 1.f) : 0.f;
            cd.flapPhase = flap ? b.phase : 0.25f;  // gliding: wings held at the top of the stroke ...
            cd.flapAmp = flap ? BIRD_FLAP_AMP : GLIDE_DIHEDRAL;  // ... a little up
            emit(cd, b.last);
        }
    }
}

void updateButterflies(float dt) {
    bool out = daylight() > 0.4f && S.rain < 0.1f && warmSeason() && S.windStrength < 0.8f;
    for (Butterfly& b : butterflies) {
        if (!out) { b.live = false; continue; }
        V3 d = b.home - cam.pos;
        if (!b.live || d.x * d.x + d.z * d.z > BUTTERFLY_RANGE * BUTTERFLY_RANGE * 1.4f) {
            // a flower near the camera to flutter about, else any spot in the grass
            b.live = true;
            b.home = randomAround(cam.pos, 3.f, BUTTERFLY_RANGE);
            for (const PlantInst& pl : plantList) {
                if (pl.v != (uint32_t)trees::PLANT_FLOWERBUSH && pl.v != (uint32_t)trees::PLANT_ROSE && pl.v != (uint32_t)trees::PLANT_DANDELION) continue;
                float dx = pl.m[3] - cam.pos.x, dz = pl.m[11] - cam.pos.z;
                if (dx * dx + dz * dz < BUTTERFLY_RANGE * BUTTERFLY_RANGE && rnd01() < 0.2f) { b.home = V3{pl.m[3], 0.f, pl.m[11]}; break; }
            }
            b.home.y = heightFn(b.home.x, b.home.z);
            b.p = b.home + V3{0.f, 0.8f, 0.f};
            b.v = V3{0, 0, 0};
            b.phase = rnd01();
            b.rest = 0.f;
            b.mesh = rnd01() < 0.6f ? trees::PLANT_BUTTERFLY : trees::PLANT_BUTTERFLY2;
            b.last.valid = false;
        }
        // erratic flight: random jinks pulled back toward the flower, now and then a rest on it
        if (b.rest > 0.f) b.rest -= dt;
        else {
            b.v = b.v + V3{(rnd01() - 0.5f) * 14.f, (rnd01() - 0.45f) * 10.f, (rnd01() - 0.5f) * 14.f} * dt + (b.home + V3{0, 0.7f, 0} - b.p) * (0.8f * dt);
            float sp = vlen(b.v);
            if (sp > BUTTERFLY_SPEED) b.v = b.v * (BUTTERFLY_SPEED / sp);
            b.p = b.p + b.v * dt;
            float g = heightFn(b.p.x, b.p.z);
            if (b.p.y < g + 0.15f) { b.p.y = g + 0.15f; b.v.y = fabsf(b.v.y); }
            if (rnd01() < 0.05f * dt) b.rest = 1.f + 3.f * rnd01();
        }
        CritterDraw cd;
        cd.mesh = b.mesh;
        V3 fwd = dot(b.v, b.v) > 1e-4f ? b.v : V3{1, 0, 0};
        fwd.y *= 0.3f;
        frame(b.p, fwd, 0.f, BUTTERFLY_SPAN, cd.m);
        cd.flapHz = b.rest > 0.f ? 0.4f : BUTTERFLY_FLAP_HZ;  // resting: wings slowly open and close
        cd.flapPhase = b.phase;
        cd.flapAmp = BUTTERFLY_FLAP_AMP;
        emit(cd, b.last);
    }
}

void updateFireflies(float dt) {
    float night = 1.f - std::clamp((S.sunElev + 4.f) / 6.f, 0.f, 1.f);
    bool out = night > 0.2f && S.rain < 0.05f && warmSeason();
    int want = out ? (int)(FIREFLIES * night) : 0;
    for (int i = 0; i < FIREFLIES; i++) {
        Firefly& f = fireflies[i];
        if (i >= want) { f.live = false; continue; }
        V3 d = f.p - cam.pos;
        if (!f.live || d.x * d.x + d.z * d.z > FIREFLY_RANGE * FIREFLY_RANGE * 1.3f) {
            f.p = randomAround(cam.pos, 2.f, FIREFLY_RANGE);
            if (riverDry(f.p.x, f.p.z, 0.f) <= 0.f) continue;  // not over the water: try again next frame
            f.live = true;
            f.height = 0.3f + 2.2f * rnd01() * rnd01();
            f.p.y = heightFn(f.p.x, f.p.z) + f.height;
            f.v = V3{0, 0, 0};
            f.period = 2.f + 3.f * rnd01();
            f.offset = rnd01() * f.period;
            f.last.valid = false;
        }
        f.v = f.v + V3{(rnd01() - 0.5f) * 2.f, (rnd01() - 0.5f) * 1.f, (rnd01() - 0.5f) * 2.f} * dt;
        float sp = vlen(f.v);
        if (sp > FIREFLY_SPEED) f.v = f.v * (FIREFLY_SPEED / sp);
        f.p = f.p + f.v * dt;
        float g = heightFn(f.p.x, f.p.z);
        f.p.y += (g + f.height - f.p.y) * std::min(dt, 1.f);  // keeps to its height over the ground
        // the flash: a quick swell and fade every period
        float ph = fmodf((float)simTime + f.offset, f.period);
        float glow = ph < FIREFLY_FLASH ? sinf(3.14159265f * ph / FIREFLY_FLASH) : 0.f;
        CritterDraw cd;
        cd.mesh = trees::PLANT_FIREFLY;
        V3 fwd = dot(f.v, f.v) > 1e-6f ? f.v : V3{0, 0, 1};
        frame(f.p, fwd, 0.f, FIREFLY_SIZE * glow, cd.m);
        emit(cd, f.last);
    }
}
}  // namespace

void updateCritters(float dt) {
    draws.clear();
    if (!S.critters) return;
    updateBirds(dt);
    updateButterflies(dt);
    updateFireflies(dt);
}

void critterAlarm(V3 at, float loudness) {
    for (Flock& f : flocks) {
        if (!f.live) continue;
        V3 c = f.anchor + V3{cosf(f.ang), 0.f, sinf(f.ang)} * f.radius, d = c - at;
        if (d.x * d.x + d.z * d.z < ALARM_REACH * ALARM_REACH * std::min(loudness, 2.f)) f.alarm = ALARM_TIME;
    }
}

const std::vector<CritterDraw>& critterDraws() { return draws; }

#include "sim/wildlife.h"

#include <algorithm>
#include <cmath>
#include <unordered_map>

#include "app/settings.h"
#include "characters/character.h"
#include "core/random.h"
#include "scene/camera.h"
#include "scene/characters.h"
#include "sim/water.h"
#include "world/terrain_query.h"

namespace {
constexpr int MAX_FOXES = 3;
constexpr const char* FOX_ASSET = "characters/Fox";  // Khronos sample fox: Survey, Walk, Run; faces -z
constexpr float FOX_HEIGHT = 0.55f;                  // m
constexpr float SPAWN_MIN = 35.f, SPAWN_MAX = 70.f;  // m from the camera
constexpr float DESPAWN = 120.f;                     // m: let go
constexpr float FLEE_DIST = 13.f;                    // m: too close, it runs
constexpr float WALK = 0.9f, RUN = 5.5f;             // m/s
constexpr float SPAWN_GAP = 4.f;                     // s between spawns

enum State { WANDER, PAUSE, FLEE };
struct Fox { State state = WANDER; float timer = 0.f, turnCool = 0.f, fleeOffset = 0.f; int clip = -1; };
std::unordered_map<const character::Character*, Fox> foxes;

bool dryLand(float x, float z) { return riverDry(x, z, 1.f) > 0.f && waterLevelAt(x, z) < heightFn(x, z); }

void setClip(character::Character& c, Fox& f, const char* name) {
    int k = character::findClip(*c.data, name);
    if (k >= 0 && k != f.clip) { character::play(c, k, 0.25f); f.clip = k; }
}
}  // namespace

void updateWildlife(float dt) {
    float day = std::clamp((S.sunElev + 4.f) / 8.f, 0.f, 1.f);
    bool out = S.wildlife && day > 0.1f && S.rain < 0.5f;
    int count = 0;
    for (character::Character* c : character::all()) {
        if (!c->wild) continue;
        count++;
        Fox& f = foxes[c];
        float dx = c->pos[0] - cam.pos.x, dz = c->pos[2] - cam.pos.z, d = sqrtf(dx * dx + dz * dz);
        if (d > DESPAWN || (!out && (d > SPAWN_MIN || !S.wildlife))) {  // far behind, night fell out of sight, or switched off
            foxes.erase(c);
            releaseCharacter(c);
            break;  // the list changed
        }
        if (c->mode != character::ANIMATED) { c->walkSpeed = 0.f; continue; }  // hit: lies limp, gets up on its own
        f.timer -= dt;
        bool near = d < FLEE_DIST;
        if (near && f.state != FLEE) { f.state = FLEE; f.timer = 4.f + 3.f * rnd01(); f.fleeOffset = (rnd01() - 0.5f) * 0.8f; }
        f.turnCool -= dt;
        float ahead = 3.f;
        float ax = c->pos[0] + sinf(c->yaw) * c->forward * ahead, az = c->pos[2] + cosf(c->yaw) * c->forward * ahead;
        if (f.turnCool <= 0.f && !dryLand(ax, az)) {  // water ahead: turn back (then give the turn time)
            c->yaw += 3.14159265f * (0.6f + 0.4f * rnd01());
            f.turnCool = 1.f;
        }
        switch (f.state) {
        case FLEE:
            if (near && f.turnCool <= 0.f) c->yaw = atan2f(dx * c->forward, dz * c->forward) + f.fleeOffset;  // it moves away from you
            c->walkSpeed = RUN;
            setClip(*c, f, "Run");
            if (f.timer <= 0.f && !near) { f.state = WANDER; f.timer = 6.f + 10.f * rnd01(); }
            break;
        case PAUSE:
            c->walkSpeed = 0.f;
            setClip(*c, f, "Survey");
            if (f.timer <= 0.f) { f.state = WANDER; f.timer = 6.f + 12.f * rnd01(); }
            break;
        default:
            c->walkSpeed = WALK;
            setClip(*c, f, "Walk");
            if (f.timer <= 0.f) { f.state = PAUSE; f.timer = 3.f + 5.f * rnd01(); }
            break;
        }
    }
    // a new fox out of sight (behind you), now and then
    static float wait = 2.f;
    if ((wait -= dt) > 0.f || !out || count >= MAX_FOXES || !charSpawns.empty()) return;
    wait = SPAWN_GAP;
    V3 back = cam.fwd() * -1.f;
    float a = atan2f(back.x, back.z) + (rnd01() - 0.5f) * 2.f, r = SPAWN_MIN + (SPAWN_MAX - SPAWN_MIN) * rnd01();
    float x = cam.pos.x + sinf(a) * r, z = cam.pos.z + cosf(a) * r;
    if (!dryLand(x, z)) return;
    CharSpawn sp{FOX_ASSET, "Walk", {x, 0.f, z}, rnd01() * 6.2831853f, FOX_HEIGHT, WALK};
    sp.wild = true;
    sp.forward = -1.f;
    charSpawns.push_back(sp);
}

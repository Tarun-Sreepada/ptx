#include "sim/trample.h"

#include <algorithm>
#include <cmath>
#include <vector>

#include "app/frame_state.h"
#include "app/settings.h"
#include "characters/character.h"
#include "destruction/thrown.h"
#include "scene/camera.h"
#include "sim/water.h"
#include "world/terrain_query.h"

namespace {
constexpr int MAX_STEPS = 48;               // remembered footsteps (trail + footprints)
constexpr float TRAMPLE_RECOVER = 70.f;     // s for trampled grass to stand up again
constexpr float FOOTPRINT_LIFE = 240.f;     // s a footprint lasts
constexpr float FOOT_LENGTH = 0.14f;        // m, half the length of a footprint
constexpr float FOOT_SIDE = 0.11f;          // m: each foot this far beside the line of walking
constexpr float BODY_PUSH_R = 0.38f;        // m: what a standing body pushes aside
constexpr float TRAIL_PUSH_R = 0.3f;        // m: a footstep's flattened patch
constexpr float PUSH_RANGE = 40.f;          // m from the camera
constexpr float ROCK_PUSH_MAX_SPEED = 1.f;  // m/s: a rock lying in the grass (moving ones are a blur anyway)

struct Step { V3 p; float yaw; double t; };
std::vector<Step> steps;

struct Pusher { V3 p; float r, flatten, d2; };
}  // namespace

void footstep(V3 p, float yaw, bool left) {
    V3 side{cosf(yaw), 0.f, -sinf(yaw)};  // to the right of the heading
    p = p + side * (left ? -FOOT_SIDE : FOOT_SIDE);
    p.y = heightFn(p.x, p.z);
    if (waterLevelAt(p.x, p.z) > p.y) return;  // under water: no print
    if ((int)steps.size() >= MAX_STEPS) steps.erase(steps.begin());
    steps.push_back({p, yaw, simTime});
}

void writeTrample(UBO& u) {
    // footprints after the impact decals (decal kind 3: n holds the heading, the surface is the ground)
    uint32_t nd = u.lightInfo[3];
    for (auto it = steps.rbegin(); it != steps.rend() && nd < (uint32_t)MAX_DECALS; ++it) {
        float age = (float)(simTime - it->t);
        if (age > FOOTPRINT_LIFE || !S.decals) continue;
        float a[8] = {it->p.x, it->p.y, it->p.z, FOOT_LENGTH, sinf(it->yaw), 0.f, cosf(it->yaw), 3.f + std::min(age, 999.f) / 1000.f};
        std::copy(a, a + 8, &u.decals[2 * nd][0]);
        nd++;
    }
    u.lightInfo[3] = nd;
    // grass pushers: bodies standing in it now, then the trail, nearest the camera first
    std::vector<Pusher> ps;
    auto add = [&](V3 p, float r, float flatten) {
        V3 d = p - cam.pos;
        float d2 = d.x * d.x + d.z * d.z;
        if (d2 < PUSH_RANGE * PUSH_RANGE) ps.push_back({p, r, flatten, d2});
    };
    if (walkMode) add(playerFeet, BODY_PUSH_R, 0.6f);
    for (character::Character* c : character::all()) add(V3{c->pos[0], c->pos[1], c->pos[2]}, BODY_PUSH_R * c->scale, 0.6f);
    for (const Thrown& t : thrown) {
        V3 p{t.xf.m[0][3], t.xf.m[1][3], t.xf.m[2][3]};
        if (dot(V3{t.v[0], t.v[1], t.v[2]}, V3{t.v[0], t.v[1], t.v[2]}) < ROCK_PUSH_MAX_SPEED * ROCK_PUSH_MAX_SPEED && p.y - t.radius < heightFn(p.x, p.z) + 0.3f)
            add(p, t.radius * 1.2f, 1.f);
    }
    size_t live = ps.size();
    for (const Step& s : steps) {
        float k = 1.f - (float)(simTime - s.t) / TRAMPLE_RECOVER;
        if (k > 0.f) add(s.p, TRAIL_PUSH_R, 0.8f * k * k);
    }
    std::sort(ps.begin() + (ptrdiff_t)live, ps.end(), [](const Pusher& a, const Pusher& b) { return a.d2 < b.d2; });
    int n = std::min((int)ps.size(), MAX_PUSHERS);
    for (int i = 0; i < n; i++) {
        float a[8] = {ps[i].p.x, ps[i].p.y, ps[i].p.z, ps[i].r, ps[i].flatten, 0.f, 0.f, 0.f};
        std::copy(a, a + 8, &u.pushers[2 * i][0]);
    }
    u.pushInfo[0] = (float)n;
}

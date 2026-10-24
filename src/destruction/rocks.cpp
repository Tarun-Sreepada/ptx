#include "destruction/rocks.h"
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include "physics/physics.h"
#include "physics/fracture.h"
#include "core/limits.h"
#include "core/math.h"
#include "core/random.h"
#include "destruction/rock_fracture.h"
#include "destruction/thrown.h"
#include "scene/rocks.h"
#include "scene/vegetation.h"

std::unordered_map<uint64_t, RockBody> rockBodies;

std::unordered_set<uint64_t> brokenRocks;

uint32_t numBreakableRocks = 0;  // the first rocks in rockXf: grid + buried + hand-placed (not river rocks)

// a ground rock hit hard enough: off the placement, into the physics as a thrown rock carrying the blow, cracked
bool knockRockLoose(V3 p, float reach, V3 J, float E) {
    uint64_t best = 0;
    float bd = 1e30f;
    for (auto& [k, rb] : rockBodies) {
        float dx = rb.xf.m[0][3] - p.x, dy = rb.xf.m[1][3] - p.y, dz = rb.xf.m[2][3] - p.z, d2 = dx * dx + dy * dy + dz * dz;
        float lim = reach + 1.1f * rb.sc;
        if (d2 < lim * lim && d2 < bd) { bd = d2; best = k; }
    }
    if (bd == 1e30f) return false;
    RockBody rb = rockBodies[best];
    float mass = 2600.f * 2.6f * rb.sc * rb.sc * rb.sc;
    if (E < 0.5f * fractureThreshold(0.85f * rb.sc)) return false;  // it holds
    physics::removeBodies(&rb.body, 1);
    rockBodies.erase(best);
    brokenRocks.insert(best);
    placeDirty = true;
    if (thrown.size() >= (size_t)MAX_THROWN) dropThrown(0);
    V3 pos{rb.xf.m[0][3], rb.xf.m[1][3] + 0.05f, rb.xf.m[2][3]};
    V3 v = J * (0.6f / mass);
    float vl = sqrtf(dot(v, v));
    if (vl > 25.f) v = v * (25.f / vl);
    float w[3] = {rnd01() * 4.f - 2.f, rnd01() * 4.f - 2.f, rnd01() * 4.f - 2.f};
    Thrown t;
    t.sc = rb.sc;
    t.radius = 0.85f * rb.sc;
    t.rockVar = rb.var;
    t.body = physics::addDynamic(rockHull[rb.var], rb.sc, &pos.x, &v.x, w, 2600.f);
    t.xf = rb.xf;
    t.poly = std::make_shared<fracture::Poly>(rockPoly[rb.var]);
    for (auto& fc : t.poly->faces) for (auto& q : fc.pts) q = {q.x * rb.sc, q.y * rb.sc, q.z * rb.sc};
    memcpy(t.v, &v.x, 12);
    t.cooldown = 0.2f;
    thrown.push_back(t);
    V3 dir = dot(J, J) > 1e-6f ? normalize(J) : V3{0, -1, 0};
    if (pendingFractures.size() < 32) pendingFractures.push_back({t.body, p, dir, E * 0.7f});
    return true;
}

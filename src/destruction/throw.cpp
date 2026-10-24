#include "destruction/throw.h"
#include <string>
#include <memory>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include "physics/physics.h"
#include "physics/fracture.h"
#include "world/terrain.h"
#include "app/frame_state.h"
#include "app/settings.h"
#include "core/limits.h"
#include "core/math.h"
#include "destruction/rock_fracture.h"
#include "destruction/thrown.h"
#include "scene/camera.h"
#include "scene/colliders.h"
#include "scene/rocks.h"
#include "tools/movelog.h"

void throwRock() {
    moveLogEvent("throw", std::to_string(S.throwSize) + " " + std::to_string(S.throwSpeed));
    throwRockImpl();
}

void throwRockImpl() {
    ensureRockShape();
    uint32_t var = (uint32_t)(hash2((int)frameIndex, 7) * ROCK_VARIANTS) % ROCK_VARIANTS;
    if (!rockHull[var]) return;
    if (thrown.size() >= (size_t)MAX_THROWN) dropThrown(0);  // recycle the oldest
    float sc = S.throwSize;
    V3 f = cam.fwd(), p = cam.pos + f * (0.6f + sc * 1.2f) - V3{0, 0.2f, 0};
    V3 v = f * S.throwSpeed;
    float w[3] = {hash2((int)frameIndex, 1) * 8.f - 4.f, hash2((int)frameIndex, 2) * 8.f - 4.f, hash2((int)frameIndex, 3) * 8.f - 4.f};
    Thrown t;
    t.sc = sc;
    t.radius = 0.85f * sc;
    t.rockVar = var;
    t.body = physics::addDynamic(rockHull[var], sc, &p.x, &v.x, w, 2600.f);
    t.xf = {{{sc, 0, 0, p.x}, {0, sc, 0, p.y}, {0, 0, sc, p.z}}};
    t.poly = std::make_shared<fracture::Poly>(rockPoly[var]);
    for (auto& fc : t.poly->faces) for (auto& q : fc.pts) q = {q.x * sc, q.y * sc, q.z * sc};
    memcpy(t.v, &v.x, 12);
    thrown.push_back(t);
}

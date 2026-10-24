#include "scene/colliders.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <utility>
#include "core/prof.h"
#include "physics/physics.h"
#include "physics/fracture.h"
#include "world/world.h"
#include "core/math.h"
#include "destruction/rock_fracture.h"
#include "destruction/rocks.h"
#include "destruction/thrown.h"
#include "destruction/tree_damage.h"
#include "scene/camera.h"
#include "scene/rocks.h"
#include "scene/scene_gpu.h"
#include "scene/vegetation.h"

// Trunk colliders only where the player can reach them (static bodies within ~40 m, dropped past 60 m).
void syncTreeBodies() {
    // colliders around the player (walk mode) and around every thrown rock, so rocks hit trunks
    std::vector<V3> around;
    if (walkMode) around.push_back(cam.pos);
    for (const Thrown& t : thrown) around.push_back({t.xf.m[0][3], t.xf.m[1][3], t.xf.m[2][3]});
    if (around.empty() && treeBodies.empty()) return;
    PROF_FUNC();
    std::vector<void*> shapes;
    std::vector<Xf> xfs;
    std::vector<uint64_t> keys;
    std::unordered_map<uint64_t, bool> nearNow;
    for (const TreeInst& t : treeList) {
        float d2 = 1e30f;
        for (const V3& a : around) d2 = std::min(d2, (t.p.x - a.x) * (t.p.x - a.x) + (t.p.z - a.z) * (t.p.z - a.z));
        if (d2 > 60.f * 60.f) continue;
        uint64_t key = treeKey(t.ix, t.iz);
        nearNow[key] = true;
        if (d2 > 40.f * 40.f || treeBodies.count(key)) continue;
        float c = cosf(t.yaw) * t.scale, sn = sinf(t.yaw) * t.scale;
        xfs.push_back({{{c, 0, sn, t.p.x}, {0, t.scale, 0, t.p.y}, {-sn, 0, c, t.p.z}}});
        shapes.push_back(trunkShape[t.v]);
        keys.push_back(key);
    }
    std::vector<uint32_t> gone;
    for (auto it = treeBodies.begin(); it != treeBodies.end();)
        if (!nearNow.count(it->first)) { gone.push_back(it->second); it = treeBodies.erase(it); } else ++it;
    if (!gone.empty()) physics::removeBodies(gone.data(), gone.size());
    if (shapes.empty()) return;
    std::vector<uint32_t> ids(shapes.size());
    physics::addStatic(shapes.data(), xfs.data(), shapes.size(), ids.data());
    for (size_t i = 0; i < keys.size(); i++) treeBodies[keys[i]] = ids[i];
}

// Hard contact on land: the kinetic energy the last step took away, E = m/2 (|v0|^2 - |v1|^2), goes into what was
// hit. A tree snaps past a threshold (the rock then mostly carries on through), the ground gets a crater, and
// anything else (a building) throws rubble. The rock itself breaks if the blow's normal part is hard enough.
void ensureRockShape() {
    for (int v = 0; v < ROCK_VARIANTS; v++) {
        const float* pts = rockPts.data() + (size_t)v * rockVerts * 6;
        if (!rockHull[v]) rockHull[v] = physics::convexShape(pts, 6, rockVerts);
        if (rockPoly[v].faces.empty()) rockPoly[v] = fracture::fromPoints(pts, 6, rockVerts);
    }
}

void syncRockBodies() {
    if (thrown.empty() && rockBodies.empty()) return;
    PROF_FUNC();
    ensureRockShape();
    std::vector<V3> around;
    for (const Thrown& t : thrown) around.push_back({t.xf.m[0][3], t.xf.m[1][3], t.xf.m[2][3]});
    const float* xf = (const float*)rockXf.map;
    std::unordered_set<uint64_t> near;
    std::vector<void*> shapes;
    std::vector<Xf> xfs;
    std::vector<std::pair<uint64_t, float>> keys;
    std::vector<uint32_t> vars;
    for (uint32_t i = 0; i < numBreakableRocks && !around.empty(); i++) {
        const float* m = xf + i * 16;
        uint32_t var;
        memcpy(&var, m + 12, 4);
        var /= std::max(rockVerts, 1u);
        if (!rockHull[var]) continue;
        float x = m[3], z = m[11], d2 = 1e30f;
        for (const V3& a : around) d2 = std::min(d2, (x - a.x) * (x - a.x) + (z - a.z) * (z - a.z));
        if (d2 > 35.f * 35.f) continue;
        uint64_t k = rockKey(x, z);
        near.insert(k);
        if (d2 > 25.f * 25.f || rockBodies.count(k)) continue;
        Xf x3;
        memcpy(x3.m, m, 48);
        float sc = sqrtf(m[0] * m[0] + m[4] * m[4] + m[8] * m[8]);
        xfs.push_back(x3);
        shapes.push_back(rockHull[var]);
        keys.push_back({k, sc});
        vars.push_back(var);
    }
    std::vector<uint32_t> gone;
    for (auto it = rockBodies.begin(); it != rockBodies.end();)
        if (!near.count(it->first)) { gone.push_back(it->second.body); it = rockBodies.erase(it); } else ++it;
    if (!gone.empty()) physics::removeBodies(gone.data(), gone.size());
    if (shapes.empty()) return;
    std::vector<uint32_t> ids(shapes.size());
    physics::addStatic(shapes.data(), xfs.data(), shapes.size(), ids.data());
    for (size_t i = 0; i < keys.size(); i++) rockBodies[keys[i].first] = {ids[i], xfs[i], keys[i].second, vars[i]};
}

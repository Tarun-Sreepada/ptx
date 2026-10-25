#include "destruction/impacts.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <utility>
#include "audio/audio.h"
#include "physics/physics.h"
#include "vegetation/trees.h"
#include "world/terrain.h"
#include "core/math.h"
#include "core/random.h"
#include "destruction/buildings.h"
#include "destruction/craters.h"
#include "destruction/rock_fracture.h"
#include "destruction/rocks.h"
#include "destruction/thrown.h"
#include "destruction/tree_break.h"
#include "destruction/wood.h"
#include "scene/vegetation.h"
#include "sim/critters.h"
#include "sim/debris.h"
#include "world/terrain_query.h"

namespace {
constexpr float IMPACT_SOUND_MIN_J = 2.f;  // quieter contacts (rolling, settling) make no sound
constexpr float BIRD_SCARE_J = 2e4f;       // impacts this loud scare the birds
enum { SOUND_STONE, SOUND_WOOD, SOUND_DIRT, SOUND_METAL };
}  // namespace

void landImpact(Thrown& t, const float v0[3], const float v1[3], float mass, float dt) {
    float s0 = v0[0] * v0[0] + v0[1] * v0[1] + v0[2] * v0[2], s1 = v1[0] * v1[0] + v1[1] * v1[1] + v1[2] * v1[2];
    float E = 0.5f * mass * std::max(s0 - s1, 0.f);
    // contact normal ~ the direction of the impulse (velocity change minus gravity's share)
    V3 dv{v1[0] - v0[0], v1[1] - v0[1] + 9.8f * dt, v1[2] - v0[2]};
    V3 n = dot(dv, dv) > 1e-6f ? normalize(dv) : V3{0, 1, 0};
    float vn = std::max(-(v0[0] * n.x + v0[1] * n.y + v0[2] * n.z), 0.f);
    float En = 0.5f * mass * vn * vn;
    V3 p{t.xf.m[0][3], t.xf.m[1][3], t.xf.m[2][3]};
    if (t.poly && En > fractureThreshold(t.radius) && pendingFractures.size() < 32)
        pendingFractures.push_back({t.body, p - n * (t.radius * 0.8f), n * -1.f, En});
    // where it touched: a rock's surface toward the blow, a wood piece's furthest point that way
    V3 pc = p - n * (t.radius * 0.8f);
    WoodFrame wf;
    V3 wl{0, 0, 0};
    if (t.woodPart) {
        wf = thrownFrame(t);
        V3 ln = woodRotT(wf, n);
        trees::woodSupport(*t.woodPart, &ln.x, &wl.x);
        pc = woodToWorld(wf, wl);
    }
    // the sound of it: what hit what (wood rings, rock on rock clacks, on soil it thuds)
    auto sound = [&](int kind) {
        if (E <= IMPACT_SOUND_MIN_J) return;
        audio::impact(kind, &pc.x, E, 2.f * t.radius);
        if (E > BIRD_SCARE_J) critterAlarm(pc, std::min(E / BIRD_SCARE_J, 4.f) * 0.5f);  // a bang: the birds take off
    };
    bool onGround = p.y - t.radius * 0.7f < heightFn(p.x, p.z) + 0.3f;
    // a procedural tree? Its limbs take the blow as beams (any impact, the stress decides)
    // (not a loose piece of tree: lying against its stump or in a crown it "struck" it every contact, each cut
    // spawning more pieces that struck again; its touchdowns break it instead, below)
    if (E > 5.f && !treeList.empty() && !t.woodPart) {
        V3 vel{v0[0], v0[1], v0[2]};
        if (strikeTrees(pc, vel, mass, t.radius * 0.3f + 0.25f)) {
            sound(SOUND_WOOD);
            t.cooldown = 0.15f;
            for (auto& q : pendingWood)  // broke through: the impactor keeps what the snap didn't take
                if (q.body == UINT32_MAX && !q.hit.cuts.empty() && dot(q.at - pc, q.at - pc) < 1e-6f) {
                    V3 nv = vel - q.J * (1.f / mass);
                    physics::setVelocity(t.body, &nv.x);
                }
            return;
        }
    }
    // a loose piece of tree landing: its own joints may give
    if (t.woodPart && pendingWood.size() < 16 && dot(dv, dv) < 50.f * 50.f) {  // (bigger: a depenetration kick, not a landing)
        V3 Jw = dv * mass, Jl = woodRotT(wf, Jw);
        // the spin before the contact took it (thrown.cpp keeps last frame's)
        V3 vl = woodRotT(wf, V3{v0[0], v0[1], v0[2]}), wl2 = woodRotT(wf, V3{t.w0[0], t.w0[1], t.w0[2]}), nl = woodRotT(wf, n);
        trees::WoodHit h = trees::woodLanding(*t.woodPart, wf.s, &wl.x, &Jl.x, &vl.x, &wl2.x, &nl.x);
        if (treeLog) fprintf(stderr, "wood landed: %.0f kg, dv %.1f m/s: stress %.2f, %zu cut, %d leaves\n", mass, sqrtf(dot(dv, dv)), h.ratio, h.cuts.size(), h.lost);
        bool queued = false;
        for (auto& q : pendingWood) queued |= q.body == t.body;
        if (!queued && (!h.cuts.empty() || h.lost >= 8)) {
            t.cooldown = 0.15f;
            pendingWood.push_back({TreeInst{{0, 0, 0}, 0, 1, (uint32_t)t.woodPart->v, (int)(t.body & 0xffff), (int)(t.body >> 16)}, t.body, std::move(h), pc, Jw});
        }
    }
    if (E < 3e3f) {
        sound(t.wood ? SOUND_WOOD : (onGround ? SOUND_DIRT : SOUND_STONE));
        return;
    }
    t.cooldown = 0.15f;
    float r = t.radius;
    // another rock (a scattered ground rock) took the blow?
    if (!rockBodies.empty()) {
        V3 J = V3{v0[0] - v1[0], v0[1] - v1[1], v0[2] - v1[2]} * mass;
        if (knockRockLoose(p, r, J, E)) { sound(SOUND_STONE); return; }
    }
    // an imported tree's trunk within reach of the rock?
    const TreeInst* best = nullptr;
    float bd = 1e9f;
    for (const TreeInst& tr : treeList) {
        if (tr.v < (uint32_t)trees::VARIANTS) continue;  // procedural trees: strikeTrees above
        float dx = tr.p.x - p.x, dz = tr.p.z - p.z, d2 = dx * dx + dz * dz;
        if (d2 < bd && d2 < (r + 0.9f) * (r + 0.9f) && p.y < tr.p.y + 6.f * tr.scale) { bd = d2; best = &tr; }
    }
    if (best) {
        sound(SOUND_WOOD);
        // ~6 kJ snaps a typical (scale 1, ~25 cm) trunk: a default throw (0.4 m rock, ~430 kg at 12 m/s, ~31 kJ)
        // breaks one even after a glancing hit, a fist-sized stone doesn't
        float Ebreak = 6e3f * best->scale * best->scale * best->scale;
        if (E > Ebreak) {
            V3 J = V3{v0[0] - v1[0], v0[1] - v1[1], v0[2] - v1[2]} * mass;
            TreeInst tr = *best;
            knockTree(tr, J, p, E / Ebreak);
            float keep = std::sqrt(std::max(1.f - Ebreak / E, 0.f));  // the rock keeps what breaking didn't use
            float nv[3] = {v0[0] * keep, v0[1] * keep, v0[2] * keep};
            physics::setVelocity(t.body, nv);
        }
        return;
    }
    float ground = heightFn(p.x, p.z);
    if (p.y - r * 0.7f < ground + 0.3f) {
        sound(t.wood ? SOUND_WOOD : SOUND_DIRT);
        if (riverDry(p.x, p.z, 0.f) > 0.f && !t.wood) groundImpact({p.x, ground, p.z}, E, !t.frag);
        return;
    }
    sound(SOUND_STONE);  // a building or other static mesh
    float Ew = std::max(En, 0.5f * E);
    if (Ew > 0.02f * WALL_GC * 3.14159f * 0.15f * 0.15f && pendingWallBreaks.size() < 8) {  // maybe a breach (breakStatic)
        pendingWallBreaks.push_back({t.body, p, normalize(V3{v0[0], v0[1], v0[2]}), V3{v0[0], v0[1], v0[2]}, Ew, r + 0.4f});
        return;
    }
    int nd = std::min((int)(E / 4e3f), 120);  // building or other static mesh too strong to break: rubble and dust
    for (int i = 0; i < nd; i++)
        addDebris(DEB_DIRT, p + rndDir() * r, rndDir() * (1.f + 4.f * rnd01()) + V3{0, 1.5f, 0}, 0.03f + 0.08f * rnd01());
}

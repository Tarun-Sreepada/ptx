#include "destruction/wood.h"
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <utility>
#include "physics/physics.h"
#include "vegetation/trees.h"
#include "world/world.h"
#include "core/limits.h"
#include "core/math.h"
#include "core/random.h"
#include "destruction/thrown.h"
#include "destruction/tree_break.h"
#include "destruction/tree_damage.h"
#include "gpu/memory.h"
#include "scene/models.h"
#include "scene/vegetation.h"
#include "sim/debris.h"
#include "sim/effects.h"

static V3 woodRot(const WoodFrame& F, V3 l) {
    return {F.R[0][0] * l.x + F.R[0][1] * l.y + F.R[0][2] * l.z, F.R[1][0] * l.x + F.R[1][1] * l.y + F.R[1][2] * l.z,
            F.R[2][0] * l.x + F.R[2][1] * l.y + F.R[2][2] * l.z};
}

V3 woodRotT(const WoodFrame& F, V3 d) {
    return {F.R[0][0] * d.x + F.R[1][0] * d.y + F.R[2][0] * d.z, F.R[0][1] * d.x + F.R[1][1] * d.y + F.R[2][1] * d.z,
            F.R[0][2] * d.x + F.R[1][2] * d.y + F.R[2][2] * d.z};
}

V3 woodToWorld(const WoodFrame& F, V3 x) { return F.T + woodRot(F, x * F.s - F.ctr); }

static V3 worldToWood(const WoodFrame& F, V3 p) { return (woodRotT(F, p - F.T) + F.ctr) * (1.f / F.s); }

WoodFrame treeFrame(const TreeInst& t) {
    float c = cosf(t.yaw), sn = sinf(t.yaw);
    return {{{c, 0, sn}, {0, 1, 0}, {-sn, 0, c}}, t.p, t.scale, {0, 0, 0}};
}

WoodFrame thrownFrame(const Thrown& t) {
    WoodFrame F;
    for (int r = 0; r < 3; r++) for (int k = 0; k < 3; k++) F.R[r][k] = t.xf.m[r][k];
    F.T = {t.xf.m[0][3], t.xf.m[1][3], t.xf.m[2][3]};
    F.s = t.woodScale;
    F.ctr = t.woodCtr;
    return F;
}

uint32_t woodSeed(const TreeInst& t) { return (uint32_t)(t.ix * 73856093) ^ (uint32_t)(t.iz * 19349663); }

std::vector<WoodReq> pendingWood;

// A rock / wood piece / blast of mass at world point at with velocity vel against tree t. True if it touched the
// wood; then a break or leaf fall is queued and an impactor that broke through keeps the rest of its momentum.
bool strikeTree(const TreeInst& t, V3 at, V3 vel, float mass, V3* took) {
    uint64_t key = treeKey(t.ix, t.iz);
    auto dt = damagedTrees.find(key);
    trees::Wood whole;
    const trees::Wood* w = nullptr;
    if (dt != damagedTrees.end()) w = &dt->second.second;
    else if (!brokenTrees.count(key)) { trees::woodWhole((int)t.v, whole); w = &whole; }
    if (!w) return false;
    WoodFrame F = treeFrame(t);
    V3 lp = worldToWood(F, at), lv = woodRotT(F, vel);
    trees::WoodHit h = trees::woodStrike(*w, t.scale, &lp.x, &lv.x, mass);
    if (treeLog) fprintf(stderr, "tree hit: %.0f kg at %.1f m/s, h %.2f m: stress %.2f, %zu cut, %d leaves\n", mass, sqrtf(dot(vel, vel)), at.y - t.p.y, h.ratio, h.cuts.size(), h.lost);
    V3 J = woodRot(F, V3{h.impulse[0], h.impulse[1], h.impulse[2]});
    if (took) *took = J;
    if (h.cuts.empty() && h.lost < 8) return true;
    for (auto& q : pendingWood) if (q.body == UINT32_MAX && q.t.ix == t.ix && q.t.iz == t.iz) return true;
    pendingWood.push_back({t, UINT32_MAX, std::move(h), at, J, sqrtf(dot(vel, vel))});
    return true;
}

// Rock (or wood piece) at contact point pc: the nearest procedural tree surface within reach takes the blow.
bool strikeTrees(V3 pc, V3 vel, float mass, float reach) {
    const TreeInst* best = nullptr;
    float bd = reach;
    auto test = [&](const TreeInst& tr, const trees::Wood* w) {
        float dx = pc.x - tr.p.x, dz = pc.z - tr.p.z, R = treeR[tr.v] * 2.f * tr.scale + reach;
        if (dx * dx + dz * dz > R * R) return;
        trees::Wood whole;
        if (!w) { trees::woodWhole((int)tr.v, whole); w = &whole; }
        V3 lp = worldToWood(treeFrame(tr), pc);
        float d;
        if (trees::woodNearest(*w, &lp.x, d) >= 0 && d * tr.scale < bd) { bd = d * tr.scale; best = &tr; }
    };
    for (const TreeInst& tr : treeList) if (tr.v < (uint32_t)trees::VARIANTS) test(tr, nullptr);
    for (auto& [k, dw] : damagedTrees) test(dw.first, &dw.second);
    if (!best) return false;
    TreeInst t = *best;
    return strikeTree(t, pc, vel, mass);
}

// Thin limbs have no collider (a rock would fly through them): sweep what moved this step against them. A limb
// in the way takes the blow like any other hit, and the rock loses the momentum it gave the wood.
void sweepThinLimbs(Thrown& t, V3 p0, V3 p1, V3 vel, float mass) {
    if (treeList.empty() && damagedTrees.empty()) return;
    V3 mid = (p0 + p1) * 0.5f;
    float half = sqrtf(dot(p1 - p0, p1 - p0)) * 0.5f;
    auto test = [&](const TreeInst& tr, const trees::Wood* w) {
        float dx = mid.x - tr.p.x, dz = mid.z - tr.p.z, R = treeR[tr.v] * 2.f * tr.scale + half + t.radius;
        if (dx * dx + dz * dz > R * R) return false;
        trees::Wood whole;
        if (!w) { trees::woodWhole((int)tr.v, whole); w = &whole; }
        WoodFrame F = treeFrame(tr);
        V3 a = worldToWood(F, p0), b = worldToWood(F, p1), at;
        if (trees::woodSweep(*w, &a.x, &b.x, t.radius * 0.8f / tr.scale, &at.x) < 0) return false;
        TreeInst ti = tr;
        V3 J{0, 0, 0}, pc = woodToWorld(F, at);
        strikeTree(ti, pc, vel, mass, &J);
        V3 nv = vel - J * (1.f / mass);
        physics::setVelocity(t.body, &nv.x);
        memcpy(t.v, &nv.x, 12);
        return true;
    };
    for (const TreeInst& tr : treeList) if (tr.v < (uint32_t)trees::VARIANTS && test(tr, nullptr)) return;
    for (auto& [k, dw] : damagedTrees) if (test(dw.first, &dw.second)) return;
}

// Meshes + bodies for woods in frame F (the pose they had as one): anchored ones become the standing Stump of
// tree cell key, loose ones thrown wood pieces moving with v + av x (centre - pivot). Pieces too small to be
// worth a body become splinters. Returns the bodies of the loose pieces (UINT32_MAX for those not spawned).
struct WoodSpawn { trees::Wood w; V3 v, av, pivot; };

static std::vector<uint32_t> spawnWoods(std::vector<WoodSpawn>& ws, const WoodFrame& F, uint64_t key, uint32_t seed) {
    std::vector<uint32_t> bodies(ws.size(), UINT32_MAX);
    std::vector<trees::Piece> pcs(ws.size());
    for (size_t i = 0; i < ws.size(); i++) trees::woodMesh(ws[i].w, seed, pcs[i]);
    auto fs = std::make_unique<FragSet>();
    Model& md = fs->model;
    md.path = "<tree pieces>";
    md.state = 1;
    takeTreeMaterials(md, pcs);
    float s3 = F.s * F.s * F.s;
    struct Out { size_t piece, mesh; Xf xf; float r; V3 ctr; };
    std::vector<Out> outs;
    for (size_t i = 0; i < pcs.size(); i++) {
        MeshData& m = pcs[i].mesh;
        if (m.idx.empty()) continue;
        if (!ws[i].w.anchored && pcs[i].mass * s3 < 0.3f) {  // a twig: splinters and its leaves
            V3 q = woodToWorld(F, V3{0.5f * (m.bmin[0] + m.bmax[0]), 0.5f * (m.bmin[1] + m.bmax[1]), 0.5f * (m.bmin[2] + m.bmax[2])});
            for (int k = 0; k < 3; k++) addDebris(DEB_WOOD, q + rndDir() * 0.1f, ws[i].v + rndDir() * 1.5f, 0.03f + 0.04f * rnd01());
            for (int k = 0; k < 6; k++) addDebris(DEB_LEAF, q + rndDir() * 0.2f, ws[i].v + rndDir() * 0.8f, 0.035f + 0.03f * rnd01());
            continue;
        }
        V3 lo{1e30f, 1e30f, 1e30f}, hi{-1e30f, -1e30f, -1e30f};
        for (size_t k = 0; k < m.verts.size(); k += 8) {
            for (int j = 0; j < 3; j++) m.verts[k + j] *= F.s;
            V3 q{m.verts[k], m.verts[k + 1], m.verts[k + 2]};
            lo = {std::min(lo.x, q.x), std::min(lo.y, q.y), std::min(lo.z, q.z)};
            hi = {std::max(hi.x, q.x), std::max(hi.y, q.y), std::max(hi.z, q.z)};
        }
        V3 ctr = (lo + hi) * 0.5f;
        float r = 0.f;
        for (size_t k = 0; k < m.verts.size(); k += 8) {
            m.verts[k] -= ctr.x; m.verts[k + 1] -= ctr.y; m.verts[k + 2] -= ctr.z;
            r = std::max(r, m.verts[k] * m.verts[k] + m.verts[k + 1] * m.verts[k + 1] + m.verts[k + 2] * m.verts[k + 2]);
        }
        for (int k = 0; k < 3; k++) { m.bmin[k] = (&lo.x)[k] - (&ctr.x)[k]; m.bmax[k] = (&hi.x)[k] - (&ctr.x)[k]; }
        for (auto* col : {&pcs[i].colVerts, &pcs[i].hullPts})
            for (size_t k = 0; k < col->size(); k += 3)
                for (int j = 0; j < 3; j++) (*col)[k + j] = (*col)[k + j] * F.s - (&ctr.x)[j];
        V3 pos = F.T + woodRot(F, ctr - F.ctr);
        Xf xf = {{{F.R[0][0], F.R[0][1], F.R[0][2], pos.x}, {F.R[1][0], F.R[1][1], F.R[1][2], pos.y}, {F.R[2][0], F.R[2][1], F.R[2][2], pos.z}}};
        md.data.meshes.push_back(std::move(m));
        outs.push_back({i, md.data.meshes.size() - 1, xf, sqrtf(r), ctr});
    }
    if (outs.empty()) return bodies;
    uploadModel(md);
    FragSet* set = fs.get();
    fragSets.push_back(std::move(fs));
    for (const Out& o : outs) {
        trees::Piece& pc = pcs[o.piece];
        const MeshData& mesh = set->model.data.meshes[o.mesh];
        if (ws[o.piece].w.anchored) {
            if (pc.colIdx.empty()) continue;
            void* shape = physics::meshShape(pc.colVerts.data(), 3, (uint32_t)pc.colVerts.size() / 3, pc.colIdx.data(), (uint32_t)pc.colIdx.size() / 3);
            if (!shape) continue;
            if (stumps.size() >= (size_t)MAX_FALLEN) dropStump(0);
            Stump st{set, (uint32_t)o.mesh, 0, shape, o.xf, key};
            physics::addStatic(&st.shape, &st.xf, 1, &st.body);
            set->live++;
            stumps.push_back(st);
            continue;
        }
        void* shape = pc.hullCount.empty() ? nullptr : physics::compoundShape(pc.hullPts.data(), pc.hullCount.data(), (uint32_t)pc.hullCount.size());
        if (!shape) {  // only thin wood: the hull of the mesh (at most ~4k points)
            uint32_t nv = (uint32_t)mesh.verts.size() / 8;
            std::vector<float> sub;
            for (uint32_t k = 0; k < nv; k += std::max(1u, nv / 4096)) sub.insert(sub.end(), &mesh.verts[(size_t)k * 8], &mesh.verts[(size_t)k * 8 + 3]);
            shape = physics::convexShape(sub.data(), 3, (uint32_t)sub.size() / 3);
        }
        if (!shape) continue;
        float dens = std::clamp(pc.mass * s3 / std::max(physics::shapeVolume(shape), 1e-5f), 20.f, 2000.f);
        const WoodSpawn& sp = ws[o.piece];
        V3 c{o.xf.m[0][3], o.xf.m[1][3], o.xf.m[2][3]};
        V3 v = sp.v + cross(sp.av, c - sp.pivot);
        if (thrown.size() >= (size_t)MAX_THROWN) dropThrown(0);
        Thrown f;
        f.sc = 1.f;
        f.radius = o.r;
        f.frag = set;
        f.fragMesh = (uint32_t)o.mesh;
        f.shape = shape;
        {   // the real mass distribution: leaves spread over the crown make it hard to spin
            float wm, com[3], I[9];
            trees::woodMass(sp.w, F.s, wm, com, I);
            for (int k = 0; k < 3; k++) com[k] -= (&o.ctr.x)[k];
            f.body = wm > 0.1f ? physics::addDynamicPoseInertia(shape, o.xf, &v.x, &sp.av.x, wm, com, I)
                               : physics::addDynamicPose(shape, o.xf, &v.x, &sp.av.x, dens);
        }
        {   // air drag: a leafy crown is a sail, it tumbles slowly and settles; bare wood keeps a little spin
            int nl = 0;
            for (uint16_t n : sp.w.leaves) nl += n;
            float leafy = std::min((float)nl / std::max(pc.mass * s3, 1.f) / 40.f, 1.f);  // leaves per kg of wood
            physics::setDamping(f.body, 0.05f + 0.5f * leafy, 0.8f + 3.f * leafy);
        }
        f.xf = o.xf;
        f.cooldown = 0.25f;  // pieces settling off what they broke from aren't impacts
        f.density = dens;
        f.wood = true;
        f.woodPart = std::make_shared<trees::Wood>(sp.w);
        f.woodScale = F.s;
        f.woodCtr = o.ctr;
        f.woodSeed = seed;
        memcpy(f.v, &v.x, 12);
        set->live++;
        bodies[o.piece] = f.body;
        thrown.push_back(std::move(f));
    }
    if (set->live == 0) {
        auto keep = std::shared_ptr<FragSet>(fragSets.back().release());
        fragSets.pop_back();
        retireNext.push_back([keep]() { if (keep->model.gpu) releaseModel(keep->model); });
    }
    return bodies;
}

void processWood() {
    std::vector<WoodReq> wq;
    wq.swap(pendingWood);
    for (WoodReq& q : wq) {
        std::vector<trees::Wood> off;
        std::vector<float> shed;
        std::vector<WoodSpawn> ws;
        WoodFrame F;
        V3 v{0, 0, 0}, av{0, 0, 0}, pivot{0, 0, 0};
        uint64_t key = 0;
        uint32_t seed;
        if (q.body == UINT32_MAX) {  // standing tree
            key = treeKey(q.t.ix, q.t.iz);
            trees::Wood w;
            auto dt = damagedTrees.find(key);
            if (dt != damagedTrees.end()) w = dt->second.second;
            else if (!brokenTrees.count(key)) trees::woodWhole((int)q.t.v, w);
            else continue;
            trees::woodApply(w, q.hit, off, &shed);
            for (size_t i = 0; i < stumps.size(); i++)
                if (stumps[i].key == key) { stumps[i].key = 0; dropStump(i); break; }
            if (brokenTrees.insert(key).second) {
                placeDirty = true;
                auto it = treeBodies.find(key);
                if (it != treeBodies.end()) { physics::removeBodies(&it->second, 1); treeBodies.erase(it); }
            }
            F = treeFrame(q.t);
            seed = woodSeed(q.t);
            damagedTrees[key] = {q.t, w};
            ws.push_back({std::move(w), v, av, pivot});
        } else {  // a loose piece landed
            size_t i = 0;
            while (i < thrown.size() && thrown[i].body != q.body) i++;
            if (i == thrown.size() || !thrown[i].woodPart) continue;
            float mass;
            if (!physics::bodyState(thrown[i].body, 1.f, thrown[i].xf, &v.x, &av.x, mass)) continue;
            F = thrownFrame(thrown[i]);
            pivot = F.T;
            seed = thrown[i].woodSeed;
            trees::Wood w = *thrown[i].woodPart;
            trees::woodApply(w, q.hit, off, &shed);
            dropThrown(i);
            ws.push_back({std::move(w), v, av, pivot});
        }
        for (trees::Wood& o : off) ws.push_back({std::move(o), v, av, pivot});
        std::vector<uint32_t> bodies = spawnWoods(ws, F, key, seed);
        if (q.body == UINT32_MAX && !q.hit.cuts.empty()) {
            // the snapped part takes the blow at the contact (it falls, twisting, the way it was pushed)
            // (no faster than what hit it: they go on together)
            for (size_t i = 1; i < bodies.size(); i++) {
                float vb[3], wb[3], mb;
                Xf xb;
                if (bodies[i] == UINT32_MAX || !physics::bodyState(bodies[i], 1.f, xb, vb, wb, mb)) continue;
                float jl = sqrtf(dot(q.J, q.J)), cap = 0.8f * q.vmax * mb;
                V3 J = jl > cap ? q.J * (cap / jl) : q.J;
                physics::addImpulse(bodies[i], &J.x, &q.at.x);
                // a snapping limb doesn't fly off like a struck propeller: the rock stays in contact while the
                // joint tears, so much of the push goes in at its centre. Spin no more than ~half a turn per
                // metre of the rock's travel over the piece's length.
                float vn[3], wn[3], mn;
                Xf xn;
                if (physics::bodyState(bodies[i], 1.f, xn, vn, wn, mn)) {
                    float wl = sqrtf(wn[0] * wn[0] + wn[1] * wn[1] + wn[2] * wn[2]), r = 0.5f;
                    for (const Thrown& th : thrown) if (th.body == bodies[i]) r = std::max(th.radius, 0.3f);
                    float cap = std::max(1.5f, 0.35f * q.vmax / r);
                    if (wl > cap) { for (float& c : wn) c *= cap / wl; physics::setAngularVelocity(bodies[i], wn); }
                }
                break;
            }
            addDecal(q.at, q.J * -1.f, 0.12f, 2);
        }
        // splinters from every snap, leaves where they hung
        float sev = std::min(q.hit.ratio, 4.f);
        if (!q.hit.cuts.empty()) {
            for (size_t c = 0; c < q.hit.cuts.size(); c++)
                for (int i = 0; i < (int)(6.f * sev); i++)
                    addDebris(DEB_WOOD, q.at + rndDir() * (0.2f * F.s), v + rndDir() * (1.5f + 3.f * rnd01()) + V3{0, 1.5f, 0}, 0.03f + 0.06f * rnd01());
        }
        V3 dir = dot(q.J, q.J) > 1e-6f ? normalize(q.J) : V3{0, 0, 0};
        size_t nShed = shed.size() / 3, stepL = std::max<size_t>(1, nShed / 600);
        for (size_t i = 0; i < nShed; i += stepL) {
            V3 lq = woodToWorld(F, V3{shed[i * 3], shed[i * 3 + 1], shed[i * 3 + 2]});
            V3 lv = v + cross(av, lq - pivot);
            addDebris(DEB_LEAF, lq, lv + rndDir() * (0.4f + 1.2f * rnd01()) + dir * (1.2f * rnd01()), 0.035f + 0.03f * rnd01(), 0.3f * rnd01());
        }
    }
}

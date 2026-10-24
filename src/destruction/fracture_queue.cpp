#include "destruction/fracture_queue.h"
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <utility>
#include "physics/physics.h"
#include "physics/fracture.h"
#include "world/world.h"
#include "app/frame_state.h"
#include "core/limits.h"
#include "core/math.h"
#include "core/random.h"
#include "destruction/breach.h"
#include "destruction/buildings.h"
#include "destruction/rock_fracture.h"
#include "destruction/thrown.h"
#include "destruction/tree_break.h"
#include "destruction/wood.h"
#include "gpu/memory.h"
#include "render/targets.h"
#include "scene/models.h"
#include "sim/debris.h"

void processFractures() {
    std::vector<TreeBreakReq> tq;
    tq.swap(pendingTreeBreaks);
    for (auto& q : tq) breakTreeNow(q.t, q.impulse, q.at, q.severity);
    processWood();
    std::vector<WallBreakReq> wq;
    wq.swap(pendingWallBreaks);
    for (auto& q : wq) {
        bool broke = breakStatic(q.p, q.dir, q.En, q.reach);
        if (!broke) {  // too strong to break: rubble and dust
            int nd = std::min((int)(q.En / 4e3f), 120);
            for (int i = 0; i < nd; i++)
                addDebris(DEB_DIRT, q.p + rndDir() * (q.reach - 0.4f), rndDir() * (1.f + 4.f * rnd01()) + V3{0, 1.5f, 0}, 0.03f + 0.08f * rnd01());
            continue;
        }
        for (auto& t : thrown)  // the rock carries on through the breach with part of its speed
            if (t.body == q.body) { V3 nv = q.v * 0.6f; physics::setVelocity(t.body, &nv.x); break; }
    }
    if (pendingFractures.empty()) return;
    static TextureData tex = rockTexture();
    std::vector<FractureReq> reqs;
    reqs.swap(pendingFractures);
    for (const FractureReq& rq : reqs) {
        size_t i = 0;
        while (i < thrown.size() && thrown[i].body != rq.body) i++;
        if (i == thrown.size() || !thrown[i].poly) continue;
        Thrown& t = thrown[i];
        float v[3], w[3], mass;
        Xf xf;
        if (!physics::bodyState(t.body, 1.f, xf, v, w, mass)) continue;
        // body frame: rotation only (bodyState with scale 1), the poly is already at the body's size
        auto toLocal = [&](V3 p) { V3 d = p - V3{xf.m[0][3], xf.m[1][3], xf.m[2][3]};
                                   return fracture::Vec{xf.m[0][0] * d.x + xf.m[1][0] * d.y + xf.m[2][0] * d.z,
                                                        xf.m[0][1] * d.x + xf.m[1][1] * d.y + xf.m[2][1] * d.z,
                                                        xf.m[0][2] * d.x + xf.m[1][2] * d.y + xf.m[2][2] * d.z}; };
        auto rotW = [&](fracture::Vec a) { return V3{xf.m[0][0] * a.x + xf.m[0][1] * a.y + xf.m[0][2] * a.z,
                                                     xf.m[1][0] * a.x + xf.m[1][1] * a.y + xf.m[1][2] * a.z,
                                                     xf.m[2][0] * a.x + xf.m[2][1] * a.y + xf.m[2][2] * a.z}; };
        fracture::Vec ip = toLocal(rq.contact), idir = toLocal(V3{xf.m[0][3], xf.m[1][3], xf.m[2][3]} + rq.dir);
        float Ec = fractureThreshold(t.radius), ratio = rq.En / Ec;
        int count = std::clamp((int)(2.f + 5.f * log2f(ratio)), 2, 64);
        float spread = std::clamp(log10f(ratio) / 3.f, 0.f, 1.f);
        std::vector<fracture::Poly> pieces;
        fracture::shatter(*t.poly, ip, idir, count, spread, frameIndex * 31u + (uint32_t)i, pieces);
        if (pieces.size() < 2) continue;

        auto fs = std::make_unique<FragSet>();
        Model& md = fs->model;
        md.path = "<fragments>";
        md.state = 1;
        md.data.textures = {tex};
        MaterialData outer, inner;  // weathered outside, fresh lighter rock on the new faces
        outer.color[0] = 0.30f; outer.color[1] = 0.29f; outer.color[2] = 0.26f; outer.tex = 0;
        inner.color[0] = 0.50f; inner.color[1] = 0.48f; inner.color[2] = 0.44f; inner.tex = 0;
        md.data.materials = {outer, inner};
        float vKick = sqrtf(2.f * 0.15f * std::max(rq.En - Ec, 0.f) / std::max(mass, 1e-3f));
        V3 P0{xf.m[0][3], xf.m[1][3], xf.m[2][3]}, W{w[0], w[1], w[2]};
        struct Spawn { size_t mesh; Xf xf; V3 v, w; fracture::Poly poly; float r; };
        std::vector<Spawn> spawns;
        for (auto& pc : pieces) {
            fracture::Vec c;
            float vol = fracture::volume(pc, &c);
            float r = fracture::radius(pc, c);
            V3 cw = P0 + rotW(c), out = normalize(rotW({c.x - ip.x, c.y - ip.y, c.z - ip.z}) + rq.dir * 0.5f);
            V3 vel = V3{v[0], v[1], v[2]} + cross(W, cw - P0) + out * (vKick * (0.5f + rnd01()));
            if (r < 0.04f || vol < 2e-5f) {  // gravel
                addDebris(DEB_DIRT, cw, vel, std::max(r, 0.01f));
                continue;
            }
            fracture::translate(pc, {-c.x, -c.y, -c.z});
            MeshData mesh;
            fracture::toMesh(pc, mesh, 0, 1);
            md.data.meshes.push_back(std::move(mesh));
            Xf px = xf;
            px.m[0][3] = cw.x; px.m[1][3] = cw.y; px.m[2][3] = cw.z;
            spawns.push_back({md.data.meshes.size() - 1, px, vel, W + rndDir() * (2.f * rnd01()), pc, r});
        }
        dropThrown(i);  // parent goes; its references are gone after this
        if (spawns.empty()) continue;
        uploadModel(md);
        FragSet* set = fs.get();
        fragSets.push_back(std::move(fs));
        for (auto& sp : spawns) {
            if (thrown.size() >= (size_t)MAX_THROWN) dropThrown(0);
            const MeshData& mesh = set->model.data.meshes[sp.mesh];
            Thrown f;
            f.sc = 1.f;
            f.radius = sp.r;
            f.frag = set;
            f.fragMesh = (uint32_t)sp.mesh;
            f.shape = physics::convexShape(mesh.verts.data(), 8, (uint32_t)mesh.verts.size() / 8);
            if (!f.shape) continue;
            f.body = physics::addDynamicPose(f.shape, sp.xf, &sp.v.x, &sp.w.x, 2600.f);
            f.xf = sp.xf;
            f.poly = std::make_shared<fracture::Poly>(std::move(sp.poly));
            f.cooldown = 0.1f;  // pieces separating aren't new impacts
            memcpy(f.v, &sp.v.x, 12);
            set->live++;
            thrown.push_back(std::move(f));
        }
        if (set->live == 0) {  // nothing spawned after all
            auto keep = std::shared_ptr<FragSet>(fragSets.back().release());
            fragSets.pop_back();
            retireNext.push_back([keep]() { if (keep->model.gpu) releaseModel(keep->model); });
        }
    }
}

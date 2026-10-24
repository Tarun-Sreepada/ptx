#include "destruction/breach.h"
#include <vector>
#include <unordered_map>
#include <array>
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
#include "destruction/buildings.h"
#include "destruction/metal.h"
#include "destruction/thrown.h"
#include "gpu/memory.h"
#include "scene/models.h"
#include "scene/stream_sync.h"
#include "sim/debris.h"
#include "sim/effects.h"

// Masonry: the wall part (a box in the mesh's local space, vertical, thin along one horizontal axis) comes apart
// along its mortar. Courses of 75 mm, bricks 225 mm in running bond; bricks inside the breach (and some at its rim)
// fly out whole, the rest of each touched course stays as convex runs between the gaps, untouched courses as one
// block. Local cells go into dm, loose bricks to flyCell (world space). False if the part isn't a wall.
template <class Fly>
static bool brickBreach(const Placed& hit, const std::array<float, 6>& box, V3 p, V3 dir, float rb, uint32_t outer, uint32_t inner, MeshData& dm, Fly&& flyCell) {
    const float* mn = &box[0];
    const float* mx = &box[3];
    int thin = mx[0] - mn[0] < mx[2] - mn[2] ? 0 : 2, along = 2 - thin;
    if (mx[1] - mn[1] < 0.3f || mx[thin] - mn[thin] > 0.6f) return false;  // a slab / pillar: shatter instead
    const float BL = 0.225f, BH = 0.075f;
    Xf inv = xfInverse(hit.xf);
    float pw[3] = {p.x, p.y, p.z}, pl[3];
    xfPoint(inv, pw, pl);
    // local box poly
    std::vector<float> corners;
    for (int i = 0; i < 8; i++) corners.insert(corners.end(), {i & 1 ? mx[0] : mn[0], i & 2 ? mx[1] : mn[1], i & 4 ? mx[2] : mn[2]});
    fracture::Poly whole = fracture::fromPoints(corners.data(), 3, 8, 26);
    auto slab = [&](const fracture::Poly& q, int axis, float lo, float hi) {
        fracture::Vec n{0, 0, 0};
        (&n.x)[axis] = 1.f;
        fracture::Poly r = fracture::clip(q, n, hi);
        (&n.x)[axis] = -1.f;
        return fracture::clip(r, n, -lo);
    };
    auto stay = [&](const fracture::Poly& q) {
        if (q.faces.empty()) return;
        MeshData pm;
        fracture::toMesh(q, pm, outer, inner, 1.f);
        uint32_t base = (uint32_t)(dm.verts.size() / 8);
        dm.verts.insert(dm.verts.end(), pm.verts.begin(), pm.verts.end());
        for (uint32_t i : pm.idx) dm.idx.push_back(base + i);
        dm.triMat.insert(dm.triMat.end(), pm.triMat.begin(), pm.triMat.end());
    };
    auto world = [&](fracture::Poly q) {
        for (auto& f : q.faces)
            for (auto& v : f.pts) { float a[3] = {v.x, v.y, v.z}, o[3]; xfPoint(hit.xf, a, o); v = {o[0], o[1], o[2]}; }
        return q;
    };
    float reach = rb * 1.3f;
    int rows = std::max(1, (int)ceilf((mx[1] - mn[1]) / BH));
    float untouched0 = mn[1];  // bottom of the current run of untouched courses
    bool inRun = false;
    for (int r = 0; r <= rows; r++) {
        float y0 = mn[1] + r * BH, y1 = std::min(mx[1], y0 + BH);
        bool touched = r < rows && fabsf((y0 + y1) * 0.5f - pl[1]) < reach;
        if (!touched || r == rows) {
            if (r < rows) { if (!inRun) { untouched0 = y0; inRun = true; } continue; }
        }
        if (inRun) { stay(slab(whole, 1, untouched0, y0)); inRun = false; }
        if (r == rows) break;
        fracture::Poly course = slab(whole, 1, y0, y1);
        float off = (r & 1) ? BL * 0.5f : 0.f;
        float runStart = mn[along];
        for (float a0 = mn[along] - off; a0 < mx[along]; a0 += BL) {
            float lo = std::max(a0, mn[along]), hi = std::min(a0 + BL, mx[along]);
            if (hi - lo < 1e-3f) continue;
            float c[3];
            c[thin] = pl[thin]; c[1] = (y0 + y1) * 0.5f; c[along] = (lo + hi) * 0.5f;
            float d = sqrtf((c[1] - pl[1]) * (c[1] - pl[1]) + (c[along] - pl[along]) * (c[along] - pl[along]));
            uint32_t h = (uint32_t)(r * 7919 + (int)(a0 * 1000.f)) * 2654435761u;
            bool loose = d < rb || (d < reach && (h >> 16) % 100 < 45);
            if (!loose) continue;
            if (lo > runStart + 1e-3f) stay(slab(course, along, runStart, lo));
            flyCell(world(slab(course, along, lo, hi)), outer, inner, d < rb);
            runStart = hi;
        }
        if (runStart < mx[along] - 1e-3f) stay(slab(course, along, runStart, mx[along]));
    }
    for (int i = 0; i < (int)std::min(30.f * rb, 100.f); i++)  // mortar dust
        addDebris(DEB_DIRT, p + rndDir() * (rb * 0.6f), (dir * 2.f + rndDir()) * (1.f + 3.f * rnd01()), 0.01f + 0.02f * rnd01());
    return true;
}

static bool breakStaticImpl(V3 p, V3 dir, float En, float reach);

bool breakStatic(V3 p, V3 dir, float En, float reach) {
    // the impact marks the wall and kicks up plaster / brick dust whether or not it breaks through
    addDecal(p, dir * -1.f, std::clamp(0.25f + 0.4f * reach, 0.25f, 1.5f), 0);
    addDust(p - dir * 0.3f, std::clamp(0.4f + 0.5f * reach, 0.4f, 2.5f), 0.6f, 0.25f);
    return breakStaticImpl(p, dir, En, reach);
}

static bool breakStaticImpl(V3 p, V3 dir, float En, float reach) {
    // the placed mesh whose nearest component is closest to the contact
    Placed hit{};
    int hitPart = -1;
    float best = reach;
    stream::forEachPlaced([&](const Placed& pl) {
        const MeshData& m = pl.model->data.meshes[pl.mesh];
        if (!m.shape || m.idx.empty()) return;
        float mn[3], mx[3];
        xfAabb(pl.xf, m.bmin, m.bmax, mn, mx);
        if (p.x < mn[0] - reach || p.y < mn[1] - reach || p.z < mn[2] - reach || p.x > mx[0] + reach || p.y > mx[1] + reach || p.z > mx[2] + reach) return;
        const MeshParts& mp = meshParts(m);
        for (size_t i = 0; i < mp.box.size(); i++) {
            float bmn[3], bmx[3];
            xfAabb(pl.xf, &mp.box[i][0], &mp.box[i][3], bmn, bmx);
            float d2 = 0.f;
            for (int k = 0; k < 3; k++) { float e = std::max({bmn[k] - (&p.x)[k], (&p.x)[k] - bmx[k], 0.f}); d2 += e * e; }
            if (sqrtf(d2) < best) { best = sqrtf(d2); hit = pl; hitPart = (int)i; }
        }
    });
    if (hitPart < 0) return false;
    const MeshData& src = hit.model->data.meshes[hit.mesh];
    float rb = sqrtf(En / (WALL_GC * std::max(src.strength, 0.01f) * 3.14159f));  // breach radius
    int kind = src.breakKind;
    if (kind == BREAK_SHATTER && !src.triMat.empty() && hit.model->data.materials[src.triMat[0]].metal > 0.5f) kind = BREAK_METAL;
    if (kind == BREAK_METAL && rb < 0.7f) return dentMetal(hit, p, dir, En);  // tears only when hit very hard
    if (rb < 0.15f) return false;
    const MeshParts& mp = meshParts(src);
    Xf inv = xfInverse(hit.xf);
    auto toLocal = [&](fracture::Vec v) { float a[3] = {v.x, v.y, v.z}, o[3]; xfPoint(inv, a, o); return fracture::Vec{o[0], o[1], o[2]}; };
    // components the breach reaches (always the one hit)
    std::vector<int> parts;
    for (size_t i = 0; i < mp.box.size(); i++) {
        float bmn[3], bmx[3];
        xfAabb(hit.xf, &mp.box[i][0], &mp.box[i][3], bmn, bmx);
        float d2 = 0.f;
        for (int k = 0; k < 3; k++) { float e = std::max({bmn[k] - (&p.x)[k], (&p.x)[k] - bmx[k], 0.f}); d2 += e * e; }
        if ((int)i == hitPart || sqrtf(d2) < 0.5f * rb) parts.push_back((int)i);
    }

    // damaged copy: the untouched triangles, then the static remains of each breached part
    auto copy = std::make_shared<Model>();
    copy->state = 1;
    ModelData& cd = copy->data;
    const ModelData& sd = hit.model->data;
    cd.materials = sd.materials;
    std::unordered_map<int, int> texMap;  // only the textures this mesh's materials use
    for (auto& mt : cd.materials)
        forEachTex(mt, [&](int t) {
            if (t < 0) return t;
            auto [e, fresh] = texMap.emplace(t, (int)cd.textures.size());
            if (fresh) cd.textures.push_back(sd.textures[t]);
            return e->second;
        });
    std::vector<bool> gone(src.idx.size() / 3, false);
    for (int pi : parts) for (uint32_t t : mp.tris[pi]) gone[t] = true;
    MeshData dm;
    dm.verts = src.verts;
    for (size_t t = 0; t < gone.size(); t++)
        if (!gone[t]) {
            dm.idx.insert(dm.idx.end(), {src.idx[t * 3], src.idx[t * 3 + 1], src.idx[t * 3 + 2]});
            dm.triMat.push_back(src.triMat.empty() ? 0u : src.triMat[t]);
        }
    auto append = [&](MeshData& a, const MeshData& b) {
        uint32_t base = (uint32_t)(a.verts.size() / 8);
        a.verts.insert(a.verts.end(), b.verts.begin(), b.verts.end());
        for (uint32_t i : b.idx) a.idx.push_back(base + i);
        a.triMat.insert(a.triMat.end(), b.triMat.begin(), b.triMat.end());
    };

    // fragments: own model (the wall's materials + a fresh broken-surface one)
    auto fs = std::make_unique<FragSet>();
    Model& fm = fs->model;
    fm.path = "<rubble>";
    fm.state = 1;
    fm.data.materials = cd.materials;
    fm.data.textures = cd.textures;
    struct Spawn { size_t mesh; Xf xf; V3 v, w; fracture::Poly poly; float r; };
    std::vector<Spawn> spawns;
    // a world-space cell coming loose: grit if tiny, else a rigid fragment; inBreach = blown out by the blow
    auto flyCell = [&](fracture::Poly cell, uint32_t outer, uint32_t inner, bool inBreach) {
        fracture::Vec c;
        float vol = fracture::volume(cell, &c);
        float r = fracture::radius(cell, c);
        V3 cw{c.x, c.y, c.z};
        V3 d = cw - p;
        // ~15% of the blow goes into throwing the breach's debris (about rb^2 * pi * 25 cm of wall)
        float vKick = inBreach ? std::min(sqrtf(0.3f * En / (2000.f * 3.14159f * rb * rb * 0.25f)), 25.f) : 0.f;
        V3 out = normalize(dir + d * (1.f / std::max(sqrtf(dot(d, d)), 0.1f)) * 0.6f);
        V3 vel = out * (vKick * (0.4f + rnd01()));
        if (r < 0.03f || vol < 1e-5f) {  // grit
            addDebris(DEB_DIRT, cw, vel, std::max(r, 0.015f));
            return;
        }
        fracture::translate(cell, {-c.x, -c.y, -c.z});
        MeshData mesh;
        fracture::toMesh(cell, mesh, outer, inner);
        fm.data.meshes.push_back(std::move(mesh));
        Xf px = xfIdentity();
        px.m[0][3] = cw.x; px.m[1][3] = cw.y; px.m[2][3] = cw.z;
        spawns.push_back({fm.data.meshes.size() - 1, px, vel, rndDir() * (3.f * rnd01()), std::move(cell), r});
    };
    for (int pi : parts) {
        uint32_t outer = src.triMat.empty() ? 0u : src.triMat[mp.tris[pi][0]];
        MaterialData in = sd.materials[outer];  // broken faces: the same stuff, paler and untextured
        for (int k = 0; k < 3; k++) in.color[k] = std::min(in.color[k] * 1.25f + 0.08f, 1.f);
        forEachTex(in, [](int) { return -1; });
        in.heightScale = 0.f;
        uint32_t inner = (uint32_t)cd.materials.size();
        cd.materials.push_back(in);
        fm.data.materials.push_back(in);
        std::vector<float> pts;
        for (uint32_t t : mp.tris[pi])
            for (int j = 0; j < 3; j++) {
                const float* v = &src.verts[src.idx[t * 3 + j] * 8];
                float o[3];
                xfPoint(hit.xf, v, o);
                pts.insert(pts.end(), {o[0], o[1], o[2]});
            }
        if (kind == BREAK_BRICK && brickBreach(hit, mp.box[pi], p, dir, rb, outer, inner, dm, flyCell)) continue;
        fracture::Poly whole = fracture::fromPoints(pts.data(), 3, (uint32_t)pts.size() / 3, 64);
        // world box of the part: cells touching its faces across the thin axis (floor, ceiling, wall ends) hold on
        float bmn[3], bmx[3];
        xfAabb(hit.xf, &mp.box[pi][0], &mp.box[pi][3], bmn, bmx);
        int thin = 0;
        for (int k = 1; k < 3; k++) if (bmx[k] - bmn[k] < bmx[thin] - bmn[thin]) thin = k;
        float size = std::max({bmx[0] - bmn[0], bmx[1] - bmn[1], bmx[2] - bmn[2]});
        int count = std::clamp((int)(8.f + 14.f * rb + 2.f * size), 8, 96);
        float spread = std::clamp(rb / std::max(size, 0.1f), 0.f, 1.f) * 0.6f;
        std::vector<fracture::Poly> cells;
        fracture::shatter(whole, {p.x, p.y, p.z}, {dir.x, dir.y, dir.z}, count, spread, frameIndex * 131u + (uint32_t)pi, cells);
        for (auto& cell : cells) {
            fracture::Vec c;
            float vol = fracture::volume(cell, &c);
            float r = fracture::radius(cell, c);
            V3 cw{c.x, c.y, c.z};
            V3 d = cw - p;
            bool held = false;
            for (auto& f : cell.faces)
                for (auto& v : f.pts)
                    for (int k = 0; k < 3; k++)
                        if (k != thin && (fabsf((&v.x)[k] - bmn[k]) < 0.01f || fabsf((&v.x)[k] - bmx[k]) < 0.01f)) held = true;
            bool flies = sqrtf(dot(d, d)) < rb || !held;
            if (!flies) {  // stays: into the damaged copy, in the mesh's local space
                fracture::Poly lp = cell;
                for (auto& f : lp.faces) for (auto& v : f.pts) v = toLocal(v);
                MeshData pm;
                fracture::toMesh(lp, pm, outer, inner);
                append(dm, pm);
                continue;
            }
            flyCell(std::move(cell), outer, inner, sqrtf(dot(d, d)) < rb);
        }
        for (int i = 0; i < (int)std::min(40.f * rb, 150.f); i++)  // dust and chips from the breach
            addDebris(DEB_DIRT, p + rndDir() * (rb * 0.6f), (dir * 2.f + rndDir()) * (1.f + 3.f * rnd01()), 0.02f + 0.05f * rnd01() * rnd01());
    }
    for (int k = 0; k < 3; k++) { dm.bmin[k] = 1e30f; dm.bmax[k] = -1e30f; }
    for (size_t i = 0; i < dm.verts.size(); i += 8)
        for (int k = 0; k < 3; k++) { dm.bmin[k] = std::min(dm.bmin[k], dm.verts[i + k]); dm.bmax[k] = std::max(dm.bmax[k], dm.verts[i + k]); }
    if (!dm.idx.empty())
        dm.shape = physics::meshShape(dm.verts.data(), 8, (uint32_t)(dm.verts.size() / 8), dm.idx.data(), (uint32_t)(dm.idx.size() / 3));
    else {  // nothing left: a degenerate sliver far below keeps the slot valid without being seen
        dm.verts = {0, -1000, 0, 0, 1, 0, 0, 0, 0.01f, -1000, 0, 0, 1, 0, 0, 0, 0, -1000, 0.01f, 0, 1, 0, 0, 0};
        dm.idx = {0, 2, 1};
        dm.triMat = {0};
        for (int k = 0; k < 3; k++) { dm.bmin[k] = dm.verts[k]; dm.bmax[k] = dm.verts[8 + k] + 0.01f; }
    }
    dm.strength = src.strength;
    dm.breakKind = src.breakKind;
    cd.meshes.push_back(std::move(dm));
    if (!uploadModel(*copy)) {
        if (cd.meshes[0].shape) physics::releaseShape(cd.meshes[0].shape);
        return false;
    }
    std::shared_ptr<Model> old = stream::overridePlaced(stream::placedKey(hit), copy);
    damagedModels.push_back(copy);
    if (old) {  // the previous damaged copy: its bodies are gone (cell re-placed), free it after this frame
        damagedModels.erase(std::remove(damagedModels.begin(), damagedModels.end(), old), damagedModels.end());
        retireNext.push_back([old]() {
            if (old->gpu) releaseModel(*old);
            if (old->data.meshes[0].shape) physics::releaseShape(old->data.meshes[0].shape);
        });
    }
    onStreamChanged();

    if (!spawns.empty()) {
        uploadModel(fm);
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
            f.body = physics::addDynamicPose(f.shape, sp.xf, &sp.v.x, &sp.w.x, 2000.f);
            f.xf = sp.xf;
            f.poly = std::make_shared<fracture::Poly>(std::move(sp.poly));
            f.cooldown = 0.2f;
            f.density = 2000.f;
            memcpy(f.v, &sp.v.x, 12);
            set->live++;
            thrown.push_back(std::move(f));
        }
        if (set->live == 0) {
            auto keep = std::shared_ptr<FragSet>(fragSets.back().release());
            fragSets.pop_back();
            retireNext.push_back([keep]() { if (keep->model.gpu) releaseModel(keep->model); });
        }
    }
    return true;
}

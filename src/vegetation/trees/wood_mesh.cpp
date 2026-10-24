// Meshes and colliders of wood parts (a standing tree, its stump, a broken limb): the limb surfaces of the
// present elements, the foliage they carry, splintered fresh wood at every break.
#include "vegetation/trees.h"

#include "vegetation/trees/foliage.h"
#include "vegetation/trees/limb_mesh.h"
#include "vegetation/trees/wood_shared.h"

namespace trees {
// closed prism along a..c, appended (pos3)
void prism(V a, V c, float ra, float rc, int S, std::vector<float>& verts, std::vector<uint32_t>& idx) {
    V ax = norm(c - a);
    V t = perpTo(ax), bt = cross(ax, t);
    uint32_t base = (uint32_t)(verts.size() / 3);
    for (int ring = 0; ring < 2; ring++)
        for (int i = 0; i < S; i++) {
            float ang = TAU * i / S, r = ring ? rc : ra;
            V o = (ring ? c : a) + (t * cosf(ang) + bt * sinf(ang)) * r;
            verts.insert(verts.end(), {o.x, o.y, o.z});
        }
    for (uint32_t i = 0; i < (uint32_t)S; i++) {
        uint32_t q0 = base + i, q1 = base + (i + 1) % S, q2 = base + S + i, q3 = base + S + (i + 1) % S;
        idx.insert(idx.end(), {q0, q2, q1, q1, q2, q3});
        if (i > 0 && i < (uint32_t)S - 1) idx.insert(idx.end(), {base, base + i, base + i + 1, base + S, base + S + i + 1, base + S + i});
    }
}
bool colliderLimb(const Limb& l) { return l.level == 0 || (l.level >= 1 && l.ra >= COLLIDER_MIN_R); }
bool isCollider(const Skel& S, int k) { return k < TRUNK_ELEMS || S.e[k].level == 0 || S.e[k].ra >= COLLIDER_MIN_R; }
void colliderOf(const Plan& p, const std::vector<bool>* skip, std::vector<float>& verts, std::vector<uint32_t>& idx) {
    verts.clear(); idx.clear();
    prism(p.trunk.a, p.trunk.b, p.trunk.ra * (1.f + 0.3f * p.flare), p.trunk.rb, 8, verts, idx);
    for (size_t i = 0; i < p.branches.size(); i++)
        if (colliderLimb(p.branches[i]) && !(skip && (*skip)[i])) prism(p.branches[i].a, p.branches[i].b, p.branches[i].ra, p.branches[i].rb, 6, verts, idx);
}
float segDist(V q, V a, V b, float& t) {
    V ab = b - a;
    t = std::clamp(dot(q - a, ab) / std::max(dot(ab, ab), 1e-8f), 0.f, 1.f);
    V d = q - (a + ab * t);
    return sqrtf(dot(d, d));
}

namespace {
// Splintered end closing a tube ring (same basis as tube()) at c: jagged fresh wood, pointing along out.
void breakEnd(Builder& b, V c, V axis, V out, float r, int sides, Rng& rng, uint32_t mat) {
    V ax = norm(axis);
    V t = norm(cross(fabsf(ax.y) < 0.99f ? V{0, 1, 0} : V{1, 0, 0}, ax)), bt = cross(ax, t);
    V o = norm(out);
    uint32_t base = (uint32_t)(b.m.verts.size() / 8);
    for (int i = 0; i <= sides; i++) {
        float ang = TAU * i / sides;
        V d = t * cosf(ang) + bt * sinf(ang);
        b.vtx(c + d * r, d.x * r, d.z * r);
    }
    uint32_t mid = (uint32_t)(b.m.verts.size() / 8);
    for (int i = 0; i <= sides; i++) {  // inner ring of splinters, each sticking out by a random amount
        float ang = TAU * (i % sides) / sides;
        V d = t * cosf(ang) + bt * sinf(ang);
        b.vtx(c + d * (r * rng.in(0.35f, 0.7f)) + o * (r * rng.in(0.f, 1.6f)), d.x * r * 0.5f, d.z * r * 0.5f);
    }
    uint32_t tip = b.vtx(c + o * (r * rng.in(0.3f, 2.2f)), 0, 0);
    bool flip = dot(cross(t, bt), o) < 0.f;  // keep the winding facing out
    for (int i = 0; i < sides; i++) {
        uint32_t a0 = base + i, a1 = a0 + 1, m0 = mid + i, m1 = m0 + 1;
        if (flip) { b.tri(a0, m0, a1, mat); b.tri(a1, m0, m1, mat); b.tri(m0, tip, m1, mat); }
        else { b.tri(a0, a1, m0, mat); b.tri(a1, m1, m0, mat); b.tri(m0, m1, tip, mat); }
    }
    b.endPart();
}
}  // namespace

void trunkCollider(const ModelData&, int v, std::vector<float>& verts, std::vector<uint32_t>& idx) { colliderOf(plan(v), nullptr, verts, idx); }

void woodMesh(const Wood& w, uint32_t seed, Piece& out) {
    out = {};
    const Skel& S = skel(w.v);
    const Plan& p = plan(w.v);
    Builder b{out.mesh};
    for (size_t k = 0; k < S.e.size(); k++)
        if (w.has[k]) out.mass += S.e[k].mass;
    emitLimbs(b, p, p.woodMat, &w.has);
    // foliage on the wood that's left: the leaves still on it, bare shoots where they all came off
    for (size_t i = 0; i < p.leaves.size(); i++) {
        if (!w.has[S.leafElem[i]]) continue;
        emitFoliage(b, p.leaves[i], w.leaves[i], p);
        out.mass += S.leafMass[i] * w.leaves[i];
    }
    // splintered fresh wood where it snapped, the same splinters on every rebuild
    for (int c : w.cuts) {
        const Elem& E = S.e[c];
        Rng rng{seed ^ ((uint32_t)c * 2654435761u)};
        V ax = E.b - E.a;
        bool trunk = c < TRUNK_ELEMS;
        int sides = trunk ? 16 : std::max(p.branches[c - TRUNK_ELEMS].sides, 7);
        float r = trunk ? E.ra * 1.05f : E.ra;
        if (w.has[c]) breakEnd(b, E.a, ax, ax * -1.f, r, sides, rng, MAT_WOOD);
        else if (E.parent >= 0 && w.has[E.parent]) { rng.s ^= 0x9e3779b9u; breakEnd(b, E.a, ax, ax, r * 0.9f, sides, rng, MAT_WOOD); }
    }
    for (size_t k = 0; k < S.e.size(); k++) {
        if (!w.has[k] || !isCollider(S, (int)k)) continue;
        const Elem& E = S.e[k];
        if (w.anchored) { prism(E.a, E.b, E.ra, E.rb, k < (size_t)TRUNK_ELEMS ? 8 : 6, out.colVerts, out.colIdx); continue; }
        V ax = norm(E.b - E.a), t = perpTo(ax), bt = cross(ax, t);  // loose: one convex prism per limb
        // the broken end starts a little in, clear of the stub it came off (overlapping bodies spawn explosively)
        V a0 = E.a;
        bool root = E.parent < 0 || !w.has[E.parent];
        if (root) a0 = E.a + (E.b - E.a) * std::min(1.5f * E.ra / std::max(length(E.b - E.a), 1e-4f), 0.4f);
        for (int ring = 0; ring < 2; ring++)
            for (int i = 0; i < 6; i++) {
                float ang = TAU * i / 6, r = ring ? E.rb : E.ra;
                V o = (ring ? E.b : a0) + (t * cosf(ang) + bt * sinf(ang)) * std::max(r, 0.01f);
                out.hullPts.insert(out.hullPts.end(), {o.x, o.y, o.z});
            }
        out.hullCount.push_back(12);
    }
    finishMesh(out.mesh);
}
}  // namespace trees

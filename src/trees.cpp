#include "trees.h"

#include <cmath>
#include <algorithm>
#include <cstring>
#include <cstdio>

namespace trees {
namespace {

struct V { float x, y, z; };
V operator+(V a, V b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
V operator-(V a, V b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V operator*(V a, float s) { return {a.x * s, a.y * s, a.z * s}; }
float dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V cross(V a, V b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
V norm(V a) { float l = sqrtf(dot(a, a)); return l > 0 ? a * (1.f / l) : V{0, 1, 0}; }

struct Rng {
    uint32_t s;
    float operator()() { s = s * 747796405u + 2891336453u; uint32_t w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return ((w >> 22u) ^ w) / 4294967296.f; }
    float in(float a, float b) { return a + (b - a) * (*this)(); }
};

// 3D value noise for canopy lumps
float h3(int x, int y, int z) { return hash2(x * 73856093 ^ z * 19349663, y * 83492791); }
float noise3(V p) {
    int ix = (int)floorf(p.x), iy = (int)floorf(p.y), iz = (int)floorf(p.z);
    float fx = p.x - ix, fy = p.y - iy, fz = p.z - iz;
    fx = fx * fx * (3 - 2 * fx); fy = fy * fy * (3 - 2 * fy); fz = fz * fz * (3 - 2 * fz);
    auto L = [](float a, float b, float t) { return a + (b - a) * t; };
    float a = L(L(h3(ix, iy, iz), h3(ix + 1, iy, iz), fx), L(h3(ix, iy + 1, iz), h3(ix + 1, iy + 1, iz), fx), fy);
    float b = L(L(h3(ix, iy, iz + 1), h3(ix + 1, iy, iz + 1), fx), L(h3(ix, iy + 1, iz + 1), h3(ix + 1, iy + 1, iz + 1), fx), fy);
    return L(a, b, fz);
}

// Appends to one mesh; normals are accumulated per part and normalized when the part ends.
struct Builder {
    MeshData& m;
    size_t partStart = 0;
    uint32_t vtx(V p, float u, float v) {
        m.verts.insert(m.verts.end(), {p.x, p.y, p.z, 0, 0, 0, u, v});
        return (uint32_t)(m.verts.size() / 8 - 1);
    }
    V pos(uint32_t i) const { return {m.verts[i * 8], m.verts[i * 8 + 1], m.verts[i * 8 + 2]}; }
    void tri(uint32_t a, uint32_t b, uint32_t c, uint32_t mat) {
        m.idx.insert(m.idx.end(), {a, b, c});
        m.triMat.push_back(mat);
        V n = cross(pos(b) - pos(a), pos(c) - pos(a));
        for (uint32_t i : {a, b, c}) for (int k = 0; k < 3; k++) m.verts[i * 8 + 3 + k] += (&n.x)[k];
    }
    void endPart() {
        for (size_t i = partStart; i < m.verts.size() / 8; i++) {
            V n = norm({m.verts[i * 8 + 3], m.verts[i * 8 + 4], m.verts[i * 8 + 5]});
            memcpy(&m.verts[i * 8 + 3], &n, 12);
        }
        partStart = m.verts.size() / 8;
    }
};

const float TAU = 6.2831853f;

// Tapered tube from a to b (open ends; caps are hidden inside the canopy / ground).
void tube(Builder& b, V a, V c, float ra, float rc, uint32_t mat, int sides = 8) {
    V ax = norm(c - a);
    V t = norm(cross(fabsf(ax.y) < 0.99f ? V{0, 1, 0} : V{1, 0, 0}, ax)), bt = cross(ax, t);
    float len = sqrtf(dot(c - a, c - a));
    uint32_t base = (uint32_t)(b.m.verts.size() / 8);
    for (int ring = 0; ring < 2; ring++)
        for (int i = 0; i <= sides; i++) {
            float ang = TAU * i / sides, r = ring ? rc : ra;
            V p = (ring ? c : a) + (t * cosf(ang) + bt * sinf(ang)) * r;
            b.vtx(p, (float)i / sides * 2.f, ring ? len * 0.5f : 0.f);
        }
    for (int i = 0; i < sides; i++) {
        uint32_t q0 = base + i, q1 = q0 + 1, q2 = q0 + sides + 1, q3 = q2 + 1;
        b.tri(q0, q2, q1, mat);
        b.tri(q1, q2, q3, mat);
    }
    b.endPart();
}

// Lumpy closed blob (subdivided icosahedron), flattened underneath like a real crown.
void blob(Builder& b, V c, V r, float seed, uint32_t mat, int subdiv = 2) {
    const float t = 1.618034f;
    std::vector<V> v = {{-1, t, 0}, {1, t, 0}, {-1, -t, 0}, {1, -t, 0}, {0, -1, t}, {0, 1, t},
                        {0, -1, -t}, {0, 1, -t}, {t, 0, -1}, {t, 0, 1}, {-t, 0, -1}, {-t, 0, 1}};
    std::vector<uint32_t> idx = {0, 11, 5, 0, 5, 1, 0, 1, 7, 0, 7, 10, 0, 10, 11, 1, 5, 9, 5, 11, 4, 11, 10, 2, 10, 7, 6, 7, 1, 8,
                                 3, 9, 4, 3, 4, 2, 3, 2, 6, 3, 6, 8, 3, 8, 9, 4, 9, 5, 2, 4, 11, 6, 2, 10, 8, 6, 7, 9, 8, 1};
    for (int sub = 0; sub < subdiv; sub++) {
        std::vector<uint32_t> out;
        std::vector<std::pair<uint64_t, uint32_t>> cache;
        auto mid = [&](uint32_t a, uint32_t bb) {
            uint64_t key = ((uint64_t)std::min(a, bb) << 32) | std::max(a, bb);
            for (auto& e : cache) if (e.first == key) return e.second;
            v.push_back((v[a] + v[bb]) * 0.5f);
            cache.push_back({key, (uint32_t)v.size() - 1});
            return (uint32_t)v.size() - 1;
        };
        for (size_t i = 0; i < idx.size(); i += 3) {
            uint32_t a = idx[i], bb = idx[i + 1], cc = idx[i + 2], ab = mid(a, bb), bc = mid(bb, cc), ca = mid(cc, a);
            out.insert(out.end(), {a, ab, ca, bb, bc, ab, cc, ca, bc, ab, bc, ca});
        }
        idx = out;
    }
    uint32_t base = (uint32_t)(b.m.verts.size() / 8);
    for (V p : v) {
        p = norm(p);
        float d = 1.f + 0.32f * (noise3(p * 1.8f + V{seed, seed * 0.7f, 0}) - 0.5f) + 0.14f * (noise3(p * 5.f + V{0, seed, 3}) - 0.5f);
        if (p.y < 0) p.y *= 0.65f;
        V q = c + V{p.x * r.x, p.y * r.y, p.z * r.z} * d;
        b.vtx(q, q.x * 0.35f + q.z * 0.2f, q.y * 0.35f - q.z * 0.15f);
    }
    for (size_t i = 0; i < idx.size(); i += 3) b.tri(base + idx[i], base + idx[i + 1], base + idx[i + 2], mat);
    b.endPart();
}

// A tree as parts, so the same tree can be built whole or broken up: the trunk (bole), limbs (leader and branch
// segments, each knowing the limb it grows from) and foliage clusters (owned by a limb).
struct Limb { V a, b; float ra, rb; int sides, parent, level = 0; };  // tapered tube; parent = limb index, -1 = on the trunk
enum { F_LEAVES, F_FRUIT };
struct Foliage { int kind; V c, dir; float spread, size; int n; float seed; int owner; };
struct Plan { Limb trunk; std::vector<Limb> branches; std::vector<Foliage> leaves; uint32_t leafMat, woodMat = 0, fruitMat = 0; float leafWidth = 0.5f, droop = 0.3f, flare = 0.f, wave = 0.f, seed = 0.f; };

const float PI = 3.14159265f;

// rotate v about unit axis k by angle a (Rodrigues)
V rot(V v, V k, float a) {
    float c = cosf(a), s = sinf(a);
    return v * c + cross(k, v) * s + k * (dot(k, v) * (1.f - c));
}
V perpTo(V d) { return norm(cross(d, fabsf(d.y) < 0.95f ? V{0, 1, 0} : V{1, 0, 0})); }
V randUnit(Rng& r) {
    float z = r.in(-1.f, 1.f), a = r.in(0.f, TAU), s = sqrtf(std::max(0.f, 1.f - z * z));
    return {s * cosf(a), z, s * sinf(a)};
}

// crown envelope: branch length factor at ratio r (1 at the crown base, 0 at the top)
float shapeRatio(int s, float r) {
    r = std::clamp(r, 0.f, 1.f);
    switch (s) {
    case CROWN_CONICAL: return 0.2f + 0.8f * r;
    case CROWN_SPHERICAL: return 0.2f + 0.8f * sinf(PI * r);
    case CROWN_HEMISPHERICAL: return 0.2f + 0.8f * sinf(0.5f * PI * r);
    case CROWN_CYLINDRICAL: return 1.f;
    case CROWN_TAPERED: return 0.5f + 0.5f * r;
    case CROWN_FLAME: return std::max(0.1f, r <= 0.7f ? r / 0.7f : (1.f - r) / 0.3f);
    default: return 1.f - 0.8f * r;  // inverse conical
    }
}
bool excurrent(int s) { return s == CROWN_CONICAL || s == CROWN_CYLINDRICAL || s == CROWN_TAPERED || s == CROWN_FLAME; }

constexpr int MAX_LEAVES = 40000;  // per tree (4 tris each)

struct Gen {
    const TreeParams& P;
    Plan& p;
    Rng rng;
    float H, boleY;
    int leafCount = 0;

    // A limb grown from a along dir as a chain of tube segments, bending with gravity / light / gnarl; children
    // spiral off it; at the last level it carries leaf clusters. level 0 = the leader.
    void branch(V a, V dir, float len, float r0, int level, int parent) {
        int nseg = level == 0 ? 7 : (level == 1 ? 5 : 3);
        float sl = len / nseg;
        std::vector<V> pts{a};
        std::vector<float> rad{r0};
        std::vector<int> seg;
        V d = dir, pos = a;
        float rTip = std::max(r0 * (level == 0 ? 0.08f : 0.3f), 0.004f);
        for (int s = 0; s < nseg; s++) {
            float t = (s + 1.f) / nseg;
            float r1 = r0 + (rTip - r0) * t;
            V bend = V{0, 1, 0} * (P.upturn * (level ? 0.5f : 0.15f)) - V{0, 1, 0} * (P.gravity * t * (level ? 1.f : 0.f)) +
                     randUnit(rng) * (P.gnarl * 0.45f);
            d = norm(d + bend * (1.f / nseg) * 2.f);
            V q = pos + d * sl;
            int sides = level == 0 ? 8 : (level == 1 ? 6 : (r1 > 0.015f ? 5 : 3));
            p.branches.push_back({pos, q, rad.back(), r1, sides, seg.empty() ? parent : seg.back(), level});
            seg.push_back((int)p.branches.size() - 1);
            pts.push_back(q);
            rad.push_back(r1);
            pos = q;
        }
        auto at = [&](float t, V& P0, float& r, V& dd, int& owner) {
            float f = std::clamp(t, 0.f, 0.9999f) * nseg;
            int i = (int)f;
            float u = f - i;
            P0 = pts[i] + (pts[i + 1] - pts[i]) * u;
            r = rad[i] + (rad[i + 1] - rad[i]) * u;
            dd = norm(pts[i + 1] - pts[i]);
            owner = seg[i];
        };
        if (level >= P.levels) {  // twig: leaf clusters along its outer part, one at the tip
            if (P.leafDensity <= 0.f) return;  // bare (dead) tree
            int nc = std::clamp((int)lroundf(1.f + len / std::max(P.leafSpread * 1.2f, 0.1f)), 1, 8);
            for (int i = 0; i < nc; i++) {
                float t = nc == 1 ? 1.f : 0.3f + 0.7f * i / (nc - 1);
                V c, dd; float r; int ow;
                at(t, c, r, dd, ow);
                int n = std::max(2, (int)lroundf(28.f * P.leafDensity));
                p.leaves.push_back({F_LEAVES, c, dd, P.leafSpread * (0.6f + 0.4f * t), P.leafSize, n, rng.in(0.f, 1000.f), ow});
                leafCount += n;
            }
            return;
        }
        int n = level == 0 ? P.branches : std::max(1, (int)lroundf(P.children * rng.in(0.7f, 1.2f)));
        float t0 = level == 0 ? 0.f : 0.2f;
        float phi = rng.in(0.f, TAU);
        for (int i = 0; i < n; i++) {
            float t = t0 + (0.97f - t0) * (i + rng.in(0.f, 0.8f)) / n;
            V c, dd; float r; int ow;
            at(t, c, r, dd, ow);
            phi += 2.3999632f + rng.in(-0.3f, 0.3f);  // golden angle
            float th = (P.branchAngle + rng.in(-P.angleVar, P.angleVar)) * PI / 180.f;
            if (level == 0) th *= 0.75f + 0.5f * t;  // lower branches flatter... upper steeper
            V perp = rot(perpTo(dd), dd, phi);
            V cd = norm(dd * cosf(th) + perp * sinf(th));
            float cl;
            if (level == 0) {
                float ratio = (H - c.y) / std::max(H - boleY, 0.5f);
                cl = (H - boleY) * P.lengthRatio * shapeRatio(P.crown, ratio) * rng.in(0.85f, 1.15f);
            } else cl = P.childLength * (len - 0.6f * t * len) * rng.in(0.8f, 1.15f);
            if (cl < 0.05f) continue;
            float cr = std::max(r * P.radiusRatio * (level == 0 ? 0.9f : 1.f), 0.005f);
            branch(c, cd, cl, cr, level + 1, ow);
        }
        if (level > 0 && level == P.levels - 1 && P.leafDensity > 0.f) {  // the outer half of a sub-branch carries leaves between its twigs too
            for (float t : {0.55f, 0.8f, 1.f}) {
                V c, dd; float r; int ow;
                at(t, c, r, dd, ow);
                int nn = std::max(2, (int)lroundf(20.f * P.leafDensity));
                p.leaves.push_back({F_LEAVES, c, dd, P.leafSpread, P.leafSize, nn, rng.in(0.f, 1000.f), ow});
                leafCount += nn;
            }
        }
    }
};

Plan growTree(const TreeParams& P0, uint32_t seed, uint32_t leafMat) {
    TreeParams P = P0;
    Rng rng{seed * 2654435761u + 1u};
    P.height *= rng.in(0.88f, 1.12f);
    Plan p;
    p.leafMat = leafMat;
    p.leafWidth = P.leafWidth;
    p.droop = P.leafDroop;
    Gen g{P, p, rng, P.height, std::clamp(P.bole, 0.05f, 0.9f) * P.height};
    V lean = V{rng.in(-1.f, 1.f), 0, rng.in(-1.f, 1.f)} * (P.lean * g.boleY);
    V top = V{0, g.boleY, 0} + lean;
    p.trunk = {{0, -0.3f, 0}, top, P.trunkRadius * 1.15f, P.trunkRadius * 0.85f, 10, -1, -1};
    p.flare = P.flare;
    p.wave = P.trunkWave;
    p.seed = (float)(seed % 997u);
    float ll = (P.height - g.boleY) * (excurrent(P.crown) ? 1.f : 0.6f);
    g.branch(top, norm(V{0, 1, 0} + lean * (1.f / std::max(g.boleY, 0.5f))), ll, P.trunkRadius * 0.85f, 0, -1);
    if (g.leafCount > MAX_LEAVES) {  // over budget: thin every cluster evenly (bigger leaves keep the crown full)
        float k = (float)MAX_LEAVES / g.leafCount;
        for (Foliage& f : p.leaves) { f.n = std::max(1, (int)(f.n * k)); f.size *= 1.f / sqrtf(std::max(k, 0.25f)); }
    }
    return p;
}

// Bush: stems out of the ground (no bole), the same branching below them, fruit hanging under twig ends.
Plan growBush(const BushParams& B, uint32_t seed, uint32_t leafMat, uint32_t woodMat, uint32_t fruitMat) {
    TreeParams P;
    P.levels = std::max(B.levels, 1);
    P.children = 4;
    P.branchAngle = B.branchAngle;
    P.angleVar = 12.f;
    P.childLength = 0.55f;
    P.radiusRatio = 0.65f;
    P.gravity = 0.25f;
    P.upturn = 0.15f;
    P.gnarl = B.gnarl;
    P.leafSize = B.leafSize;
    P.leafWidth = 0.55f;
    P.leafDensity = B.leafDensity;
    P.leafSpread = 0.12f * B.size + 0.05f;
    P.leafDroop = 0.25f;
    Plan p;
    p.leafMat = leafMat;
    p.woodMat = woodMat;
    p.fruitMat = fruitMat;
    p.leafWidth = P.leafWidth;
    p.droop = P.leafDroop;
    Rng rng{seed * 3266489917u + 11u};
    Gen g{P, p, rng, B.size, 0.f};
    p.trunk = {{0, -0.05f, 0}, {0, 0.02f, 0}, 0.02f * B.size, 0.02f * B.size, 6, -1};  // root crown, hidden in the ground
    for (int i = 0; i < std::max(B.stems, 1); i++) {
        float a = TAU * (i + rng.in(-0.35f, 0.35f)) / std::max(B.stems, 1);
        V dir = norm(V{cosf(a) * B.spread, 1.f, sinf(a) * B.spread} + randUnit(rng) * 0.15f);
        V base{cosf(a) * 0.04f * B.size, 0.f, sinf(a) * 0.04f * B.size};
        g.branch(base, dir, B.size * rng.in(0.65f, 1.05f), 0.014f * B.size + 0.004f, 1, -1);
    }
    if (B.fruitDensity > 0.f && fruitMat) {
        size_t nl = p.leaves.size();
        for (size_t i = 0; i < nl; i++) {
            if (rng() > 0.45f * B.fruitDensity) continue;
            const Foliage& f = p.leaves[i];
            int n = 2 + (int)(rng() * 5.f * std::min(B.fruitDensity, 2.f));
            p.leaves.push_back({F_FRUIT, f.c + V{0, -B.fruitSize * 2.f, 0}, f.dir, B.fruitSize * 3.f, B.fruitSize, n, rng.in(0.f, 1000.f), f.owner});
        }
    }
    return p;
}

// One leaf: a pointed oval from base along d, curling down by droop (single-sided: shaders face it to the viewer).
void leaf(Builder& b, V base, V d, V up, float L, float W, float droop, uint32_t mat) {
    V side = norm(cross(d, up)), n = cross(side, d);
    auto P = [&](float t) { return base + d * (L * t) - n * (droop * L * t * t); };
    uint32_t v0 = b.vtx(base, 0.5f, 0.f);
    uint32_t l1 = b.vtx(P(0.3f) - side * (W * 0.85f), 0.f, 0.3f), r1 = b.vtx(P(0.3f) + side * (W * 0.85f), 1.f, 0.3f);
    uint32_t l2 = b.vtx(P(0.65f) - side * (W * 0.7f), 0.1f, 0.65f), r2 = b.vtx(P(0.65f) + side * (W * 0.7f), 0.9f, 0.65f);
    uint32_t tp = b.vtx(P(1.f), 0.5f, 1.f);
    b.tri(v0, l1, r1, mat);
    b.tri(l1, l2, r1, mat);
    b.tri(r1, l2, r2, mat);
    b.tri(l2, tp, r2, mat);
    b.endPart();
}

// Real wood isn't a cylinder: the cross-section is lumpy (fluted, varying with height), the centre line wanders,
// and the trunk flares into root buttresses at the ground. The radius is a function of the world-space direction
// and ring centre, so adjoining limb segments and the logs of a broken trunk meet without seams.
// t0..t1: the part of the limb to build (a broken trunk's log); flare only on the trunk.
void wavyTube(Builder& b, const Limb& l, float t0, float t1, uint32_t mat, const Plan& p, bool trunk) {
    V axis = l.b - l.a;
    float len = sqrtf(dot(axis, axis));
    V ax = axis * (1.f / std::max(len, 1e-5f));
    V t = norm(cross(fabsf(ax.y) < 0.99f ? V{0, 1, 0} : V{1, 0, 0}, ax)), bt = cross(ax, t);
    float wave = p.wave * (trunk ? 1.f : std::min(1.f, l.ra / 0.08f));  // twigs stay round
    if (wave <= 0.f && !trunk) { tube(b, l.a + axis * t0, l.a + axis * t1, l.ra + (l.rb - l.ra) * t0, l.ra + (l.rb - l.ra) * t1, mat, l.sides); return; }
    int sides = trunk ? 16 : std::max(l.sides, 7);
    int rings = std::max(2, (int)ceilf((t1 - t0) * len / (trunk ? 0.18f : 0.3f)) + 1);
    uint32_t base = (uint32_t)(b.m.verts.size() / 8);
    float S = p.seed;
    for (int r = 0; r < rings; r++) {
        float tt = t0 + (t1 - t0) * r / (rings - 1);
        float y = tt * len;
        V c = l.a + axis * tt;
        if (trunk) {  // the centre line wanders (pinned at both ends so the leader joins on)
            float w = sinf(3.14159265f * tt) * wave * l.ra * 1.2f;
            c = c + t * ((noise3({y * 0.35f, S, 1.7f}) - 0.5f) * 2.f * w) + bt * ((noise3({y * 0.35f, S + 9.f, 4.1f}) - 0.5f) * 2.f * w);
        }
        float rr = l.ra + (l.rb - l.ra) * tt;
        for (int i = 0; i <= sides; i++) {
            float ang = TAU * (i % sides) / sides;
            V d = t * cosf(ang) + bt * sinf(ang);
            // lumpy, fluted cross-section that changes up the trunk
            float lob = (noise3(d * 1.3f + c * 0.45f + V{S, 0, 0}) - 0.5f) * 0.5f + (noise3(d * 3.1f + c * 1.3f) - 0.5f) * 0.18f;
            float rad = rr * (1.f + wave * lob);
            if (trunk && p.flare > 0.f) {  // root buttresses: lobes that fan out into the ground
                float roots = powf(fabsf(cosf(ang * 2.5f + S)), 3.f) * 0.7f + 0.3f;
                rad += p.flare * l.ra * expf(-std::max(y - 0.3f, 0.f) / (0.35f + 1.2f * l.ra)) * roots * (0.7f + 0.6f * noise3(d * 2.f + V{S, 3, 0}));
            }
            b.vtx(c + d * rad, (float)i / sides * 2.f, y * 0.5f);
        }
    }
    for (int r = 0; r + 1 < rings; r++)
        for (int i = 0; i < sides; i++) {
            uint32_t q0 = base + r * (sides + 1) + i, q1 = q0 + 1, q2 = q0 + sides + 1, q3 = q2 + 1;
            b.tri(q0, q2, q1, mat);
            b.tri(q1, q2, q3, mat);
        }
    b.endPart();
}
void emit(Builder& b, const Limb& l, uint32_t mat, const Plan& p) { wavyTube(b, l, 0.f, 1.f, mat, p, l.parent < 0 && l.level < 0); }
void emit(Builder& b, const Foliage& f, const Plan& p) {
    float leafWidth = p.leafWidth, droop = p.droop;
    Rng r{(uint32_t)(f.seed * 9973.f) + 17u};
    for (int i = 0; i < f.n; i++) {
        if (f.kind == F_FRUIT) {
            V c = f.c + randUnit(r) * (f.spread * 0.5f) - V{0, r.in(0.f, f.spread), 0};
            float s = f.size * r.in(0.8f, 1.2f);
            blob(b, c, {s, s * 1.05f, s}, f.seed + i, p.fruitMat, 1);
            continue;
        }
        V off = randUnit(r) * (f.spread * cbrtf(r()));
        V c = f.c + off;
        V d = norm(f.dir * 0.5f + norm(off + V{0, 0.001f, 0}) + V{0, r.in(-0.1f, 0.4f), 0});
        V up = fabsf(d.y) > 0.95f ? V{1, 0, 0} : V{0, 1, 0};
        up = rot(up, d, r.in(-0.8f, 0.8f));  // twist about the leaf axis
        float L = f.size * r.in(0.75f, 1.25f);
        leaf(b, c, d, up, L, L * leafWidth * 0.5f, droop * r.in(0.5f, 1.3f), p.leafMat);
    }
}
float tubeVolume(const Limb& l) {
    float len = sqrtf(dot(l.b - l.a, l.b - l.a));
    return 1.0471976f * len * (l.ra * l.ra + l.ra * l.rb + l.rb * l.rb);
}
float foliageVolume(const Foliage& f) { return 4.18879f * f.spread * f.spread * f.spread; }

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

constexpr float WOOD_KG = 700.f, LEAF_KG = 6.f;  // per m^3: green wood; a crown's twigs + leaves per m^3 of envelope

// Flat tapering leaf strip from base along dir, sagging by droop * len at the tip (single-sided: the shaders
// flip normals toward the viewer). jag notches the edge every other segment (dandelion leaves).
void ribbon(Builder& b, V base, V dir, float len, float w, float droop, uint32_t mat, bool jag = false) {
    const int N = 6;
    V side = norm(cross(dir, fabsf(dir.y) < 0.99f ? V{0, 1, 0} : V{1, 0, 0}));
    uint32_t first = (uint32_t)(b.m.verts.size() / 8);
    for (int i = 0; i <= N; i++) {
        float t = (float)i / N;
        V c = base + dir * (len * t) + V{0, -droop * len * t * t, 0};
        float wi = w * (t < 0.15f ? 0.4f + 4.f * t : (1.f - t) / 0.85f) * (jag && (i & 1) ? 0.45f : 1.f);
        b.vtx(c - side * wi, 0, t);
        b.vtx(c + side * wi, 1, t);
    }
    for (int i = 0; i < N; i++) {
        uint32_t a = first + 2 * i;
        b.tri(a, a + 2, a + 1, mat);
        b.tri(a + 1, a + 2, a + 3, mat);
    }
    b.endPart();
}

// plant materials (indices into buildPlants' material list)
enum : uint32_t { P_STEM, P_CATTAIL, P_LEAF, P_YELLOW, P_PUFF, P_ROSE, P_ROSELEAF, P_BUSH_A, P_BUSH_B, P_BLOSSOM, P_SPRAY, P_CLOD, P_WOOD,
                  P_FERN, P_PEBBLE, P_TWIG, P_LITTER, P_BERRYLEAF, P_BERRY };

void cattails(Builder& b, uint32_t seed) {
    Rng rng{seed * 2654435761u + 3u};
    int n = 7 + (int)(rng() * 4);
    for (int i = 0; i < n; i++) {
        float a = rng.in(0, TAU), r = rng.in(0.f, 0.3f), h = rng.in(1.2f, 1.9f);
        V base{cosf(a) * r, -0.05f, sinf(a) * r};
        V top = base + V{rng.in(-0.12f, 0.12f), h, rng.in(-0.12f, 0.12f)};
        V ax = norm(top - base);
        tube(b, base, top, 0.012f, 0.006f, P_STEM, 4);
        V h0 = base + ax * (h * 0.74f), h1 = base + ax * (h * 0.9f);
        tube(b, h0 - ax * 0.01f, h0, 0.008f, 0.034f, P_CATTAIL, 6);
        tube(b, h0, h1, 0.034f, 0.032f, P_CATTAIL, 6);
        tube(b, h1, h1 + ax * 0.012f, 0.032f, 0.006f, P_CATTAIL, 6);
    }
    int nl = 12 + (int)(rng() * 6);
    for (int i = 0; i < nl; i++) {
        float a = rng.in(0, TAU), r = rng.in(0.f, 0.3f);
        V dir = norm({cosf(a), rng.in(3.f, 6.f), sinf(a)});
        ribbon(b, {cosf(a) * r, -0.05f, sinf(a) * r}, dir, rng.in(1.f, 1.7f), 0.022f, rng.in(0.1f, 0.35f), P_LEAF);
    }
}

void dandelions(Builder& b, uint32_t seed) {
    Rng rng{seed * 2246822519u + 5u};
    int nl = 8 + (int)(rng() * 4);
    for (int i = 0; i < nl; i++) {
        float a = TAU * (i + rng.in(-0.3f, 0.3f)) / nl;
        ribbon(b, {0, 0.01f, 0}, norm({cosf(a), rng.in(0.15f, 0.4f), sinf(a)}), rng.in(0.16f, 0.26f), 0.03f, 0.25f, P_LEAF, true);
    }
    int nf = 2 + (int)(rng() * 3);
    for (int i = 0; i < nf; i++) {
        float a = rng.in(0, TAU);
        V base{cosf(a) * 0.03f, 0, sinf(a) * 0.03f};
        V top = base + V{cosf(a) * rng.in(0.02f, 0.08f), rng.in(0.18f, 0.34f), sinf(a) * rng.in(0.02f, 0.08f)};
        tube(b, base, top, 0.005f, 0.004f, P_STEM, 4);
        if (i == 0 && rng() < 0.6f) blob(b, top + V{0, 0.03f, 0}, {0.04f, 0.04f, 0.04f}, seed + i * 7.f, P_PUFF, 1);  // gone to seed
        else blob(b, top + V{0, 0.008f, 0}, {0.032f, 0.016f, 0.032f}, seed + i * 7.f, P_YELLOW, 1);
    }
}

// Generated bush (growBush) emitted into a plant mesh: wood in twig brown.
void bush(Builder& b, const BushParams& B, uint32_t seed, uint32_t leafMat, uint32_t fruitMat) {
    Plan p = growBush(B, B.seed * 7919u + seed, leafMat, P_TWIG, fruitMat);
    for (auto& l : p.branches) emit(b, l, p.woodMat, p);
    for (auto& f : p.leaves) emit(b, f, p);
}

// Fern: a crown of pinnate fronds. Each frond is a thin rachis rising steeply then arching over (bare stipe at the
// base), carrying paired leaflets (pinnae) that angle toward the tip and follow a lanceolate size profile
// (short at the base, longest ~40% up, tapering to a point). Inner fronds stand taller, outer ones splay and
// droop; a few curled fiddleheads sit in the middle and one old frond lies browned on the ground.
void fern(Builder& b, uint32_t seed) {
    Rng rng{seed * 2654435761u + 3u};
    auto frond = [&](float a, float len, float rise, float droop, uint32_t mat, uint32_t stemMat) {
        V h{cosf(a), 0, sinf(a)}, up{0, 1, 0};
        auto P = [&](float t) { return h * (len * (0.55f * t + 0.1f * t * t)) + up * (len * (rise * t - droop * t * t)) + V{0, 0.01f, 0}; };
        const int S = 10;
        for (int i = 0; i < S; i++) {
            float t0 = (float)i / S, t1 = (float)(i + 1) / S;
            tube(b, P(t0), P(t1), 0.006f * (1.f - 0.7f * t0), 0.006f * (1.f - 0.7f * t1), stemMat, 3);
        }
        int pairs = 14 + (int)(rng() * 6);
        float Lmax = len * rng.in(0.16f, 0.21f);
        for (int k = 0; k < pairs; k++) {
            float t = 0.22f + 0.76f * (k + 0.5f) / pairs;  // stipe: bare lower 22%
            float u = (t - 0.22f) / 0.78f;
            float L = Lmax * powf(std::max(u, 0.f), 0.45f) * (1.f - u) * 1.9f + 0.012f;
            V c = P(t), tan = norm(P(std::min(t + 0.02f, 1.f)) - P(std::max(t - 0.02f, 0.f)));
            V sd = norm(cross(tan, up)), nrm = norm(cross(sd, tan));
            for (int s = -1; s <= 1; s += 2) {
                float ang = rng.in(0.95f, 1.15f);  // ~60 deg off the rachis, toward the tip
                V d = norm(tan * cosf(ang) + sd * (sinf(ang) * s) + nrm * rng.in(-0.05f, 0.1f));
                V base = c + tan * (s > 0 ? 0.f : 0.25f * len / pairs);  // slightly alternate, not perfectly opposite
                leaf(b, base, d, nrm, L, L * rng.in(0.17f, 0.23f), rng.in(0.1f, 0.3f), mat);
            }
        }
    };
    int n = 8 + (int)(rng() * 5);
    for (int i = 0; i < n; i++) {
        float a = TAU * (i + rng.in(-0.35f, 0.35f)) / n;
        float inner = rng();  // 1: young, upright centre frond; 0: old, splayed outer one
        frond(a, rng.in(0.5f, 0.8f), 0.7f + 0.7f * inner, 0.35f + 0.5f * (1.f - inner), P_FERN, P_FERN);
    }
    // one dead frond lying on the ground
    frond(rng.in(0, TAU), rng.in(0.5f, 0.7f), 0.12f, 0.1f, P_LITTER, P_LITTER);
    // fiddleheads: short stalks ending in a tight spiral
    int nf = 1 + (int)(rng() * 3);
    for (int i = 0; i < nf; i++) {
        float a = rng.in(0, TAU), hgt = rng.in(0.1f, 0.22f);
        V h{cosf(a), 0, sinf(a)}, base = h * 0.03f, top = base + V{0, hgt, 0};
        tube(b, base, top, 0.006f, 0.005f, P_FERN, 3);
        V prev = top;
        for (int j = 1; j <= 10; j++) {
            float th = j * 0.6f, r = 0.035f * expf(-0.12f * j);
            V q = top + h * (r * sinf(th)) + V{0, r * (1.f - cosf(th)), 0};
            tube(b, prev, q, 0.006f * expf(-0.08f * j), 0.006f * expf(-0.08f * (j + 1)), P_FERN, 3);
            prev = q;
        }
    }
}
// A few half-buried pebbles.
void pebbles(Builder& b, uint32_t seed) {
    Rng rng{seed * 2246822519u + 7u};
    int n = 5 + (int)(rng() * 5);
    for (int i = 0; i < n; i++) {
        float a = rng.in(0, TAU), d = rng.in(0.f, 0.35f), r = rng.in(0.025f, 0.08f);
        blob(b, {cosf(a) * d, r * 0.3f, sinf(a) * d}, {r * rng.in(0.9f, 1.3f), r * rng.in(0.6f, 0.9f), r * rng.in(0.9f, 1.2f)}, seed + i * 5.f, P_PEBBLE, 1);
    }
}
// Fallen twigs: thin sticks lying on the ground, some with a side branch.
void twigs(Builder& b, uint32_t seed) {
    Rng rng{seed * 3266489917u + 9u};
    int n = 2 + (int)(rng() * 3);
    for (int i = 0; i < n; i++) {
        float a = rng.in(0, TAU), len = rng.in(0.25f, 0.7f), r = rng.in(0.006f, 0.014f);
        V p0{rng.in(-0.2f, 0.2f), r, rng.in(-0.2f, 0.2f)}, dir = norm({cosf(a), rng.in(-0.04f, 0.06f), sinf(a)});
        V p1 = p0 + dir * len;
        tube(b, p0, p1, r, r * 0.6f, P_TWIG, 4);
        if (rng() < 0.6f) {
            V m = p0 + dir * (len * rng.in(0.3f, 0.7f));
            float sa = a + (rng() < 0.5f ? 0.6f : -0.6f);
            tube(b, m, m + V{cosf(sa), 0.02f, sinf(sa)} * (len * 0.35f), r * 0.6f, r * 0.3f, P_TWIG, 3);
        }
    }
}
// Leaf litter: small flat leaves lying in a loose patch.
void litter(Builder& b, uint32_t seed) {
    Rng rng{seed * 2654435761u + 13u};
    int n = 14 + (int)(rng() * 10);
    for (int i = 0; i < n; i++) {
        float a = rng.in(0, TAU), d = sqrtf(rng()) * 0.45f, ang = rng.in(0, TAU);
        V base{cosf(a) * d, 0.008f + 0.004f * i / n, sinf(a) * d};
        ribbon(b, base, norm({cosf(ang), rng.in(0.02f, 0.12f), sinf(ang)}), rng.in(0.05f, 0.09f), 0.035f, rng.in(-0.05f, 0.1f), P_LITTER);
    }
}

void finishMesh(MeshData& m) {
    for (int k = 0; k < 3; k++) { m.bmin[k] = 1e30f; m.bmax[k] = -1e30f; }
    for (size_t i = 0; i < m.verts.size(); i += 8)
        for (int k = 0; k < 3; k++) { m.bmin[k] = std::min(m.bmin[k], m.verts[i + k]); m.bmax[k] = std::max(m.bmax[k], m.verts[i + k]); }
}

uint8_t srgb8(float lin) {
    lin = std::min(std::max(lin, 0.f), 1.f);
    float s = lin <= 0.0031308f ? lin * 12.92f : 1.055f * powf(lin, 1.f / 2.4f) - 0.055f;
    return (uint8_t)(s * 255.f + 0.5f);
}

// bark: vertical furrows; foliage: mottled with darker gaps (multiplied by each material's tint)
TextureData barkTex(const float col[3]) {
    TextureData t;
    t.w = 64; t.h = 128;
    t.rgba.resize(t.w * t.h * 4);
    for (uint32_t y = 0; y < t.h; y++)
        for (uint32_t x = 0; x < t.w; x++) {
            float fx = x / 64.f * 8.f, fy = y / 128.f * 4.f;
            float f = vnoise(fx * 2.f, fy * 0.6f) * 0.6f + vnoise(fx * 6.f, fy * 2.f) * 0.4f;
            float k = 0.55f + 0.6f * f;
            uint8_t* p = &t.rgba[(y * t.w + x) * 4];
            p[0] = srgb8(col[0] * k); p[1] = srgb8(col[1] * k); p[2] = srgb8(col[2] * k); p[3] = 255;
        }
    return t;
}
TextureData leafTex() {
    TextureData t;
    t.w = t.h = 128;
    t.rgba.resize(t.w * t.h * 4);
    for (uint32_t y = 0; y < t.h; y++)
        for (uint32_t x = 0; x < t.w; x++) {
            float fx = x / 128.f * 16.f, fy = y / 128.f * 16.f;
            float f = vnoise(fx, fy) * 0.5f + vnoise(fx * 3.1f, fy * 3.1f) * 0.5f;
            float k = 0.45f + 0.8f * f * f;
            uint8_t* p = &t.rgba[(y * t.w + x) * 4];
            p[0] = p[1] = p[2] = srgb8(std::min(k, 1.f));
            p[3] = 255;
        }
    return t;
}

}  // namespace

void defaults() {
    const int pick[SPECIES] = {0, 3, 1, 4};  // oak, pine, birch, spruce
    for (int s = 0; s < SPECIES; s++) { treePreset(pick[s], treeParams[s]); treeParams[s].seed = 1 + s; }
    BushParams& plain = bushParams[BUSH_PLAIN];
    plain = {};
    plain.fruitDensity = 0.f;
    plain.size = 1.1f;
    BushParams& berry = bushParams[BUSH_BERRY];
    berry = {};
    berry.size = 0.9f; berry.stems = 7; berry.leafColor[0] = 0.06f; berry.leafColor[1] = 0.14f; berry.leafColor[2] = 0.035f;
    berry.fruitDensity = 1.3f; berry.fruitSize = 0.011f;
    berry.fruitColor[0] = 0.04f; berry.fruitColor[1] = 0.03f; berry.fruitColor[2] = 0.16f;  // blueberry
    BushParams& rose = bushParams[BUSH_ROSE];
    rose = {};
    rose.size = 0.7f; rose.stems = 5; rose.spread = 0.4f; rose.leafSize = 0.045f; rose.leafColor[0] = 0.05f; rose.leafColor[1] = 0.11f; rose.leafColor[2] = 0.035f;
    rose.fruitDensity = 0.6f; rose.fruitSize = 0.03f; rose.fruitColor[0] = 0.45f; rose.fruitColor[1] = 0.012f; rose.fruitColor[2] = 0.02f;
    BushParams& fl = bushParams[BUSH_FLOWER];
    fl = {};
    fl.size = 0.9f; fl.leafColor[0] = 0.12f; fl.leafColor[1] = 0.2f; fl.leafColor[2] = 0.06f;
    fl.fruitDensity = 1.f; fl.fruitSize = 0.022f; fl.fruitColor[0] = 0.75f; fl.fruitColor[1] = 0.6f; fl.fruitColor[2] = 0.7f;
}

void treePreset(int i, TreeParams& p) {
    p = {};
    auto col = [](float* c, float r, float g, float b) { c[0] = r; c[1] = g; c[2] = b; };
    switch (i) {
    case 0:  // oak: short bole, wide spreading crown, gnarled
        p.height = 11.f; p.trunkRadius = 0.36f; p.bole = 0.28f; p.gnarl = 0.4f; p.branches = 9; p.children = 5; p.branchAngle = 60.f;
        p.lengthRatio = 0.7f; p.crown = CROWN_HEMISPHERICAL; p.gravity = 0.25f; p.upturn = 0.2f; p.leafSize = 0.13f; p.leafDensity = 1.6f;
        col(p.leafColor, 0.08f, 0.17f, 0.04f);
        break;
    case 1:  // birch: slim, pale bark, light open crown with small leaves
        p.height = 13.f; p.trunkRadius = 0.18f; p.bole = 0.35f; p.gnarl = 0.15f; p.branches = 12; p.children = 4; p.branchAngle = 40.f;
        p.lengthRatio = 0.45f; p.crown = CROWN_TAPERED; p.gravity = 0.35f; p.leafSize = 0.07f; p.leafDensity = 1.2f;
        col(p.leafColor, 0.14f, 0.22f, 0.05f); col(p.barkColor, 0.55f, 0.53f, 0.5f);
        break;
    case 2:  // maple: round dense crown
        p.height = 12.f; p.trunkRadius = 0.3f; p.bole = 0.3f; p.branches = 10; p.children = 5; p.branchAngle = 45.f; p.lengthRatio = 0.6f;
        p.crown = CROWN_SPHERICAL; p.leafSize = 0.16f; p.leafWidth = 0.8f; p.leafDensity = 1.1f;
        col(p.leafColor, 0.12f, 0.2f, 0.045f);
        break;
    case 3:  // pine: tall bole, conical crown of needle tufts
        p.height = 14.f; p.trunkRadius = 0.28f; p.bole = 0.4f; p.gnarl = 0.12f; p.levels = 2; p.branches = 22; p.children = 6;
        p.branchAngle = 80.f; p.angleVar = 8.f; p.lengthRatio = 0.35f; p.childLength = 0.45f; p.crown = CROWN_CONICAL; p.gravity = 0.15f;
        p.upturn = 0.3f; p.leafSize = 0.22f; p.leafWidth = 0.12f; p.leafDensity = 2.f; p.leafSpread = 0.35f; p.leafDroop = 0.05f; p.hueVar = 0.15f;
        col(p.leafColor, 0.04f, 0.1f, 0.045f); col(p.barkColor, 0.2f, 0.12f, 0.08f);
        break;
    case 4:  // spruce: narrow, branches all the way down, drooping
        p.height = 15.f; p.trunkRadius = 0.26f; p.bole = 0.08f; p.gnarl = 0.08f; p.levels = 2; p.branches = 34; p.children = 6;
        p.branchAngle = 85.f; p.angleVar = 6.f; p.lengthRatio = 0.3f; p.childLength = 0.4f; p.crown = CROWN_CONICAL; p.gravity = 0.45f;
        p.upturn = 0.15f; p.leafSize = 0.18f; p.leafWidth = 0.1f; p.leafDensity = 2.f; p.leafSpread = 0.3f; p.leafDroop = 0.05f;
        col(p.leafColor, 0.03f, 0.08f, 0.05f); col(p.barkColor, 0.13f, 0.09f, 0.07f);
        break;
    case 5:  // willow: weeping
        p.height = 10.f; p.trunkRadius = 0.34f; p.bole = 0.25f; p.gnarl = 0.3f; p.branches = 10; p.children = 6; p.branchAngle = 35.f;
        p.lengthRatio = 0.8f; p.childLength = 0.8f; p.crown = CROWN_SPHERICAL; p.gravity = 1.2f; p.upturn = 0.05f; p.leafSize = 0.1f;
        p.leafWidth = 0.25f; p.leafDensity = 1.4f; p.leafDroop = 0.5f;
        col(p.leafColor, 0.12f, 0.2f, 0.06f);
        break;
    case 7:  // fir: dense narrow cone, upturned branch tips
        p.height = 16.f; p.trunkRadius = 0.27f; p.bole = 0.06f; p.gnarl = 0.06f; p.levels = 2; p.branches = 40; p.children = 7;
        p.branchAngle = 75.f; p.angleVar = 5.f; p.lengthRatio = 0.28f; p.childLength = 0.45f; p.crown = CROWN_CONICAL; p.gravity = 0.1f;
        p.upturn = 0.45f; p.leafSize = 0.16f; p.leafWidth = 0.1f; p.leafDensity = 2.4f; p.leafSpread = 0.3f; p.leafDroop = 0.03f;
        col(p.leafColor, 0.025f, 0.075f, 0.045f); col(p.barkColor, 0.17f, 0.15f, 0.14f);
        break;
    case 8:  // cypress: tall flame, branches hugging the trunk
        p.height = 14.f; p.trunkRadius = 0.22f; p.bole = 0.05f; p.gnarl = 0.1f; p.levels = 2; p.branches = 36; p.children = 6;
        p.branchAngle = 18.f; p.angleVar = 6.f; p.lengthRatio = 0.22f; p.childLength = 0.5f; p.crown = CROWN_FLAME; p.gravity = 0.f;
        p.upturn = 0.6f; p.leafSize = 0.12f; p.leafWidth = 0.15f; p.leafDensity = 2.6f; p.leafSpread = 0.3f; p.leafDroop = 0.05f;
        col(p.leafColor, 0.03f, 0.08f, 0.04f); col(p.barkColor, 0.22f, 0.13f, 0.09f);
        break;
    case 9:  // acacia: crooked trunk, flat umbrella crown of tiny leaves
        p.height = 8.f; p.trunkRadius = 0.25f; p.bole = 0.45f; p.lean = 0.25f; p.gnarl = 0.55f; p.branches = 6; p.children = 6;
        p.branchAngle = 55.f; p.lengthRatio = 1.f; p.childLength = 0.7f; p.crown = CROWN_INVERSE; p.gravity = 0.f; p.upturn = 0.05f;
        p.leafSize = 0.05f; p.leafWidth = 0.6f; p.leafDensity = 2.f; p.leafSpread = 0.5f; p.leafDroop = 0.1f;
        col(p.leafColor, 0.1f, 0.16f, 0.05f); col(p.barkColor, 0.2f, 0.17f, 0.13f);
        break;
    case 10:  // cherry: low spreading, pink blossom
        p.height = 7.f; p.trunkRadius = 0.22f; p.bole = 0.25f; p.gnarl = 0.35f; p.branches = 7; p.children = 6; p.branchAngle = 50.f;
        p.lengthRatio = 0.75f; p.crown = CROWN_HEMISPHERICAL; p.gravity = 0.15f; p.leafSize = 0.07f; p.leafWidth = 0.8f; p.leafDensity = 1.8f;
        col(p.leafColor, 0.75f, 0.42f, 0.5f); col(p.barkColor, 0.18f, 0.09f, 0.08f);
        break;
    case 11:  // beech: smooth grey bark, tall dense dome
        p.height = 16.f; p.trunkRadius = 0.35f; p.bole = 0.35f; p.gnarl = 0.15f; p.trunkWave = 0.25f; p.branches = 11; p.children = 6;
        p.branchAngle = 45.f; p.lengthRatio = 0.6f; p.crown = CROWN_SPHERICAL; p.leafSize = 0.1f; p.leafWidth = 0.6f; p.leafDensity = 1.7f;
        col(p.leafColor, 0.1f, 0.2f, 0.04f); col(p.barkColor, 0.35f, 0.34f, 0.32f);
        break;
    case 12:  // ash: open, upswept, compound leaves (narrow)
        p.height = 15.f; p.trunkRadius = 0.3f; p.bole = 0.4f; p.branches = 9; p.children = 4; p.branchAngle = 35.f; p.lengthRatio = 0.55f;
        p.crown = CROWN_TAPERED; p.upturn = 0.45f; p.gravity = 0.1f; p.leafSize = 0.12f; p.leafWidth = 0.3f; p.leafDensity = 1.4f;
        col(p.leafColor, 0.09f, 0.18f, 0.05f); col(p.barkColor, 0.28f, 0.26f, 0.22f);
        break;
    case 13:  // dead tree: bare, twisted, grey
        p.height = 9.f; p.trunkRadius = 0.3f; p.bole = 0.3f; p.gnarl = 0.6f; p.lean = 0.2f; p.levels = 2; p.branches = 7; p.children = 4;
        p.branchAngle = 50.f; p.lengthRatio = 0.6f; p.crown = CROWN_SPHERICAL; p.leafDensity = 0.f; p.trunkWave = 0.9f;
        col(p.barkColor, 0.3f, 0.28f, 0.25f);
        break;
    default:  // poplar: columnar
        p.height = 17.f; p.trunkRadius = 0.28f; p.bole = 0.15f; p.branches = 18; p.children = 4; p.branchAngle = 22.f; p.angleVar = 8.f;
        p.lengthRatio = 0.25f; p.crown = CROWN_CYLINDRICAL; p.gravity = 0.05f; p.upturn = 0.4f; p.leafSize = 0.1f; p.leafWidth = 0.8f;
        col(p.leafColor, 0.1f, 0.19f, 0.05f);
        break;
    }
    // green wood (kg/m^3, modulus of rupture MPa): oak birch maple pine spruce willow poplar fir cypress acacia
    // cherry beech ash, then a dead tree: dried out and rotten, light and brittle
    static const float wood[TREE_PRESET_COUNT][2] = {{1000, 57}, {900, 44}, {900, 53}, {800, 35}, {700, 34}, {900, 33}, {800, 36},
                                                     {750, 38}, {800, 43}, {1100, 70}, {900, 55}, {1000, 59}, {900, 66}, {450, 18}};
    if (i >= 0 && i < TREE_PRESET_COUNT) { p.density = wood[i][0]; p.strength = wood[i][1]; }
    p.seed = 1;
}

namespace {
uint32_t variantSeed(int v) { return treeParams[v / 2].seed * 7919u + (uint32_t)v * 101u + 13u; }
// generated once per build() (impacts query limbs, breaks need the parts)
Plan planCache[VARIANTS];
bool planValid[VARIANTS] = {};
const Plan& plan(int v) {
    if (!planValid[v]) { planCache[v] = growTree(treeParams[v / 2], variantSeed(v), 1 + v); planValid[v] = true; }
    return planCache[v];
}
uint32_t barkMat(int v) { return v < 2 ? 0u : MAT_WOOD + (uint32_t)(v / 2); }

// The tree as beams (Wood mechanics): trunk elements, then one per limb segment (parents before children).
struct Elem { V a, b; float ra, rb; int parent, level; float t0, t1, mass; };  // t0..t1: the part of the trunk
struct Skel { std::vector<Elem> e; std::vector<int> leafElem; std::vector<float> leafMass; };  // leafMass: per leaf (kg)
Skel skelCache[VARIANTS];
bool skelValid[VARIANTS] = {};
const Skel& skel(int v) {
    if (skelValid[v]) return skelCache[v];
    const Plan& p = plan(v);
    float dens = treeParams[v / 2].density;
    Skel& S = skelCache[v];
    S = {};
    const Limb& T = p.trunk;
    V ax = T.b - T.a;
    for (int k = 0; k < TRUNK_ELEMS; k++) {
        float t0 = (float)k / TRUNK_ELEMS, t1 = (k + 1.f) / TRUNK_ELEMS;
        Elem e{T.a + ax * t0, T.a + ax * t1, T.ra + (T.rb - T.ra) * t0, T.ra + (T.rb - T.ra) * t1, k - 1, -1, t0, t1, 0.f};
        if (k == 0) e.ra *= 1.f + 0.3f * p.flare;  // root buttresses
        S.e.push_back(e);
    }
    auto trunkElem = [&](V q) { return std::clamp((int)(dot(q - T.a, ax) / std::max(dot(ax, ax), 1e-6f) * TRUNK_ELEMS), 0, TRUNK_ELEMS - 1); };
    for (const Limb& l : p.branches) S.e.push_back({l.a, l.b, l.ra, l.rb, l.parent >= 0 ? TRUNK_ELEMS + l.parent : trunkElem(l.a), l.level, -1.f, -1.f, 0.f});
    for (Elem& e : S.e) e.mass = tubeVolume({e.a, e.b, e.ra, e.rb, 0, 0}) * dens;
    for (const Foliage& f : p.leaves) {
        S.leafElem.push_back(f.owner >= 0 ? TRUNK_ELEMS + f.owner : trunkElem(f.c));
        S.leafMass.push_back(foliageVolume(f) * LEAF_KG / std::max(f.n, 1));  // twigs and all
    }
    skelValid[v] = true;
    return S;
}
}  // namespace

void build(ModelData& out) {
    for (bool& b : planValid) b = false;
    for (bool& b : skelValid) b = false;
    out = {};
    out.textures = {barkTex(treeParams[0].barkColor), leafTex()};  // species s > 0 bark: texture 1 + s
    for (int s = 1; s < SPECIES; s++) out.textures.push_back(barkTex(treeParams[s].barkColor));
    MaterialData bark;  // material 0: species A bark; B's bark goes after the wood (MAT_WOOD + 1)
    bark.tex = 0;
    out.materials.push_back(bark);
    for (int v = 0; v < VARIANTS; v++) {
        const TreeParams& P = treeParams[v / 2];
        Rng r{variantSeed(v) * 31u + 5u};
        MaterialData m;
        float h = P.hueVar * r.in(-0.5f, 0.5f);  // shift toward yellow / blue green
        m.color[0] = P.leafColor[0] * (1.f + h);
        m.color[1] = P.leafColor[1] * (1.f + 0.3f * h);
        m.color[2] = P.leafColor[2] * (1.f - h);
        m.color[3] = 1;
        m.tex = 1;
        m.translucency = 0.45f;  // leaves glow when backlit
        m.rough = 0.55f;         // waxy cuticle
        out.materials.push_back(m);
    }
    MaterialData wood;  // fresh wood where a trunk or branch snapped (MAT_WOOD)
    wood.color[0] = 0.55f; wood.color[1] = 0.40f; wood.color[2] = 0.24f;
    out.materials.push_back(wood);
    for (int s = 1; s < SPECIES; s++) {  // other species' bark after the wood (MAT_WOOD + s)
        MaterialData bk;
        bk.tex = 1 + s;
        out.materials.push_back(bk);
    }
    out.meshes.resize(VARIANTS);
    for (int v = 0; v < VARIANTS; v++) {
        MeshData& m = out.meshes[v];
        Builder b{m};
        const Plan& p = plan(v);
        uint32_t bm = barkMat(v);
        emit(b, p.trunk, bm, p);
        for (auto& l : p.branches) emit(b, l, bm, p);
        for (auto& f : p.leaves) emit(b, f, p);
        finishMesh(m);
    }
}

void buildPlants(ModelData& out) {
    out = {};
    out.textures = {leafTex()};
    const float cols[][3] = {{0.09f, 0.16f, 0.04f}, {0.13f, 0.06f, 0.025f}, {0.11f, 0.2f, 0.05f}, {0.8f, 0.55f, 0.02f}, {0.75f, 0.75f, 0.7f},
                             {0.45f, 0.012f, 0.02f}, {0.05f, 0.11f, 0.035f}, {0.08f, 0.17f, 0.04f}, {0.12f, 0.2f, 0.06f}, {0.75f, 0.6f, 0.7f}, {0.82f, 0.86f, 0.9f}, {0.11f, 0.08f, 0.05f}, {0.32f, 0.22f, 0.12f},
                             {0.06f, 0.14f, 0.035f}, {0.20f, 0.19f, 0.17f}, {0.12f, 0.085f, 0.055f}, {0.30f, 0.16f, 0.06f}};
    auto bushCol = [&](int i) -> const float* {  // bush leaf / fruit colours come from their sliders
        if (i == P_BUSH_A) return bushParams[BUSH_PLAIN].leafColor;
        if (i == P_BUSH_B) return bushParams[BUSH_FLOWER].leafColor;
        if (i == P_BLOSSOM) return bushParams[BUSH_FLOWER].fruitColor;
        if (i == P_ROSELEAF) return bushParams[BUSH_ROSE].leafColor;
        if (i == P_ROSE) return bushParams[BUSH_ROSE].fruitColor;
        if (i == P_BERRYLEAF) return bushParams[BUSH_BERRY].leafColor;
        if (i == P_BERRY) return bushParams[BUSH_BERRY].fruitColor;
        return cols[i];
    };
    for (int i = 0; i <= (int)P_BERRY; i++) {
        MaterialData m;
        memcpy(m.color, bushCol(i), 12);
        bool leafy = i == P_LEAF || i == P_ROSELEAF || i == P_BUSH_A || i == P_BUSH_B || i == P_BERRYLEAF;
        m.tex = leafy ? 0 : -1;
        if (leafy || i == P_STEM || i == P_BLOSSOM || i == P_FERN || i == P_LITTER) m.translucency = 0.4f;  // thin leaves, stems, petals pass light
        if (i == P_PEBBLE) m.rough = 0.7f;  // weathered
        if (i == P_BERRY || i == P_ROSE) { m.rough = 0.25f; m.translucency = 0.25f; }  // glossy, skin lets light in
        if (i == P_FERN || i == P_LITTER) m.tex = 0;
        out.materials.push_back(m);
    }
    out.meshes.resize(PLANT_VARIANTS);
    for (int v = 0; v < PLANT_VARIANTS; v++) {
        MeshData& m = out.meshes[v];
        Builder b{m};
        if (v == PLANT_CATTAIL) cattails(b, 41u);
        else if (v == PLANT_DANDELION) dandelions(b, 53u);
        else if (v == PLANT_ROSE) bush(b, bushParams[BUSH_ROSE], 67u, P_ROSELEAF, P_ROSE);
        else if (v == PLANT_BUSH) bush(b, bushParams[BUSH_PLAIN], 79u, P_BUSH_A, 0);
        else if (v == PLANT_FLOWERBUSH) bush(b, bushParams[BUSH_FLOWER], 97u, P_BUSH_B, P_BLOSSOM);
        else if (v == PLANT_BERRYBUSH) bush(b, bushParams[BUSH_BERRY], 113u, P_BERRYLEAF, P_BERRY);
        else if (v == PLANT_SPRAY) blob(b, {0, 0, 0}, {1, 1, 1}, 5.f, P_SPRAY, 1);
        else if (v == PLANT_CLOD) blob(b, {0, 0, 0}, {1, 0.8f, 1}, 9.f, P_CLOD, 1);
        else if (v == PLANT_SPLINTER) tube(b, {0, -1, 0}, {0.1f, 1, 0}, 0.22f, 0.08f, P_WOOD, 5);
        else if (v == PLANT_LEAF) ribbon(b, {0, 0, -1}, {0, 0, 1}, 2.f, 0.5f, 0.15f, P_LEAF);
        else if (v == PLANT_FERN) fern(b, 131u);
        else if (v == PLANT_PEBBLES) pebbles(b, 151u);
        else if (v == PLANT_TWIGS) twigs(b, 173u);
        else litter(b, 191u);
        finishMesh(m);
    }
}

namespace {
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
bool colliderLimb(const Limb& l) { return l.level == 0 || (l.level >= 1 && l.ra >= 0.035f); }
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
}  // namespace

void trunkCollider(const ModelData&, int v, std::vector<float>& verts, std::vector<uint32_t>& idx) { colliderOf(plan(v), nullptr, verts, idx); }

namespace {
constexpr float GRAV = 9.8f;
float elemDist(const Elem& e, V q, float& t) { return segDist(q, e.a, e.b, t) - (e.ra + (e.rb - e.ra) * t); }
int nearestElem(const Wood& w, const Skel& S, V q, float& dist, float& tHit) {
    int best = -1;
    dist = 1e30f;
    for (size_t k = 0; k < S.e.size(); k++) {
        if (!w.has[k]) continue;
        float t, d = elemDist(S.e[k], q, t);
        if (d < dist) { dist = d; tHit = t; best = (int)k; }
    }
    return best;
}
bool isCollider(const Skel& S, int k) { return k < TRUNK_ELEMS || S.e[k].level == 0 || S.e[k].ra >= 0.035f; }
// mass (kg) and first moment (kg m) of each element plus everything of w hanging off it, at scale s (metres)
struct Loads { std::vector<float> m; std::vector<V> mom; float M = 0.f; V com{0, 0, 0}; };
void loads(const Wood& w, const Skel& S, float s, Loads& L) {
    size_t n = S.e.size();
    float s3 = s * s * s;
    L.m.assign(n, 0.f);
    L.mom.assign(n, V{0, 0, 0});
    for (size_t i = 0; i < n; i++)
        if (w.has[i]) { L.m[i] = S.e[i].mass * s3; L.mom[i] = (S.e[i].a + S.e[i].b) * (0.5f * s * L.m[i]); }
    const Plan& p = plan(w.v);
    for (size_t i = 0; i < S.leafElem.size(); i++) {
        int k = S.leafElem[i];
        if (!w.has[k]) continue;
        float m = S.leafMass[i] * w.leaves[i] * s3;
        L.m[k] += m;
        L.mom[k] = L.mom[k] + p.leaves[i].c * (s * m);
    }
    for (size_t i = n; i-- > 0;) {
        int pa = S.e[i].parent;
        if (w.has[i] && pa >= 0 && w.has[pa]) { L.m[pa] += L.m[i]; L.mom[pa] = L.mom[pa] + L.mom[i]; }
    }
    L.M = 0.f;
    V mm{0, 0, 0};
    for (size_t i = 0; i < n; i++)
        if (w.has[i] && (S.e[i].parent < 0 || !w.has[S.e[i].parent])) { L.M += L.m[i]; mm = mm + L.mom[i]; }
    L.com = mm * (1.f / std::max(L.M, 1e-6f));
}
// stress / strength at a round section of radius r (m), axis ax, carrying moment M (N m): bending + torsion
float sectionRatio(V M, V ax, float r, float strength) {
    ax = norm(ax);
    float t = dot(M, ax);
    V mb = M - ax * t;
    float r3 = PI * r * r * r, sig = 4.f * sqrtf(dot(mb, mb)) / r3, tau = 2.f * fabsf(t) / r3;
    return sqrtf(sig * sig + 3.f * tau * tau) / strength;
}
// a limb's union with its parent is a weak spot (grain turns, included bark)
float jointFactor(const Skel& S, int k) { int pa = S.e[k].parent; return pa >= 0 && S.e[pa].level != S.e[k].level ? 0.85f : 1.f; }
// green twigs this thin bend out of the way instead of snapping (huge strain before failure)
constexpr float MIN_SNAP_R = 0.012f;
// inertia (kg m^2) about the centre of mass: elements as rods, leaf clusters as points
void inertia(const Wood& w, const Skel& S, float s, const Loads& L, float I[3][3]) {
    const Plan& p = plan(w.v);
    size_t n = S.e.size();
    float s3 = s * s * s;
    memset(I, 0, 9 * sizeof(float));
    auto addI = [&](V r, float m) {
        float rv[3] = {r.x, r.y, r.z}, rr = dot(r, r);
        for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) I[a][b] += m * ((a == b ? rr : 0.f) - rv[a] * rv[b]);
    };
    for (size_t k = 0; k < n; k++) {
        if (!w.has[k]) continue;
        const Elem& E = S.e[k];
        float m = E.mass * s3;
        V d = (E.b - E.a) * s, x = (E.a + E.b) * (0.5f * s);
        addI(x - L.com, m);
        float dv[3] = {d.x, d.y, d.z}, dd = dot(d, d);
        for (int a = 0; a < 3; a++) for (int b = 0; b < 3; b++) I[a][b] += m / 12.f * ((a == b ? dd : 0.f) - dv[a] * dv[b]);
    }
    for (size_t i = 0; i < S.leafElem.size(); i++)
        if (w.has[S.leafElem[i]]) addI(p.leaves[i].c * s - L.com, S.leafMass[i] * w.leaves[i] * s3);
}
bool inSubtree(const Skel& S, int root, int k) {
    for (; k >= root; k = S.e[k].parent) if (k == root) return true;
    return false;
}
}  // namespace

void woodWhole(int v, Wood& w) {
    const Skel& S = skel(v);
    const Plan& p = plan(v);
    w = {};
    w.v = v;
    w.has.assign(S.e.size(), 1);
    for (const Foliage& f : p.leaves) w.leaves.push_back((uint16_t)std::min(f.n, 65535));
}

int woodNearest(const Wood& w, const float p[3], float& dist) {
    float t;
    return nearestElem(w, skel(w.v), V{p[0], p[1], p[2]}, dist, t);
}

bool woodElem(const Wood& w, int k, float a[3], float b[3], float& r) {
    const Skel& S = skel(w.v);
    if (k < 0 || k >= (int)S.e.size() || k >= (int)w.has.size() || !w.has[k]) return false;
    const Elem& E = S.e[k];
    memcpy(a, &E.a, 12); memcpy(b, &E.b, 12);
    r = E.ra;
    return true;
}

int woodSweep(const Wood& w, const float pa[3], const float pb[3], float r, float at[3]) {
    const Skel& S = skel(w.v);
    if (w.has.size() != S.e.size()) return -1;
    V a{pa[0], pa[1], pa[2]}, d = V{pb[0], pb[1], pb[2]} - a;
    float len = sqrtf(dot(d, d)), best = 2.f;
    int hit = -1;
    if (len < 1e-5f) return -1;
    int steps = std::clamp((int)(len / std::max(r, 0.02f)) + 1, 2, 64);
    for (size_t k = TRUNK_ELEMS; k < S.e.size(); k++) {
        if (!w.has[k] || isCollider(S, (int)k)) continue;
        const Elem& E = S.e[k];
        // cheap reject: element bounding sphere vs the swept segment
        V c = (E.a + E.b) * 0.5f;
        float er = sqrtf(dot(E.b - E.a, E.b - E.a)) * 0.5f + E.ra + r, u = std::clamp(dot(c - a, d) / (len * len), 0.f, 1.f);
        V q = a + d * u - c;
        if (dot(q, q) > er * er) continue;
        for (int i = 0; i <= steps; i++) {  // first sample along the path that touches the limb
            float s = (float)i / steps;
            if (s >= best) break;
            float t;
            if (elemDist(E, a + d * s, t) < r) { best = s; hit = (int)k; break; }
        }
    }
    if (hit >= 0) {
        float t;
        V p = a + d * best;
        elemDist(S.e[hit], p, t);
        V on = S.e[hit].a + (S.e[hit].b - S.e[hit].a) * t;
        at[0] = on.x; at[1] = on.y; at[2] = on.z;
    }
    return hit;
}

void woodMass(const Wood& w, float s, float& mass, float com[3], float I[9]) {
    const Skel& S = skel(w.v);
    Loads L;
    loads(w, S, s, L);
    float J[3][3];
    inertia(w, S, s, L, J);
    mass = L.M;
    memcpy(com, &L.com, 12);
    memcpy(I, J, sizeof J);
}

void woodSupport(const Wood& w, const float dir[3], float out[3]) {
    const Skel& S = skel(w.v);
    V d = norm(V{dir[0], dir[1], dir[2]}), best{0, 0, 0};
    float bd = 1e30f;
    for (size_t k = 0; k < S.e.size(); k++) {
        if (!w.has[k]) continue;
        const Elem& E = S.e[k];
        for (int i = 0; i < 2; i++) {
            V q = (i ? E.b : E.a) - d * (i ? E.rb : E.ra);
            if (dot(q, d) < bd) { bd = dot(q, d); best = q; }
        }
    }
    out[0] = best.x; out[1] = best.y; out[2] = best.z;
}

WoodHit woodStrike(const Wood& w, float s, const float pp[3], const float vel[3], float m) {
    WoodHit h;
    const Skel& S = skel(w.v);
    const Plan& p = plan(w.v);
    const TreeParams& P = treeParams[w.v / 2];
    h.lose.assign(S.leafElem.size(), 0);
    float strength = P.strength * 1e6f, Emod = 9e9f * std::clamp(P.strength / 50.f, 0.4f, 1.6f), G = Emod / 16.f, s3 = s * s * s;
    float dist, tHit;
    int e = nearestElem(w, S, V{pp[0], pp[1], pp[2]}, dist, tHit);
    if (e < 0) return h;
    V q = V{pp[0], pp[1], pp[2]} * s;  // metres from here on
    const Elem& E = S.e[e];
    // only the velocity across the struck limb bends it: a glancing blow along the wood slides off
    V ax = norm(E.b - E.a), u{vel[0], vel[1], vel[2]};
    V un = u - ax * dot(u, ax);
    float vn = sqrtf(dot(un, un));
    if (vn < 0.05f) return h;
    V f = un * (1.f / vn);
    Loads L;
    loads(w, S, s, L);
    // effective mass the impactor meets: ~a quarter of the limb it hit (its chain of segments from the joint,
    // with all that grows on it), the whole tree for the trunk
    int j = e;
    while (S.e[j].parent >= 0 && w.has[S.e[j].parent] && S.e[S.e[j].parent].level == E.level) j = S.e[j].parent;
    float meff = std::max(0.25f * L.m[j] + 0.5f * E.mass * s3, 0.02f);
    float mu = m * meff / (m + meff), Ed = 0.5f * mu * vn * vn;
    // compliance at the contact (unit-load method): bending + torsion of every element from there to the roots,
    // plus local crushing of bark and wood under the impactor
    float c = 1.f / 5e7f;
    for (int k = e; k >= 0 && w.has[k]; k = S.e[k].parent) {
        const Elem& K = S.e[k];
        float t1 = k == e ? tHit : 1.f;
        V a = K.a * s, b = K.b * s, kax = norm(b - a);
        float ds = sqrtf(dot(b - a, b - a)) * t1 / 4.f;
        for (int i = 0; i < 4; i++) {
            float t = t1 * (i + 0.5f) / 4.f, r = (K.ra + (K.rb - K.ra) * t) * s;
            V mv = cross(q - (a + (b - a) * t), f);
            float tq = dot(mv, kax), I = PI * r * r * r * r / 4.f;
            V mb = mv - kax * tq;
            c += (dot(mb, mb) / (Emod * I) + tq * tq / (G * 2.f * I)) * ds;
        }
    }
    float F = sqrtf(2.f * Ed / c), J = 1.3f * mu * vn;  // peak force; momentum exchanged (restitution ~0.3)
    // every joint on the way down: impact moment + the weight hanging outboard; the tree already carries its
    // own weight, so that share never breaks it by itself
    int cut = -1;
    for (int k = e; k >= 0 && w.has[k]; k = S.e[k].parent) {
        const Elem& K = S.e[k];
        V a = K.a * s;
        float r = K.ra * s;
        V Mg = cross(L.mom[k] - a * L.m[k], V{0, -GRAV, 0}), Mi = cross(q - a, f * F);
        float sg = sectionRatio(Mg, K.b - K.a, r, strength);
        float rt = sectionRatio(Mi + Mg, K.b - K.a, r, strength * jointFactor(S, k)) / std::max(1.f, 1.25f * sg);
        if (rt > h.ratio && r >= MIN_SNAP_R) { h.ratio = rt; cut = k; }
    }
    // A fast blow on a light limb snaps it right where it lands, before the limb as a whole can flex: the span
    // the bending wave reaches during the contact (~10 diameters) acts as a beam between its own inertia.
    {
        float r = (E.ra + (E.rb - E.ra) * tHit) * s, span = 20.f * r, I = PI * r * r * r * r / 4.f;
        float mloc = treeParams[w.v / 2].density * PI * r * r * span, muL = m * mloc / (m + mloc);
        float cl = span * span * span / (48.f * Emod * I) + 1.f / 5e7f;
        float Fl = vn * sqrtf(muL / cl), rt = Fl * span / 4.f * 4.f / (PI * r * r * r) / (strength * 1.1f);
        if (rt > h.ratio && r >= MIN_SNAP_R) {
            h.ratio = rt;
            cut = e;  // at the joint nearer the blow
            if (tHit > 0.5f)
                for (size_t k = e + 1; k < S.e.size(); k++)
                    if (S.e[k].parent == e && S.e[k].level == E.level && w.has[k]) { cut = (int)k; break; }
        }
    }
    float took = J;
    if (h.ratio > 1.f) {
        h.cuts.push_back(std::max(cut, 1));  // the trunk snaps above the root plate
        h.keep = sqrtf(std::max(1.f - 1.f / (h.ratio * h.ratio), 0.f));
        took = m * vn * (1.f - h.keep);
    }
    h.impulse[0] = f.x * took; h.impulse[1] = f.y * took; h.impulse[2] = f.z * took;
    // leaves near the blow are jerked off their twigs
    float dv0 = std::min(J / meff, 1.3f * vn);
    for (size_t i = 0; i < S.leafElem.size(); i++) {
        if (!w.has[S.leafElem[i]] || !w.leaves[i]) continue;
        V d = p.leaves[i].c * s - q;
        float shake = dv0 * expf(-sqrtf(dot(d, d)) / 1.2f), frac = std::clamp((shake - 1.f) * 0.05f, 0.f, 0.8f);
        h.lose[i] = (uint16_t)(w.leaves[i] * frac + 0.5f);
        h.lost += h.lose[i];
    }
    return h;
}

WoodHit woodLanding(const Wood& w, float s, const float pp[3], const float Jp[3]) {
    WoodHit h;
    const Skel& S = skel(w.v);
    const Plan& p = plan(w.v);
    float strength = treeParams[w.v / 2].strength * 1e6f, s3 = s * s * s;
    size_t n = S.e.size();
    h.lose.assign(S.leafElem.size(), 0);
    Loads L;
    loads(w, S, s, L);
    if (L.M <= 0.f) return h;
    V q = V{pp[0], pp[1], pp[2]} * s;
    constexpr float TAU_C = 0.01f;  // contact time of wood landing on ground / rock
    V F = V{Jp[0], Jp[1], Jp[2]} * (1.f / TAU_C);
    float I[3][3];
    inertia(w, S, s, L, I);
    for (int a = 0; a < 3; a++) I[a][a] += 1e-4f * L.M + 1e-6f;
    float det = I[0][0] * (I[1][1] * I[2][2] - I[1][2] * I[2][1]) - I[0][1] * (I[1][0] * I[2][2] - I[1][2] * I[2][0]) + I[0][2] * (I[1][0] * I[2][1] - I[1][1] * I[2][0]);
    V tq = cross(q - L.com, F);
    // alpha = I^-1 tq (Cramer)
    auto col = [&](int c, V r) { float M[3][3]; memcpy(M, I, sizeof M); M[0][c] = r.x; M[1][c] = r.y; M[2][c] = r.z;
        return M[0][0] * (M[1][1] * M[2][2] - M[1][2] * M[2][1]) - M[0][1] * (M[1][0] * M[2][2] - M[1][2] * M[2][0]) + M[0][2] * (M[1][0] * M[2][1] - M[1][1] * M[2][0]); };
    V alpha = fabsf(det) > 1e-12f ? V{col(0, tq), col(1, tq), col(2, tq)} * (1.f / det) : V{0, 0, 0};
    V a0 = F * (1.f / L.M);
    auto acc = [&](V x) { return a0 + cross(alpha, x - L.com); };
    float dist, tHit;
    int e = nearestElem(w, S, V{pp[0], pp[1], pp[2]}, dist, tHit);
    // every joint holds what it takes to drag the part beyond it (the side away from the contact) along
    struct Cand { int k; float r; };
    std::vector<Cand> cand;
    for (size_t k = 0; k < n; k++) {
        int pa = S.e[k].parent;
        if (!w.has[k] || pa < 0 || !w.has[pa]) continue;
        bool contactSide = e >= 0 && inSubtree(S, (int)k, e);
        float ms = contactSide ? L.M - L.m[k] : L.m[k];
        if (ms < 1e-3f) continue;
        V cs = (contactSide ? L.com * L.M - L.mom[k] : L.mom[k]) * (1.f / ms);
        const Elem& K = S.e[k];
        V a = K.a * s;
        float rt = sectionRatio(cross(cs - a, acc(cs) * ms), K.b - K.a, K.ra * s, strength * jointFactor(S, (int)k));
        if (K.ra * s < MIN_SNAP_R) continue;
        h.ratio = std::max(h.ratio, rt);
        if (rt > 1.f) cand.push_back({(int)k, rt});
    }
    std::sort(cand.begin(), cand.end(), [](const Cand& a, const Cand& b) { return a.r > b.r; });
    for (const Cand& c : cand) {
        if (h.cuts.size() >= 3) break;
        bool nested = false;
        for (int k : h.cuts) nested |= inSubtree(S, k, c.k) || inSubtree(S, c.k, k);
        if (!nested) h.cuts.push_back(c.k);
    }
    for (size_t i = 0; i < S.leafElem.size(); i++) {
        if (!w.has[S.leafElem[i]] || !w.leaves[i]) continue;
        V av = acc(p.leaves[i].c * s);
        float dv = sqrtf(dot(av, av)) * TAU_C, frac = std::clamp((dv - 1.5f) * 0.05f, 0.f, 0.6f);
        h.lose[i] = (uint16_t)(w.leaves[i] * frac + 0.5f);
        h.lost += h.lose[i];
    }
    return h;
}

void woodApply(Wood& w, const WoodHit& h, std::vector<Wood>& off, std::vector<float>* shed) {
    const Skel& S = skel(w.v);
    const Plan& p = plan(w.v);
    size_t n = S.e.size();
    Rng rng{(uint32_t)h.lost * 7919u + (uint32_t)w.cuts.size() * 31u + 1u};
    for (size_t i = 0; i < h.lose.size() && i < w.leaves.size(); i++) {
        int k = std::min<int>(h.lose[i], w.leaves[i]);
        if (!k) continue;
        w.leaves[i] = (uint16_t)(w.leaves[i] - k);
        const Foliage& f = p.leaves[i];
        if (shed)
            for (int j = 0; j < k; j++) {
                V q = f.c + randUnit(rng) * (f.spread * cbrtf(rng()));
                shed->insert(shed->end(), {q.x, q.y, q.z});
            }
    }
    for (int c : h.cuts) {
        if (c < 0 || c >= (int)n || !w.has[c]) continue;
        Wood o;
        o.v = w.v;
        o.anchored = false;
        o.leaves = w.leaves;
        o.has.assign(n, 0);
        for (size_t i = c; i < n; i++)  // parents come first: one pass collects the subtree
            o.has[i] = w.has[i] && ((int)i == c || (S.e[i].parent >= 0 && o.has[S.e[i].parent]));
        for (size_t i = 0; i < n; i++) if (o.has[i]) w.has[i] = 0;
        o.cuts = {c};
        for (int k : w.cuts)  // older breaks inside the piece: their splinters and stubs go with it
            if (k != c && (o.has[k] || (S.e[k].parent >= 0 && o.has[S.e[k].parent]))) o.cuts.push_back(k);
        w.cuts.push_back(c);
        off.push_back(std::move(o));
    }
}

void woodMesh(const Wood& w, uint32_t seed, Piece& out) {
    out = {};
    const Skel& S = skel(w.v);
    const Plan& p = plan(w.v);
    uint32_t bm = barkMat(w.v);
    Builder b{out.mesh};
    for (size_t k = 0; k < S.e.size(); k++) {
        if (!w.has[k]) continue;
        const Elem& E = S.e[k];
        out.mass += E.mass;
        if (k >= (size_t)TRUNK_ELEMS) { emit(b, p.branches[k - TRUNK_ELEMS], bm, p); continue; }
        size_t j = k;  // a run of trunk elements is one tube (no seams)
        while (j + 1 < (size_t)TRUNK_ELEMS && w.has[j + 1]) out.mass += S.e[++j].mass;
        wavyTube(b, p.trunk, E.t0, S.e[j].t1, bm, p, true);
        k = j;
    }
    for (size_t i = 0; i < p.leaves.size(); i++) {
        if (!w.has[S.leafElem[i]] || !w.leaves[i]) continue;
        Foliage f = p.leaves[i];
        f.n = w.leaves[i];  // the first n of the cluster: the ones shaken off are gone
        emit(b, f, p);
        out.mass += S.leafMass[i] * f.n;
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
        if (root) a0 = E.a + (E.b - E.a) * std::min(1.5f * E.ra / std::max(sqrtf(dot(E.b - E.a, E.b - E.a)), 1e-4f), 0.4f);
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

void bounds(const MeshData& m, float center[3], float& radius) {
    for (int k = 0; k < 3; k++) center[k] = 0.5f * (m.bmin[k] + m.bmax[k]);
    radius = 0;
    for (size_t i = 0; i < m.verts.size(); i += 8) {
        float dx = m.verts[i] - center[0], dy = m.verts[i + 1] - center[1], dz = m.verts[i + 2] - center[2];
        radius = std::max(radius, sqrtf(dx * dx + dy * dy + dz * dz));
    }
}

}  // namespace trees

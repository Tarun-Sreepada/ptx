#include "trees.h"

#include <cmath>
#include <algorithm>
#include <cstring>

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

// Conifer tier: jagged open cone (skirt) from a rim at height y up to an apex, closed underneath.
void tier(Builder& b, float y, float rad, float h, float seed, uint32_t mat) {
    const int sides = 14;
    Rng rng{(uint32_t)(seed * 9973.f) + 17u};
    uint32_t apex = b.vtx({0, y + h, 0}, 0.5f, h);
    uint32_t under = b.vtx({0, y + h * 0.25f, 0}, 0.5f, 0);
    uint32_t base = (uint32_t)(b.m.verts.size() / 8);
    for (int i = 0; i < sides; i++) {
        float ang = TAU * (i + rng.in(-0.2f, 0.2f)) / sides, r = rad * rng.in(0.82f, 1.12f);
        b.vtx({cosf(ang) * r, y + rng.in(-0.15f, 0.1f) * h, sinf(ang) * r}, cosf(ang) * r * 0.5f, sinf(ang) * r * 0.5f);
    }
    for (int i = 0; i < sides; i++) {
        uint32_t a = base + i, c = base + (i + 1) % sides;
        b.tri(apex, c, a, mat);
        b.tri(under, a, c, mat);
    }
    b.endPart();
}

// A tree as parts, so the same tree can be built whole or broken up: the trunk, branches (each carrying a crown
// lump at its end) and foliage (broadleaf lumps or conifer tiers).
struct Limb { V a, b; float ra, rb; int sides, leaves; };   // tapered tube; leaves = foliage at its tip, or -1
struct Foliage { bool tier; V c, r; float seed; };          // blob: centre, radii. tier: c.y rim height, r.x radius, r.y height
struct Plan { Limb trunk; std::vector<Limb> branches; std::vector<Foliage> leaves; int topLeaves = -1; uint32_t leafMat; };

Plan broadleaf(uint32_t seed, uint32_t leafMat) {
    Rng rng{seed * 2654435761u + 1u};
    Plan p;
    p.leafMat = leafMat;
    float th = rng.in(3.0f, 4.2f), tr = rng.in(0.22f, 0.32f);
    V lean = {rng.in(-0.25f, 0.25f), 0, rng.in(-0.25f, 0.25f)};
    V top = V{0, th, 0} + lean;
    p.trunk = {{0, -0.3f, 0}, top, tr * 1.25f, tr * 0.7f, 10, 0};
    int nb = 4 + (int)(rng() * 3);
    std::vector<std::pair<V, float>> crowns = {{top + V{0, rng.in(1.6f, 2.2f), 0}, rng.in(2.3f, 2.8f)}};
    for (int i = 0; i < nb; i++) {
        float ang = TAU * (i + rng.in(-0.25f, 0.25f)) / nb, up = rng.in(0.55f, 1.0f), len = rng.in(1.8f, 2.8f);
        V s = V{0, th * rng.in(0.62f, 0.95f), 0} + lean * 0.8f;
        V e = s + norm({cosf(ang), up, sinf(ang)}) * len;
        p.branches.push_back({s, e, tr * 0.42f, tr * 0.18f, 6, i + 1});
        crowns.push_back({e + V{0, 0.4f, 0}, rng.in(1.5f, 2.2f)});
    }
    for (size_t i = 0; i < crowns.size(); i++) {
        float r = crowns[i].second;
        p.leaves.push_back({false, crowns[i].first, {r * rng.in(1.0f, 1.2f), r * rng.in(0.75f, 0.95f), r * rng.in(1.0f, 1.2f)}, (float)(seed * 13 + i)});
    }
    p.topLeaves = 0;
    return p;
}

Plan conifer(uint32_t seed, uint32_t leafMat) {
    Rng rng{seed * 2246822519u + 7u};
    Plan p;
    p.leafMat = leafMat;
    float H = rng.in(9.f, 12.5f), base = rng.in(1.4f, 2.0f), rad = rng.in(2.6f, 3.4f);
    p.trunk = {{0, -0.3f, 0}, {0, H * 0.9f, 0}, 0.3f, 0.06f, 8, -1};
    int tiers = 6 + (int)(rng() * 3);
    for (int i = 0; i < tiers; i++) {
        float k = (float)i / (tiers - 1);
        float y = base + (H - base - 1.8f) * k;
        p.leaves.push_back({true, {0, y, 0}, {rad * (1.f - 0.82f * k), 1.8f + 1.4f * (1.f - k), 0}, (float)(seed * 31 + i)});
    }
    return p;
}

Plan plan(int v) { return v < 2 ? broadleaf(11u + v * 5u, 1 + v) : conifer(23u + v * 3u, 1 + v); }

void emit(Builder& b, const Limb& l) { tube(b, l.a, l.b, l.ra, l.rb, 0, l.sides); }
void emit(Builder& b, const Foliage& f, uint32_t mat) {
    if (f.tier) tier(b, f.c.y, f.r.x, f.r.y, f.seed, mat);
    else blob(b, f.c, f.r, f.seed, mat);
}

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

float tubeVolume(const Limb& l) {
    float len = sqrtf(dot(l.b - l.a, l.b - l.a));
    return 1.0471976f * len * (l.ra * l.ra + l.ra * l.rb + l.rb * l.rb);
}
float foliageVolume(const Foliage& f) { return f.tier ? 1.0471976f * f.r.x * f.r.x * f.r.y : 4.18879f * f.r.x * f.r.y * f.r.z; }
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
                  P_FERN, P_PEBBLE, P_TWIG, P_LITTER };

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

// Leafy crown of a few lumps; flowers (if any) dotted over its upper surface.
void shrub(Builder& b, uint32_t seed, float size, uint32_t leafMat, uint32_t flowerMat, int flowers, float flowerR) {
    Rng rng{seed * 3266489917u + 11u};
    int nc = 4 + (int)(rng() * 3);
    std::vector<std::pair<V, float>> crowns;
    for (int i = 0; i < nc; i++) {
        float a = TAU * i / nc + rng.in(-0.4f, 0.4f), d = i ? rng.in(0.3f, 0.55f) * size : 0.f;
        float r = (i ? rng.in(0.38f, 0.5f) : 0.55f) * size;
        crowns.push_back({{cosf(a) * d, r * 0.85f + (i ? 0.f : 0.15f * size), sinf(a) * d}, r});
    }
    for (int i = 1; i < nc; i++) tube(b, {0, 0, 0}, crowns[i].first, 0.03f * size, 0.012f * size, P_STEM, 4);
    for (size_t i = 0; i < crowns.size(); i++) {
        float r = crowns[i].second;
        blob(b, crowns[i].first, {r * rng.in(1.f, 1.15f), r * rng.in(0.8f, 0.95f), r * rng.in(1.f, 1.15f)}, (float)(seed * 17 + i), leafMat);
    }
    for (int i = 0; i < flowers; i++) {
        auto& c = crowns[(size_t)(rng() * crowns.size()) % crowns.size()];
        V d = norm({rng.in(-1, 1), rng.in(0.f, 1.f), rng.in(-1, 1)});
        V p = c.first + V{d.x, d.y * 0.85f, d.z} * (c.second * 0.98f);
        blob(b, p, {flowerR, flowerR * 0.8f, flowerR}, seed + i * 3.f, flowerMat, 1);
    }
}

// Fern: arching fronds from one crown, jagged (pinnate) ribbons.
void fern(Builder& b, uint32_t seed) {
    Rng rng{seed * 2654435761u + 3u};
    int n = 7 + (int)(rng() * 4);
    for (int i = 0; i < n; i++) {
        float a = TAU * (i + rng.in(-0.3f, 0.3f)) / n;
        ribbon(b, {0, 0.02f, 0}, norm({cosf(a), rng.in(0.9f, 1.6f), sinf(a)}), rng.in(0.45f, 0.7f), 0.11f, rng.in(0.45f, 0.7f), P_FERN, true);
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
TextureData barkTex() {
    TextureData t;
    t.w = 64; t.h = 128;
    t.rgba.resize(t.w * t.h * 4);
    for (uint32_t y = 0; y < t.h; y++)
        for (uint32_t x = 0; x < t.w; x++) {
            float fx = x / 64.f * 8.f, fy = y / 128.f * 4.f;
            float f = vnoise(fx * 2.f, fy * 0.6f) * 0.6f + vnoise(fx * 6.f, fy * 2.f) * 0.4f;
            float k = 0.55f + 0.6f * f;
            uint8_t* p = &t.rgba[(y * t.w + x) * 4];
            p[0] = srgb8(0.16f * k); p[1] = srgb8(0.11f * k); p[2] = srgb8(0.07f * k); p[3] = 255;
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

void build(ModelData& out) {
    out = {};
    out.textures = {barkTex(), leafTex()};
    MaterialData bark;
    bark.tex = 0;
    out.materials.push_back(bark);
    const float tints[VARIANTS][3] = {{0.10f, 0.20f, 0.045f}, {0.15f, 0.21f, 0.05f}, {0.05f, 0.11f, 0.05f}, {0.07f, 0.13f, 0.06f}};
    for (auto& c : tints) {
        MaterialData m;
        memcpy(m.color, c, 12);
        m.color[3] = 1;
        m.tex = 1;
        m.translucency = 0.45f;  // leaves glow when backlit
        m.rough = 0.55f;         // waxy cuticle
        out.materials.push_back(m);
    }
    MaterialData wood;  // fresh wood where a trunk or branch snapped (MAT_WOOD)
    wood.color[0] = 0.55f; wood.color[1] = 0.40f; wood.color[2] = 0.24f;
    out.materials.push_back(wood);
    out.meshes.resize(VARIANTS);
    for (int v = 0; v < VARIANTS; v++) {
        MeshData& m = out.meshes[v];
        Builder b{m};
        Plan p = plan(v);
        emit(b, p.trunk);
        for (auto& l : p.branches) emit(b, l);
        for (auto& f : p.leaves) emit(b, f, p.leafMat);
        finishMesh(m);
    }
}

void breakApart(int v, float cutY, float severity, uint32_t seed, std::vector<Piece>& out) {
    out.clear();
    Plan p = plan(v);
    Rng rng{seed * 2654435761u + 977u};
    severity = std::max(severity, 1.f);
    const Limb& T = p.trunk;
    V axis = T.b - T.a;
    float H = axis.y;
    auto at = [&](float t) { return T.a + axis * t; };
    auto rad = [&](float t) { return T.ra + (T.rb - T.ra) * t; };
    // cut points along the trunk: the snap, then more breaks further up the harder the blow
    int logs = std::clamp(1 + (int)log2f(severity), 1, 4);
    std::vector<float> cuts = {std::clamp((cutY - T.a.y) / H, 0.06f, 0.8f)};
    for (int i = 1; i < logs; i++) {
        float lo = cuts.back(), t = lo + (1.f - lo) * rng.in(0.3f, 0.6f);
        if ((t - lo) * H < 0.8f || (1.f - t) * H < 0.8f) break;
        cuts.push_back(t);
    }
    cuts.push_back(1.f);
    // piece 0 stump, 1..n logs (from the bottom); branches and leaves that come off get pieces of their own
    out.resize(cuts.size());
    out[0].kind = PIECE_STUMP;
    for (size_t i = 1; i < out.size(); i++) out[i].kind = PIECE_LOG;
    out.reserve(out.size() + p.branches.size() + p.leaves.size());
    auto add = [&](int kind) { out.push_back({}); out.back().kind = kind; return (int)out.size() - 1; };
    auto segOf = [&](float y) {  // which trunk piece holds height y
        float t = (y - T.a.y) / H;
        int k = 0;
        while (k < (int)cuts.size() - 1 && t > cuts[k]) k++;
        return k;
    };
    auto emitTo = [&](int k, auto&& fn) {
        Builder b{out[k].mesh};
        b.partStart = out[k].mesh.verts.size() / 8;
        fn(b);
    };
    for (size_t k = 0; k < cuts.size(); k++) {
        float t0 = k ? cuts[k - 1] : 0.f, t1 = cuts[k];
        Limb seg{at(t0), at(t1), rad(t0), rad(t1), T.sides, -1};
        emitTo((int)k, [&](Builder& b) {
            emit(b, seg);
            if (k) breakEnd(b, seg.a, axis, axis * -1.f, seg.ra, T.sides, rng, MAT_WOOD);
            if (k + 1 < cuts.size()) breakEnd(b, seg.b, axis, axis, seg.rb, T.sides, rng, MAT_WOOD);
        });
        out[k].mass += tubeVolume(seg) * WOOD_KG;
    }
    float loose = std::clamp(0.2f * (severity - 1.f), 0.f, 0.9f);  // chance a branch / crown tears off on its own
    std::vector<int> leafOwner(p.leaves.size(), -1);
    for (const Limb& l : p.branches) {
        int k = segOf(l.a.y);
        if (k > 0 && rng() < loose) {
            k = add(PIECE_BRANCH);
            emitTo(k, [&](Builder& b) { breakEnd(b, l.a, l.b - l.a, l.a - l.b, l.ra, l.sides, rng, MAT_WOOD); });
        }
        emitTo(k, [&](Builder& b) { emit(b, l); });
        out[k].mass += tubeVolume(l) * WOOD_KG;
        if (l.leaves >= 0) leafOwner[l.leaves] = k;
    }
    if (p.topLeaves >= 0) leafOwner[p.topLeaves] = (int)cuts.size() - 1;
    for (size_t i = 0; i < p.leaves.size(); i++) {
        const Foliage& f = p.leaves[i];
        int k = leafOwner[i] >= 0 ? leafOwner[i] : segOf(f.c.y);
        if (k > 0 && rng() < loose * 0.6f) k = add(PIECE_LEAVES);
        emitTo(k, [&](Builder& b) { emit(b, f, p.leafMat); });
        out[k].mass += foliageVolume(f) * LEAF_KG;
    }
    for (auto& pc : out) finishMesh(pc.mesh);
}

void buildPlants(ModelData& out) {
    out = {};
    out.textures = {leafTex()};
    const float cols[][3] = {{0.09f, 0.16f, 0.04f}, {0.13f, 0.06f, 0.025f}, {0.11f, 0.2f, 0.05f}, {0.8f, 0.55f, 0.02f}, {0.75f, 0.75f, 0.7f},
                             {0.45f, 0.012f, 0.02f}, {0.05f, 0.11f, 0.035f}, {0.08f, 0.17f, 0.04f}, {0.12f, 0.2f, 0.06f}, {0.75f, 0.6f, 0.7f}, {0.82f, 0.86f, 0.9f}, {0.11f, 0.08f, 0.05f}, {0.32f, 0.22f, 0.12f},
                             {0.06f, 0.14f, 0.035f}, {0.20f, 0.19f, 0.17f}, {0.12f, 0.085f, 0.055f}, {0.30f, 0.16f, 0.06f}};
    for (int i = 0; i <= (int)P_LITTER; i++) {
        MaterialData m;
        memcpy(m.color, cols[i], 12);
        bool leafy = i == P_LEAF || i == P_ROSELEAF || i == P_BUSH_A || i == P_BUSH_B;
        m.tex = leafy ? 0 : -1;
        if (leafy || i == P_STEM || i == P_BLOSSOM || i == P_FERN || i == P_LITTER) m.translucency = 0.4f;  // thin leaves, stems, petals pass light
        if (i == P_PEBBLE) m.rough = 0.7f;  // weathered
        if (i == P_FERN || i == P_LITTER) m.tex = 0;
        out.materials.push_back(m);
    }
    out.meshes.resize(PLANT_VARIANTS);
    for (int v = 0; v < PLANT_VARIANTS; v++) {
        MeshData& m = out.meshes[v];
        Builder b{m};
        if (v == PLANT_CATTAIL) cattails(b, 41u);
        else if (v == PLANT_DANDELION) dandelions(b, 53u);
        else if (v == PLANT_ROSE) shrub(b, 67u, 0.7f, P_ROSELEAF, P_ROSE, 16, 0.05f);
        else if (v == PLANT_BUSH) shrub(b, 79u, 1.1f, P_BUSH_A, 0, 0, 0);
        else if (v == PLANT_FLOWERBUSH) shrub(b, 97u, 0.9f, P_BUSH_B, P_BLOSSOM, 22, 0.045f);
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

void trunkCollider(const ModelData& md, int v, std::vector<float>& verts, std::vector<uint32_t>& idx) {
    // first part of every variant is the trunk tube: 2 rings, read its bottom/top centres and radii
    const MeshData& m = md.meshes[v];
    int sides = v < 2 ? 10 : 8;
    auto P = [&](int i) { return V{m.verts[i * 8], m.verts[i * 8 + 1], m.verts[i * 8 + 2]}; };
    V a{0, 0, 0}, c{0, 0, 0};
    for (int i = 0; i < sides; i++) { a = a + P(i); c = c + P(sides + 1 + i); }
    a = a * (1.f / sides); c = c * (1.f / sides);
    float ra = sqrtf(dot(P(0) - a, P(0) - a)), rc = sqrtf(dot(P(sides + 1) - c, P(sides + 1) - c));
    verts.clear(); idx.clear();
    const int S = 8;
    for (int ring = 0; ring < 2; ring++)
        for (int i = 0; i < S; i++) {
            float ang = TAU * i / S, r = ring ? rc : ra;
            V o = ring ? c : a;
            verts.insert(verts.end(), {o.x + cosf(ang) * r, o.y, o.z + sinf(ang) * r});
        }
    for (int i = 0; i < S; i++) {
        uint32_t q0 = i, q1 = (i + 1) % S, q2 = S + i, q3 = S + (i + 1) % S;
        idx.insert(idx.end(), {q0, q2, q1, q1, q2, q3});
        if (i > 0 && i < S - 1) idx.insert(idx.end(), {0, (uint32_t)i, (uint32_t)i + 1, S, (uint32_t)(S + i + 1), (uint32_t)(S + i)});
    }
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

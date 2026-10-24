#include "vegetation/trees/mesh_builder.h"

#include <cstring>
#include <utility>
#include <vector>

namespace trees {

uint32_t Builder::vtx(V p, float u, float v) {
    m.verts.insert(m.verts.end(), {p.x, p.y, p.z, 0, 0, 0, u, v});
    return (uint32_t)(m.verts.size() / 8 - 1);
}

void Builder::tri(uint32_t a, uint32_t b, uint32_t c, uint32_t mat) {
    m.idx.insert(m.idx.end(), {a, b, c});
    m.triMat.push_back(mat);
    V n = cross(pos(b) - pos(a), pos(c) - pos(a));
    for (uint32_t i : {a, b, c})
        for (int k = 0; k < 3; k++) m.verts[i * 8 + 3 + k] += (&n.x)[k];
}

void Builder::endPart() {
    for (size_t i = partStart; i < m.verts.size() / 8; i++) {
        V n = norm({m.verts[i * 8 + 3], m.verts[i * 8 + 4], m.verts[i * 8 + 5]});
        memcpy(&m.verts[i * 8 + 3], &n, 12);
    }
    partStart = m.verts.size() / 8;
}

void tube(Builder& b, V a, V c, float ra, float rc, uint32_t mat, int sides) {
    V ax = norm(c - a);
    V t = norm(cross(fabsf(ax.y) < 0.99f ? V{0, 1, 0} : V{1, 0, 0}, ax)), bt = cross(ax, t);
    float len = length(c - a);
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

void blob(Builder& b, V c, V r, float seed, uint32_t mat, int subdiv) {
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

void needle(Builder& b, V base, V d, V side, float L, float W, uint32_t mat) {
    uint32_t a = b.vtx(base - side * (0.5f * W), 0.f, 0.f), c = b.vtx(base + side * (0.5f * W), 1.f, 0.f);
    uint32_t tip = b.vtx(base + d * L, 0.5f, 1.f);
    b.tri(a, c, tip, mat);
    b.endPart();
}

void ribbon(Builder& b, V base, V dir, float len, float w, float droop, uint32_t mat, bool jag) {
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

void finishMesh(MeshData& m) {
    for (int k = 0; k < 3; k++) { m.bmin[k] = 1e30f; m.bmax[k] = -1e30f; }
    for (size_t i = 0; i < m.verts.size(); i += 8)
        for (int k = 0; k < 3; k++) { m.bmin[k] = std::min(m.bmin[k], m.verts[i + k]); m.bmax[k] = std::max(m.bmax[k], m.verts[i + k]); }
}

}  // namespace trees

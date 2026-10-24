#include "physics/fracture.h"

#include <algorithm>
#include <cmath>

namespace fracture {
namespace {

Vec operator+(Vec a, Vec b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
Vec operator-(Vec a, Vec b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
Vec operator*(Vec a, float s) { return {a.x * s, a.y * s, a.z * s}; }
float dot(Vec a, Vec b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
Vec cross(Vec a, Vec b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
Vec norm(Vec a) { float l = std::sqrt(dot(a, a)); return l > 1e-12f ? a * (1.f / l) : Vec{0, 1, 0}; }

struct Rng {
    uint32_t s;
    float operator()() { s = s * 747796405u + 2891336453u; uint32_t w = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u; return ((w >> 22u) ^ w) / 4294967296.f; }
};

Vec faceNormal(const std::vector<Vec>& p) {  // Newell
    Vec n{0, 0, 0};
    for (size_t i = 0; i < p.size(); i++) {
        const Vec &a = p[i], &b = p[(i + 1) % p.size()];
        n.x += (a.y - b.y) * (a.z + b.z);
        n.y += (a.z - b.z) * (a.x + b.x);
        n.z += (a.x - b.x) * (a.y + b.y);
    }
    return norm(n);
}

Poly box(Vec mn, Vec mx) {
    Vec c[8];
    for (int i = 0; i < 8; i++) c[i] = {i & 1 ? mx.x : mn.x, i & 2 ? mx.y : mn.y, i & 4 ? mx.z : mn.z};
    // CCW seen from outside
    const int f[6][4] = {{0, 4, 6, 2}, {1, 3, 7, 5}, {0, 1, 5, 4}, {2, 6, 7, 3}, {0, 2, 3, 1}, {4, 5, 7, 6}};
    Poly p;
    for (auto& q : f) p.faces.push_back({{c[q[0]], c[q[1]], c[q[2]], c[q[3]]}, false});
    return p;
}

}  // namespace

Poly clip(const Poly& p, Vec n, float d) {
    Poly out;
    std::vector<Vec> cap;
    float scale = 0.f;
    for (auto& f : p.faces) for (auto& v : f.pts) scale = std::max(scale, std::fabs(dot(v, n) - d));
    const float eps = 1e-6f * std::max(scale, 1e-3f);
    for (auto& f : p.faces) {
        Face nf;
        nf.inner = f.inner;
        size_t k = f.pts.size();
        for (size_t i = 0; i < k; i++) {
            Vec a = f.pts[i], b = f.pts[(i + 1) % k];
            float da = dot(a, n) - d, db = dot(b, n) - d;
            bool ia = da <= eps, ib = db <= eps;
            if (ia) nf.pts.push_back(a);
            if (ia != ib) {
                Vec x = a + (b - a) * (da / (da - db));
                nf.pts.push_back(x);
                cap.push_back(x);
            } else if (ia && std::fabs(da) <= eps) cap.push_back(a);
        }
        if (nf.pts.size() >= 3) out.faces.push_back(std::move(nf));
    }
    if (cap.size() >= 3) {
        // dedupe, then order around the centroid, CCW seen from outside (looking down -n)
        std::vector<Vec> u;
        float tol2 = 1e-10f * std::max(scale * scale, 1e-6f) + 1e-12f;
        for (auto& v : cap) {
            bool dup = false;
            for (auto& w : u) if (dot(v - w, v - w) < tol2) { dup = true; break; }
            if (!dup) u.push_back(v);
        }
        if (u.size() >= 3) {
            Vec c{0, 0, 0};
            for (auto& v : u) c = c + v;
            c = c * (1.f / u.size());
            Vec t = norm(cross(std::fabs(n.y) < 0.9f ? Vec{0, 1, 0} : Vec{1, 0, 0}, n)), b = cross(n, t);
            std::sort(u.begin(), u.end(), [&](Vec a, Vec bb) {
                return std::atan2(dot(a - c, b), dot(a - c, t)) < std::atan2(dot(bb - c, b), dot(bb - c, t));
            });
            out.faces.push_back({u, true});
        }
    }
    return out;
}

Poly fromPoints(const float* pts, uint32_t stride, uint32_t n, int dirs) {
    Vec mn{1e30f, 1e30f, 1e30f}, mx{-1e30f, -1e30f, -1e30f};
    for (uint32_t i = 0; i < n; i++) {
        Vec v{pts[i * stride], pts[i * stride + 1], pts[i * stride + 2]};
        mn = {std::min(mn.x, v.x), std::min(mn.y, v.y), std::min(mn.z, v.z)};
        mx = {std::max(mx.x, v.x), std::max(mx.y, v.y), std::max(mx.z, v.z)};
    }
    Poly p = box(mn, mx);
    for (int k = 0; k < dirs; k++) {  // Fibonacci sphere directions
        float y = 1.f - 2.f * (k + 0.5f) / dirs, r = std::sqrt(std::max(0.f, 1.f - y * y)), a = k * 2.39996323f;
        Vec d{std::cos(a) * r, y, std::sin(a) * r};
        float s = -1e30f;
        for (uint32_t i = 0; i < n; i++) s = std::max(s, dot(d, {pts[i * stride], pts[i * stride + 1], pts[i * stride + 2]}));
        p = clip(p, d, s);
    }
    for (auto& f : p.faces) f.inner = false;
    return p;
}

float volume(const Poly& p, Vec* centroid) {
    Vec o{0, 0, 0};
    size_t cnt = 0;
    for (auto& f : p.faces) for (auto& v : f.pts) { o = o + v; cnt++; }
    if (!cnt) { if (centroid) *centroid = o; return 0.f; }
    o = o * (1.f / cnt);
    float V = 0.f;
    Vec c{0, 0, 0};
    for (auto& f : p.faces)
        for (size_t i = 1; i + 1 < f.pts.size(); i++) {
            Vec a = f.pts[0] - o, b = f.pts[i] - o, d = f.pts[i + 1] - o;
            float v = dot(a, cross(b, d)) / 6.f;
            V += v;
            c = c + (o + (a + b + d) * 0.25f) * v;
        }
    if (centroid) *centroid = V > 1e-12f ? c * (1.f / V) : o;
    return V;
}

float radius(const Poly& p, Vec c) {
    float r = 0.f;
    for (auto& f : p.faces) for (auto& v : f.pts) r = std::max(r, dot(v - c, v - c));
    return std::sqrt(r);
}

void translate(Poly& p, Vec by) {
    for (auto& f : p.faces) for (auto& v : f.pts) v = v + by;
}

static bool inside(const Poly& p, Vec x) {
    for (auto& f : p.faces) {
        Vec n = faceNormal(f.pts);
        if (dot(x - f.pts[0], n) > 0.f) return false;
    }
    return true;
}

void shatter(const Poly& p, Vec impact, Vec dir, int count, float spread, uint32_t seed, std::vector<Poly>& out) {
    Vec c;
    volume(p, &c);
    float R = radius(p, c);
    Rng rng{seed * 2654435761u + 12345u};
    std::vector<Vec> seeds;
    Vec t = norm(cross(std::fabs(dir.y) < 0.9f ? Vec{0, 1, 0} : Vec{1, 0, 0}, dir)), b = cross(dir, t);
    float cone = 0.5f + 0.7f * spread;  // half-angle (rad) of the cone cracks open in
    for (int i = 0, tries = 0; (int)seeds.size() < count && tries < count * 40; tries++) {
        Vec s;
        if (i < count / 6 || rng() < spread * 0.5f) {  // a few seeds anywhere: the big chunks away from the blow
            s = c + Vec{rng() * 2 - 1, rng() * 2 - 1, rng() * 2 - 1} * R;
        } else {
            float u = rng(), dist = 2.f * R * (0.03f + 0.97f * u * u * (0.4f + 0.6f * spread));  // dense near contact
            float th = cone * std::sqrt(rng()), ph = 6.2831853f * rng();
            Vec d = norm(dir * std::cos(th) + (t * std::cos(ph) + b * std::sin(ph)) * std::sin(th));
            s = impact + d * dist;
        }
        if (!inside(p, s)) continue;
        seeds.push_back(s);
        i++;
    }
    if (seeds.size() < 2) { out.push_back(p); return; }
    for (size_t i = 0; i < seeds.size(); i++) {
        Poly cell = p;
        for (size_t j = 0; j < seeds.size() && !cell.faces.empty(); j++) {
            if (i == j) continue;
            Vec n = seeds[j] - seeds[i];
            float l2 = dot(n, n);
            if (l2 < 1e-12f) continue;
            n = n * (1.f / std::sqrt(l2));
            cell = clip(cell, n, dot(n, (seeds[i] + seeds[j]) * 0.5f));
        }
        if (cell.faces.size() >= 4 && volume(cell) > 1e-7f) out.push_back(std::move(cell));
    }
}

void toMesh(const Poly& p, MeshData& m, uint32_t matOuter, uint32_t matInner, float uvScale) {
    m.verts.clear(); m.idx.clear(); m.triMat.clear();
    for (auto& f : p.faces) {
        Vec n = faceNormal(f.pts);
        Vec t = norm(cross(std::fabs(n.y) < 0.9f ? Vec{0, 1, 0} : Vec{1, 0, 0}, n)), b = cross(n, t);
        uint32_t base = (uint32_t)(m.verts.size() / 8);
        for (auto& v : f.pts) m.verts.insert(m.verts.end(), {v.x, v.y, v.z, n.x, n.y, n.z, dot(v, t) * uvScale, dot(v, b) * uvScale});
        for (size_t i = 1; i + 1 < f.pts.size(); i++) {
            m.idx.insert(m.idx.end(), {base, base + (uint32_t)i, base + (uint32_t)i + 1});
            m.triMat.push_back(f.inner ? matInner : matOuter);
        }
    }
    for (int k = 0; k < 3; k++) { m.bmin[k] = 1e30f; m.bmax[k] = -1e30f; }
    for (size_t i = 0; i < m.verts.size(); i += 8)
        for (int k = 0; k < 3; k++) { m.bmin[k] = std::min(m.bmin[k], m.verts[i + k]); m.bmax[k] = std::max(m.bmax[k], m.verts[i + k]); }
}

}  // namespace fracture

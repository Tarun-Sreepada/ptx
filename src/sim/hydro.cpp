#include "sim/hydro.h"

#include <algorithm>
#include <cmath>
#include <mutex>

#include "core/constants.h"
#include "physics/physics.h"
#include "scene/rocks.h"
#include "scene/scene_gpu.h"

namespace {
constexpr int DIR_BINS = 26;           // facing directions: cube faces, edges, corners
constexpr int MAX_SEGMENTS = 6;        // elements along a long body
constexpr int TUBE_SIDES = 6;          // faces around a limb
constexpr float C_PRESSURE = 1.0f;     // stagnation pressure coefficient on faces moving into the water
constexpr float C_SUCTION = 0.15f;     // low pressure on faces moving away
constexpr float C_FRICTION = 0.01f;    // skin friction
constexpr float BED_CURRENT = 0.5f;    // current near the bed, share of the surface's
constexpr float MAX_STOP = 0.8f;       // a step's drag removes at most this share of an element's relative motion
constexpr float MIN_DEPTH = 0.3f;      // m: shallower channels count as this deep (current profile)

struct V { float x, y, z; };
V operator+(V a, V b) { return {a.x + b.x, a.y + b.y, a.z + b.z}; }
V operator-(V a, V b) { return {a.x - b.x, a.y - b.y, a.z - b.z}; }
V operator*(V a, float s) { return {a.x * s, a.y * s, a.z * s}; }
float dot(V a, V b) { return a.x * b.x + a.y * b.y + a.z * b.z; }
V cross(V a, V b) { return {a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x}; }
float len(V a) { return sqrtf(dot(a, a)); }
V norm(V a) { float l = len(a); return l > 0.f ? a * (1.f / l) : V{0, 1, 0}; }
V perpTo(V d) { return norm(cross(d, fabsf(d.y) < 0.95f ? V{0, 1, 0} : V{1, 0, 0})); }
V linear(const Xf& x, V v) {  // rotation x scale part
    return {x.m[0][0] * v.x + x.m[0][1] * v.y + x.m[0][2] * v.z, x.m[1][0] * v.x + x.m[1][1] * v.y + x.m[1][2] * v.z,
            x.m[2][0] * v.x + x.m[2][1] * v.y + x.m[2][2] * v.z};
}

const V* dirs() {
    static V d[DIR_BINS];
    static std::once_flag once;
    std::call_once(once, [] {
        int k = 0;
        for (int x = -1; x <= 1; x++)
            for (int y = -1; y <= 1; y++)
                for (int z = -1; z <= 1; z++)
                    if (x || y || z) d[k++] = norm(V{(float)x, (float)y, (float)z});
    });
    return d;
}
int dirBin(V n) {
    const V* D = dirs();
    int best = 0;
    float bd = -2.f;
    for (int k = 0; k < DIR_BINS; k++) { float s = dot(n, D[k]); if (s > bd) { bd = s; best = k; } }
    return best;
}

// running weighted sums of a bin
struct VolAcc { V p{0, 0, 0}; float v = 0.f; };
struct FaceAcc { V p{0, 0, 0}, n{0, 0, 0}; float a = 0.f; };
void addFace(std::vector<FaceAcc>& bins, int i, V p, V n, float a) { bins[i].p = bins[i].p + p * a; bins[i].n = bins[i].n + n * a; bins[i].a += a; }

// bins -> elements, the volumes rescaled to sum to volume (if given)
void finish(HydroShape& out, const std::vector<VolAcc>& vols, const std::vector<FaceAcc>& faces, float volume) {
    float total = 0.f;
    for (const VolAcc& b : vols) total += std::max(b.v, 0.f);
    float k = volume > 0.f && total > 0.f ? volume / total : 1.f;
    for (const VolAcc& b : vols) {
        if (b.v <= 0.f) continue;
        V p = b.p * (1.f / b.v);
        out.vols.push_back({{p.x, p.y, p.z}, b.v * k, cbrtf(b.v * k)});
    }
    for (const FaceAcc& b : faces) {
        if (b.a <= 0.f) continue;
        V p = b.p * (1.f / b.a), n = norm(b.n);
        out.faces.push_back({{p.x, p.y, p.z}, {n.x, n.y, n.z}, b.a});
    }
}
}  // namespace

std::shared_ptr<const HydroShape> hydroFromMesh(const float* verts, uint32_t stride, const uint32_t* idx, uint32_t ntris, float volume,
                                                const std::vector<uint8_t>* solid) {
    auto out = std::make_shared<HydroShape>();
    struct Tri { V a, b, c, m, cr; float area; bool solid; };
    std::vector<Tri> tris;
    tris.reserve(ntris);
    auto P = [&](uint32_t i) { return V{verts[i * stride], verts[i * stride + 1], verts[i * stride + 2]}; };
    V c{0, 0, 0};
    float total = 0.f;
    for (uint32_t t = 0; t < ntris; t++) {
        V a = P(idx[t * 3]), b = P(idx[t * 3 + 1]), d = P(idx[t * 3 + 2]);
        V cr = cross(b - a, d - a);
        float area = 0.5f * len(cr);
        if (area <= 0.f) continue;
        V m = (a + b + d) * (1.f / 3.f);
        tris.push_back({a, b, d, m, cr, area, !solid || t >= solid->size() || (*solid)[t]});
        c = c + m * area;
        total += area;
    }
    if (total <= 0.f) return out;
    c = c * (1.f / total);
    // the long axis (power iteration on the area-weighted spread of the surface) and the body's length / girth
    float C[3][3] = {};
    for (const Tri& t : tris) {
        V d = t.m - c;
        float dv[3] = {d.x, d.y, d.z};
        for (int i = 0; i < 3; i++) for (int j = 0; j < 3; j++) C[i][j] += t.area * dv[i] * dv[j];
    }
    V ax = norm(V{1.f, 0.7f, 0.3f});
    for (int it = 0; it < 24; it++)
        ax = norm(V{C[0][0] * ax.x + C[0][1] * ax.y + C[0][2] * ax.z, C[1][0] * ax.x + C[1][1] * ax.y + C[1][2] * ax.z,
                    C[2][0] * ax.x + C[2][1] * ax.y + C[2][2] * ax.z});
    V e1 = perpTo(ax), e2 = cross(ax, e1);
    float s0 = 1e30f, s1 = -1e30f, girth = 0.f;
    for (const Tri& t : tris) {
        for (V q : {t.a, t.b, t.c}) { float s = dot(q - c, ax); s0 = std::min(s0, s); s1 = std::max(s1, s); }
        V d = t.m - c;
        V r = d - ax * dot(d, ax);
        girth += t.area * dot(r, r);
    }
    girth = sqrtf(girth / total);
    float length = std::max(s1 - s0, 1e-6f);
    int nseg = std::clamp((int)lroundf(length / std::max(2.f * girth, 1e-6f)), 1, MAX_SEGMENTS);
    int nsegV = std::max(nseg, 2);  // volume split in halves at least: a partly sunk rock gets its righting torque
    auto seg = [&](float s, int n) { return std::clamp((int)((s - s0) / length * n), 0, n - 1); };
    std::vector<VolAcc> vols((size_t)nsegV * 4);
    std::vector<FaceAcc> faces((size_t)nseg * DIR_BINS);
    for (const Tri& t : tris) {
        float s = std::clamp(dot(t.m - c, ax), s0, s1);
        V ref = c + ax * s;  // the axis point under this triangle: its wedge of the solid reaches there
        V n = norm(t.cr);
        if (dot(n, t.m - ref) < 0.f) n = n * -1.f;  // outward, whatever the winding
        int fs = seg(s, nseg);
        addFace(faces, fs * DIR_BINS + dirBin(n), t.m, n, t.area);
        if (!t.solid) {  // a thin sheet: water pushes on both sides
            addFace(faces, fs * DIR_BINS + dirBin(n * -1.f), t.m, n * -1.f, t.area);
            continue;
        }
        float vt = fabsf(dot(t.a - ref, cross(t.b - ref, t.c - ref))) / 6.f;
        V g = (ref + t.a + t.b + t.c) * 0.25f, gd = g - c;
        int q = (dot(gd, e1) >= 0.f ? 1 : 0) + (dot(gd, e2) >= 0.f ? 2 : 0);
        VolAcc& b = vols[(size_t)seg(s, nsegV) * 4 + q];
        b.p = b.p + g * vt;
        b.v += vt;
    }
    finish(*out, vols, faces, volume);
    return out;
}

std::shared_ptr<const HydroShape> hydroFromTubes(const std::vector<float>& tubes, const std::vector<float>& plates) {
    auto out = std::make_shared<HydroShape>();
    std::vector<VolAcc> vols;
    std::vector<FaceAcc> faces;
    for (size_t i = 0; i + 8 <= tubes.size(); i += 8) {
        V a{tubes[i], tubes[i + 1], tubes[i + 2]}, b{tubes[i + 3], tubes[i + 4], tubes[i + 5]};
        float ra = tubes[i + 6], rb = tubes[i + 7], L = len(b - a);
        if (L <= 0.f) continue;
        V ax = (b - a) * (1.f / L), t = perpTo(ax), bt = cross(ax, t);
        int n = std::clamp((int)(L / std::max(ra + rb, 1e-3f)), 1, 4);
        for (int k = 0; k < n; k++) {
            float u0 = (float)k / n, u1 = (k + 1.f) / n, r0 = ra + (rb - ra) * u0, r1 = ra + (rb - ra) * u1;
            float r = 0.5f * (r0 + r1), l = L / n;
            V m = a + (b - a) * (0.5f * (u0 + u1));
            float vol = phys::PI / 3.f * l * (r0 * r0 + r0 * r1 + r1 * r1);
            vols.push_back({m * vol, vol});
            float side = 2.f * phys::PI * r * l / TUBE_SIDES;
            for (int s = 0; s < TUBE_SIDES; s++) {
                float ang = 2.f * phys::PI * s / TUBE_SIDES;
                V d = t * cosf(ang) + bt * sinf(ang);
                faces.push_back({(m + d * r) * side, d * side, side});
            }
        }
    }
    // a leaf cluster: plates at every angle, ~half their area facing any one way (both sides of each catch water)
    const V axes[6] = {{1, 0, 0}, {-1, 0, 0}, {0, 1, 0}, {0, -1, 0}, {0, 0, 1}, {0, 0, -1}};
    for (size_t i = 0; i + 4 <= plates.size(); i += 4) {
        V c{plates[i], plates[i + 1], plates[i + 2]};
        float a = plates[i + 3] * 0.5f;
        if (a <= 0.f) continue;
        for (V d : axes) faces.push_back({c * a, d * a, a});
    }
    finish(*out, vols, faces, 0.f);
    return out;
}

std::shared_ptr<const HydroShape> hydroRockVariant(uint32_t var) {
    static std::shared_ptr<const HydroShape> cache[ROCK_VARIANTS];
    static std::mutex m;
    std::lock_guard<std::mutex> lock(m);
    if (var >= (uint32_t)ROCK_VARIANTS) var = 0;
    if (!cache[var] && !rockPts.empty() && !rockTriIdx.empty()) {
        // the unit rock's volume: divergence theorem over its (closed) surface
        const float* v = rockPts.data() + (size_t)var * rockVerts * 6;
        double vol = 0.0;
        for (size_t t = 0; t + 2 < rockTriIdx.size(); t += 3) {
            const float *a = v + rockTriIdx[t] * 6, *b = v + rockTriIdx[t + 1] * 6, *c = v + rockTriIdx[t + 2] * 6;
            vol += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6.0;
        }
        cache[var] = hydroFromMesh(v, 6, rockTriIdx.data(), (uint32_t)(rockTriIdx.size() / 3), (float)std::fabs(vol));
    }
    return cache[var];
}

float applyWater(const HydroShape& shape, const HydroBody& b, const HydroWater& wt, float dt) {
    if (shape.vols.empty() && shape.faces.empty()) return 0.f;
    float s = b.scale, s2 = s * s, s3 = s2 * s;
    V O{b.xf.m[0][3], b.xf.m[1][3], b.xf.m[2][3]}, C{b.com[0], b.com[1], b.com[2]};
    V vel{b.v[0], b.v[1], b.v[2]}, om{b.w[0], b.w[1], b.w[2]};
    V F{0, 0, 0}, T{0, 0, 0};
    auto push = [&](V p, V f) { F = F + f; T = T + cross(p - C, f); };
    // buoyancy, each volume element where it is
    float volAll = 0.f, volIn = 0.f;
    for (const HydroShape::Vol& e : shape.vols) {
        V p = O + linear(b.xf, V{e.p[0], e.p[1], e.p[2]});
        float v = e.v * s3, f = std::clamp(0.5f + (wt.level - p.y) / std::max(e.size * s, 1e-4f), 0.f, 1.f);
        volAll += v;
        if (f <= 0.f) continue;
        volIn += v * f;
        push(p, V{0.f, phys::WATER_DENSITY * phys::GRAVITY * v * f, 0.f});
    }
    // the push on every wet face, against the water's own motion
    float areaAll = 0.f;
    for (const HydroShape::Face& e : shape.faces) areaAll += e.a;
    float massPerArea = b.mass / std::max(areaAll * s2, 1e-6f);
    float depth = std::max(wt.depth, MIN_DEPTH);
    for (const HydroShape::Face& e : shape.faces) {
        V p = O + linear(b.xf, V{e.p[0], e.p[1], e.p[2]});
        float a = e.a * s2, f = std::clamp(0.5f + (wt.level - p.y) / std::max(sqrtf(a), 1e-4f), 0.f, 1.f);
        if (f <= 0.f) continue;
        V n = norm(linear(b.xf, V{e.n[0], e.n[1], e.n[2]}));
        float k = 1.f - (1.f - BED_CURRENT) * std::clamp((wt.level - p.y) / depth, 0.f, 1.f);
        V u = vel + cross(om, p - C) - V{wt.cur[0] * k, 0.f, wt.cur[1] * k};
        float un = dot(u, n);
        V ut = u - n * un;
        float q = 0.5f * phys::WATER_DENSITY * a * f;
        V drag = n * (un > 0.f ? -q * C_PRESSURE * un * un : q * C_SUCTION * un * un) + ut * (-q * C_FRICTION * len(ut));
        // never more than stops this face's share of the body within the step
        float dl = len(drag), cap = MAX_STOP * massPerArea * a * len(u) / std::max(dt, 1e-4f);
        if (dl > cap) drag = drag * (cap / dl);
        push(p, drag);
    }
    if (volIn <= 0.f && dot(F, F) <= 0.f) return 0.f;
    float Fa[3] = {F.x, F.y, F.z}, Ta[3] = {T.x, T.y, T.z};
    physics::addForce(b.id, Fa, Ta);
    return volAll > 0.f ? std::min(volIn / volAll, 1.f) : 1.f;
}

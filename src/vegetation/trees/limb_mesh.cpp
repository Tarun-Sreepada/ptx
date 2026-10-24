#include "vegetation/trees/limb_mesh.h"

namespace trees {
namespace {
constexpr int TRUNK_SIDES = 16;
constexpr int WAVY_MIN_SIDES = 7;           // a lumpy cross-section needs this many
constexpr float WAVY_MIN_R = 0.03f;         // m: thinner limbs stay round
constexpr float WAVE_FULL_R = 0.08f;        // m: limbs this thick get the full lumpiness
constexpr float TWIG_R = 0.02f;             // m: thinner chains get rings at their joints only
constexpr float TRUNK_RING_SPACING = 0.18f; // m between rings up the trunk path (flare and lumps need them)
constexpr float RING_SPACING_PER_R = 4.f, RING_SPACING_MIN = 0.12f, RING_SPACING_MAX = 0.4f;
constexpr float COLLAR_SWELL = 0.4f;        // a branch is this much thicker where it leaves its parent ...
constexpr float COLLAR_FADE = 1.5f;         // ... fading over this many of its own radii
constexpr float BARK_V_SCALE = 0.5f;        // bark texture v per metre along a limb
constexpr float TANGENT_STEP = 0.02f;       // in s, for the curve's direction

// centripetal Catmull-Rom from p1 (u = 0) to p2 (u = 1): no loops or overshoot with uneven joint spacing
V catmullRom(V p0, V p1, V p2, V p3, float u) {
    auto knot = [](V a, V b, float t) { return t + sqrtf(std::max(length(b - a), 1e-5f)); };
    float t0 = 0.f, t1 = knot(p0, p1, t0), t2 = knot(p1, p2, t1), t3 = knot(p2, p3, t2);
    float t = t1 + (t2 - t1) * u;
    V A1 = p0 * ((t1 - t) / (t1 - t0)) + p1 * ((t - t0) / (t1 - t0));
    V A2 = p1 * ((t2 - t) / (t2 - t1)) + p2 * ((t - t1) / (t2 - t1));
    V A3 = p2 * ((t3 - t) / (t3 - t2)) + p3 * ((t - t2) / (t3 - t2));
    V B1 = A1 * ((t2 - t) / (t2 - t0)) + A2 * ((t - t0) / (t2 - t0));
    V B2 = A2 * ((t3 - t) / (t3 - t1)) + A3 * ((t - t1) / (t3 - t1));
    return B1 * ((t2 - t) / (t2 - t1)) + B2 * ((t - t1) / (t2 - t1));
}

void locate(const LimbPath& L, float s, int& i, float& u) {
    int n = (int)L.pts.size() - 1;
    i = std::clamp((int)floorf(s), 0, n - 1);
    u = std::clamp(s - (float)i, 0.f, 1.f);
}

V centreAt(const LimbPath& L, float s) {
    int i; float u;
    locate(L, s, i, u);
    int n = (int)L.pts.size() - 1;
    V p1 = L.pts[i], p2 = L.pts[i + 1];
    V p0 = i > 0 ? L.pts[i - 1] : p1 * 2.f - p2, p3 = i + 2 <= n ? L.pts[i + 2] : p2 * 2.f - p1;
    V c = catmullRom(p0, p1, p2, p3, u);
    if (L.wander > 0.f && i == 0) {  // the trunk's centre line wanders, pinned at both ends so the leader joins on
        float w = sinf(PI * u) * L.wander, y = c.y - L.pts[0].y;
        c = c + V{(noise3({y * 0.35f, L.seed, 1.7f}) - 0.5f) * 2.f * w, 0.f, (noise3({y * 0.35f, L.seed + 9.f, 4.1f}) - 0.5f) * 2.f * w};
    }
    return c;
}

V tangentAt(const LimbPath& L, float s) {
    float n = (float)(L.pts.size() - 1);
    return norm(centreAt(L, std::min(s + TANGENT_STEP, n)) - centreAt(L, std::max(s - TANGENT_STEP, 0.f)));
}

float arcAt(const LimbPath& L, float s) {
    int i; float u;
    locate(L, s, i, u);
    return L.arc[i] + (L.arc[i + 1] - L.arc[i]) * u;
}

float radiusAt(const LimbPath& L, float s) {
    int i; float u;
    locate(L, s, i, u);
    float r = L.r[i] + (L.r[i + 1] - L.r[i]) * u;
    if (L.collarSwell > 0.f) r *= 1.f + L.collarSwell * expf(-std::max(arcAt(L, s) - L.collarExit, 0.f) / (COLLAR_FADE * L.r[0]));
    return r;
}

// keep n perpendicular to t, turning it as little as possible (parallel transport between nearby rings)
V transport(V n, V t) {
    V m = n - t * dot(n, t);
    return dot(m, m) > 1e-8f ? norm(m) : perpTo(t);
}
}  // namespace

LimbPath limbPath(const Plan& p, int c) {
    const Chain& ch = p.chains[c];
    LimbPath L;
    L.trunk = c == 0 && p.trunkPath;
    if (L.trunk) { L.pts = {p.trunk.a, p.trunk.b}; L.r = {p.trunk.ra, p.trunk.rb}; }
    else { const Limb& l0 = p.branches[ch.segs[0]]; L.pts = {l0.a}; L.r = {l0.ra}; }
    for (int i : ch.segs) { L.pts.push_back(p.branches[i].b); L.r.push_back(p.branches[i].rb); }
    float r0 = L.r[L.trunk ? 1 : 0];  // the limb's own base radius
    L.wave = p.wave;
    L.sides = L.trunk ? TRUNK_SIDES : (p.wave > 0.f && r0 > WAVY_MIN_R ? std::max(ch.sides, WAVY_MIN_SIDES) : ch.sides);
    if (L.trunk) { L.flare = p.flare; L.flareR = p.trunk.ra; L.wander = p.wave * p.trunk.ra * 1.2f; }
    if (ch.parentR > 0.f && L.pts.size() > 1) {
        // it leaves the parent's surface after parentR / sin(angle) along itself; the collar never outgrows the parent
        float c0 = dot(norm(L.pts[1] - L.pts[0]), ch.parentDir);
        L.collarR = ch.parentR;
        L.collarExit = ch.parentR / sqrtf(std::max(1.f - c0 * c0, 0.04f));
        L.collarSwell = std::clamp(0.95f * ch.parentR / std::max(r0, 1e-4f) - 1.f, 0.f, COLLAR_SWELL);
    }
    L.seed = p.seed;
    L.arc = {0.f};
    float spacing = L.trunk ? TRUNK_RING_SPACING : std::clamp(RING_SPACING_PER_R * r0, RING_SPACING_MIN, RING_SPACING_MAX);
    bool twig = !L.trunk && r0 < TWIG_R;
    for (size_t i = 0; i + 1 < L.pts.size(); i++) {
        float len = length(L.pts[i + 1] - L.pts[i]);
        L.arc.push_back(L.arc.back() + len);
        int n = twig ? 1 : std::max(1, (int)ceilf(len / spacing));
        for (int k = 0; k < n; k++) L.rings.push_back((float)i + (float)k / n);
    }
    L.rings.push_back((float)(L.pts.size() - 1));
    return L;
}

void sweepLimb(Builder& b, const LimbPath& L, float s0, float s1, uint32_t mat) {
    if (s1 <= s0 || L.pts.size() < 2) return;
    // ring frames: carried from the base through every canonical ring, so a piece's rings match the whole limb's
    std::vector<V> frames(L.rings.size());
    V n = perpTo(tangentAt(L, 0.f));
    for (size_t k = 0; k < L.rings.size(); k++) frames[k] = n = transport(n, tangentAt(L, L.rings[k]));
    size_t k = 0;  // the last canonical ring at or before s (rings are visited in order)
    auto normalAt = [&](float s) {
        while (k + 1 < L.rings.size() && L.rings[k + 1] <= s) k++;
        return transport(frames[k], tangentAt(L, s));
    };
    std::vector<float> ss{s0};
    for (float s : L.rings) if (s > s0 + 1e-4f && s < s1 - 1e-4f) ss.push_back(s);
    ss.push_back(s1);
    int S = L.sides;
    uint32_t base = (uint32_t)(b.m.verts.size() / 8);
    for (float s : ss) {
        V c = centreAt(L, s), t = tangentAt(L, s), nn = normalAt(s), bt = cross(t, nn);
        float rr = radiusAt(L, s), v = arcAt(L, s) * BARK_V_SCALE;
        float wave = L.wave * std::min(1.f, rr / WAVE_FULL_R);
        float flare = L.flare > 0.f ? L.flare * L.flareR * expf(-std::max(c.y, 0.f) / (0.35f + 1.2f * L.flareR)) : 0.f;
        for (int i = 0; i <= S; i++) {
            float ang = TAU * (i % S) / S;
            V d = nn * cosf(ang) + bt * sinf(ang);
            float rad = rr;
            if (wave > 0.f) {  // fluted, lumpy, changing along the limb
                float lob = (noise3(d * 1.3f + c * 0.45f + V{L.seed, 0, 0}) - 0.5f) * 0.5f + (noise3(d * 3.1f + c * 1.3f) - 0.5f) * 0.18f;
                rad *= 1.f + wave * lob;
            }
            if (flare > 0.f) {  // root buttresses: lobes fanning out into the ground
                float roots = powf(fabsf(cosf(ang * 2.5f + L.seed)), 3.f) * 0.7f + 0.3f;
                rad += flare * roots * (0.7f + 0.6f * noise3(d * 2.f + V{L.seed, 3, 0}));
            }
            b.vtx(c + d * rad, (float)i / S * 2.f, v);
        }
    }
    for (size_t r = 0; r + 1 < ss.size(); r++)
        for (int i = 0; i < S; i++) {
            uint32_t q0 = base + (uint32_t)(r * (S + 1) + i), q1 = q0 + 1, q2 = q0 + S + 1, q3 = q2 + 1;
            b.tri(q0, q2, q1, mat);
            b.tri(q1, q2, q3, mat);
        }
    b.endPart();
}

void emitLimbs(Builder& b, const Plan& p, uint32_t mat, const std::vector<uint8_t>* has) {
    auto present = [&](int e) { return !has || (e < (int)has->size() && (*has)[e]); };
    struct Part { float s0, s1; int elem; };
    std::vector<Part> parts;
    for (int c = 0; c < (int)p.chains.size(); c++) {
        const Chain& ch = p.chains[c];
        if (ch.segs.empty()) continue;
        LimbPath path = limbPath(p, c);
        parts.clear();
        if (path.trunk)
            for (int k = 0; k < TRUNK_ELEMS; k++) parts.push_back({(float)k / TRUNK_ELEMS, (k + 1.f) / TRUNK_ELEMS, k});
        float off = path.trunk ? 1.f : 0.f;
        for (size_t j = 0; j < ch.segs.size(); j++) parts.push_back({off + j, off + j + 1.f, TRUNK_ELEMS + ch.segs[j]});
        for (size_t i = 0; i < parts.size();) {  // each unbroken run of present parts is one surface
            if (!present(parts[i].elem)) { i++; continue; }
            size_t j = i;
            while (j + 1 < parts.size() && present(parts[j + 1].elem)) j++;
            sweepLimb(b, path, parts[i].s0, parts[j].s1, mat);
            i = j + 1;
        }
    }
}
}  // namespace trees

// Wood mechanics of the procedural trees (trees.h): the beam skeleton of every variant, limb queries, mass and
// inertia of a part, impacts on standing wood, landings of loose wood, and the bookkeeping of breaks.
#include "vegetation/trees.h"

#include <cstring>

#include "core/constants.h"
#include "vegetation/trees/tree_internal.h"
#include "vegetation/trees/wood_shared.h"

namespace trees {
namespace {
Skel skelCache[VARIANTS];
bool skelValid[VARIANTS] = {};
}  // namespace

float tubeVolume(V a, V b, float ra, float rb) { return PI / 3.f * length(b - a) * (ra * ra + ra * rb + rb * rb); }

void resetSkeletons() { for (bool& b : skelValid) b = false; }

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
    for (Elem& e : S.e) e.mass = tubeVolume(e.a, e.b, e.ra, e.rb) * dens;
    for (const Foliage& f : p.leaves) {
        S.leafElem.push_back(f.owner >= 0 ? TRUNK_ELEMS + f.owner : trunkElem(f.c));
        S.leafMass.push_back(4.18879f * f.spread * f.spread * f.spread * LEAF_KG / std::max(f.n, 1));  // twigs and all, over the cluster's sphere
    }
    skelValid[v] = true;
    return S;
}

namespace {
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

void woodHydro(const Wood& w, std::vector<float>& tubes, std::vector<float>& plates) {
    const Skel& S = skel(w.v);
    const Plan& p = plan(w.v);
    tubes.clear();
    plates.clear();
    for (size_t k = 0; k < S.e.size() && k < w.has.size(); k++) {
        if (!w.has[k]) continue;
        const Elem& E = S.e[k];
        tubes.insert(tubes.end(), {E.a.x, E.a.y, E.a.z, E.b.x, E.b.y, E.b.z, E.ra, E.rb});
    }
    for (size_t i = 0; i < p.leaves.size() && i < w.leaves.size(); i++) {
        const Foliage& f = p.leaves[i];
        if (!w.has[S.leafElem[i]] || !w.leaves[i] || f.kind != F_LEAVES) continue;
        float leafArea = LEAF_AREA * f.size * f.size * p.leafWidth;  // a pointed oval, length x width
        plates.insert(plates.end(), {f.c.x, f.c.y, f.c.z, leafArea * w.leaves[i]});
    }
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
        V Mg = cross(L.mom[k] - a * L.m[k], V{0, -phys::GRAVITY, 0}), Mi = cross(q - a, f * F);
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

}  // namespace trees

#include "vegetation/trees/growth.h"

#include "vegetation/trees/foliage.h"

namespace trees {
namespace {
constexpr int MAX_FOLIAGE_TRIS = 220000;  // per tree: leaves / needles and their shoots
constexpr float TWIG_TIP_R = 0.004f;      // m, the thinnest a limb gets

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

int sidesFor(int level, float r) { return level == 0 ? 8 : (level == 1 ? 6 : (r > 0.015f ? 5 : 3)); }

struct Gen {
    const TreeParams& P;
    Plan& p;
    Rng rng;
    float H, boleY;

    // A limb grown from a along dir as a chain of segments, bending with gravity / light / gnarl; children spiral
    // off it; at the last level it carries leaf clusters. level 0 = the leader. parentR / parentDir: the limb it
    // grows out of (its collar), parentR 0 for the leader and for stems out of the ground.
    void branch(V a, V dir, float len, float r0, int level, int parent, float parentR, V parentDir) {
        int nseg = level == 0 ? 7 : (level == 1 ? 5 : 3);
        float sl = len / nseg;
        int chain = (int)p.chains.size();
        Chain ch;
        ch.sides = sidesFor(level, r0);
        ch.parentR = parentR;
        ch.parentDir = parentDir;
        p.chains.push_back(ch);
        std::vector<V> pts{a};
        std::vector<float> rad{r0};
        std::vector<int> seg;
        V d = dir, pos = a;
        float rTip = std::max(r0 * (level == 0 ? 0.08f : 0.3f), TWIG_TIP_R);
        for (int s = 0; s < nseg; s++) {
            float t = (s + 1.f) / nseg;
            float r1 = r0 + (rTip - r0) * t;
            V bend = V{0, 1, 0} * (P.upturn * (level ? 0.5f : 0.15f)) - V{0, 1, 0} * (P.gravity * t * (level ? 1.f : 0.f)) +
                     randUnit(rng) * (P.gnarl * 0.45f);
            d = norm(d + bend * (1.f / nseg) * 2.f);
            V q = pos + d * sl;
            p.branches.push_back({pos, q, rad.back(), r1, ch.sides, seg.empty() ? parent : seg.back(), level, chain});
            seg.push_back((int)p.branches.size() - 1);
            p.chains[chain].segs.push_back(seg.back());
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
            phi += GOLDEN_ANGLE + rng.in(-0.3f, 0.3f);
            float th = (P.branchAngle + rng.in(-P.angleVar, P.angleVar)) * PI / 180.f;
            if (level == 0) th *= 0.75f + 0.5f * t;  // lower branches flatter, upper steeper
            V perp = rot(perpTo(dd), dd, phi);
            V cd = norm(dd * cosf(th) + perp * sinf(th));
            float cl;
            if (level == 0) {
                float ratio = (H - c.y) / std::max(H - boleY, 0.5f);
                cl = (H - boleY) * P.lengthRatio * shapeRatio(P.crown, ratio) * rng.in(0.85f, 1.15f);
            } else cl = P.childLength * (len - 0.6f * t * len) * rng.in(0.8f, 1.15f);
            if (cl < 0.05f) continue;
            float cr = std::max(r * P.radiusRatio * (level == 0 ? 0.9f : 1.f), 0.005f);
            branch(c, cd, cl, cr, level + 1, ow, r, dd);
        }
        if (level > 0 && level == P.levels - 1 && P.leafDensity > 0.f) {  // the outer half of a sub-branch carries leaves between its twigs too
            for (float t : {0.55f, 0.8f, 1.f}) {
                V c, dd; float r; int ow;
                at(t, c, r, dd, ow);
                int nn = std::max(2, (int)lroundf(20.f * P.leafDensity));
                p.leaves.push_back({F_LEAVES, c, dd, P.leafSpread, P.leafSize, nn, rng.in(0.f, 1000.f), ow});
            }
        }
    }
};

}  // namespace

// over budget: thin every cluster evenly (bigger leaves keep the crown full)
void fitFoliageBudget(Plan& p) {
    long fixed = 0, perLeaf = 0;
    for (const Foliage& f : p.leaves) {
        fixed += foliageShootTris(p, f);
        perLeaf += foliageTris(p, f) - foliageShootTris(p, f);
    }
    if (fixed + perLeaf <= MAX_FOLIAGE_TRIS || perLeaf <= 0) return;
    float k = std::max((float)(MAX_FOLIAGE_TRIS - fixed), 0.1f * MAX_FOLIAGE_TRIS) / (float)perLeaf;
    for (Foliage& f : p.leaves) { f.n = std::max(1, (int)(f.n * k)); f.size *= 1.f / sqrtf(std::max(k, 0.25f)); }
}

Plan growTree(const TreeParams& P0, uint32_t seed, uint32_t leafMat, uint32_t woodMat) {
    if (P0.growth == GROWTH_COLONIZE) return colonizeTree(P0, seed, leafMat, woodMat);
    TreeParams P = P0;
    Rng rng{seed * 2654435761u + 1u};
    P.height *= rng.in(0.88f, 1.12f);
    Plan p;
    p.leafMat = leafMat;
    p.woodMat = woodMat;
    p.leafWidth = P.leafWidth;
    p.droop = P.leafDroop;
    p.upturn = P.upturn;
    p.needles = conifer(P);
    Gen g{P, p, rng, P.height, std::clamp(P.bole, 0.05f, 0.9f) * P.height};
    V lean = V{rng.in(-1.f, 1.f), 0, rng.in(-1.f, 1.f)} * (P.lean * g.boleY);
    V top = V{0, g.boleY, 0} + lean;
    p.trunk = {{0, -0.3f, 0}, top, P.trunkRadius * 1.15f, P.trunkRadius * 0.85f, 10, -1, -1};
    p.flare = P.flare;
    p.wave = P.trunkWave;
    p.seed = (float)(seed % 997u);
    float ll = (P.height - g.boleY) * (excurrent(P.crown) ? 1.f : 0.6f);
    g.branch(top, norm(V{0, 1, 0} + lean * (1.f / std::max(g.boleY, 0.5f))), ll, P.trunkRadius * 0.85f, 0, -1, 0.f, V{0, 1, 0});
    fitFoliageBudget(p);
    return p;
}

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
    p.upturn = P.upturn;
    p.trunkPath = false;
    Rng rng{seed * 3266489917u + 11u};
    Gen g{P, p, rng, B.size, 0.f};
    p.trunk = {{0, -0.05f, 0}, {0, 0.02f, 0}, 0.02f * B.size, 0.02f * B.size, 6, -1};  // root crown, hidden in the ground
    for (int i = 0; i < std::max(B.stems, 1); i++) {
        float a = TAU * (i + rng.in(-0.35f, 0.35f)) / std::max(B.stems, 1);
        V dir = norm(V{cosf(a) * B.spread, 1.f, sinf(a) * B.spread} + randUnit(rng) * 0.15f);
        V base{cosf(a) * 0.04f * B.size, 0.f, sinf(a) * 0.04f * B.size};
        g.branch(base, dir, B.size * rng.in(0.65f, 1.05f), 0.014f * B.size + 0.004f, 1, -1, 0.f, V{0, 1, 0});
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
}  // namespace trees

#include "vegetation/trees/foliage.h"

namespace trees {
namespace {
constexpr int LEAVES_PER_SHOOT = 9;
constexpr int WHORLS_PER_SHOOT = 10;           // conifer: needle whorls (leaf units) per shoot
constexpr int NEEDLES_PER_WHORL = 5;
constexpr int MAX_SHOOTS = 8;
constexpr int SHOOT_SIDES = 3;
constexpr float SHOOT_TILT_MIN = 0.45f, SHOOT_TILT_MAX = 0.95f;  // radians off the twig
constexpr float SHOOT_SPREAD = 0.15f;          // shoots leave the twig within this share of the cluster spread
constexpr float SHOOT_R_PER_LEAF = 0.03f;      // shoot base radius per metre of leaf length
constexpr float SHOOT_R_MIN = 0.0025f;         // m
constexpr float LEAF_ANGLE = 1.0f;             // radians between a leaf and its shoot
constexpr float LEAF_LIGHT_TURN = 0.25f;       // leaves turn up toward the sky by this much
constexpr float NEEDLE_ANGLE_MIN = 0.6f, NEEDLE_ANGLE_MAX = 1.1f;  // radians off the shoot, pointing forward
constexpr float NEEDLE_MIN_WIDTH = 0.0025f;    // m at the base
constexpr float FRUIT_STALK = 0.12f;           // stalk radius / fruit radius
constexpr int FRUIT_TRIS = 80 + 2 * 3;         // subdivided blob + its stalk

int shootSegments(const Plan& p) { return p.needles ? 1 : 2; }  // conifer shoots are stiff, leafy ones arch
int shootCount(const Plan& p, int n) {
    int per = p.needles ? WHORLS_PER_SHOOT : LEAVES_PER_SHOOT;
    return std::clamp((n + per - 1) / per, 1, MAX_SHOOTS);
}

void emitFruit(Builder& b, const Foliage& f, int count, const Plan& p, Rng& r) {
    V hang = f.c + V{0, 2.f * f.size, 0};  // the twig point the cluster hangs from (growBush puts it 2 sizes below)
    for (int i = 0; i < count; i++) {
        V c = f.c + randUnit(r) * (f.spread * 0.5f) - V{0, r.in(0.f, f.spread), 0};
        float s = f.size * r.in(0.8f, 1.2f);
        tube(b, hang, c + V{0, 0.9f * s, 0}, std::max(s * FRUIT_STALK, 0.001f), std::max(0.7f * s * FRUIT_STALK, 0.0008f), p.woodMat, 3);
        blob(b, c, {s, s * 1.05f, s}, f.seed + i, p.fruitMat, 1);
    }
}
}  // namespace

int foliageShootTris(const Plan& p, const Foliage& f) {
    return f.kind == F_FRUIT ? 0 : shootCount(p, f.n) * shootSegments(p) * SHOOT_SIDES * 2;
}

int foliageTris(const Plan& p, const Foliage& f) {
    if (f.kind == F_FRUIT) return f.n * FRUIT_TRIS;
    return foliageShootTris(p, f) + f.n * (p.needles ? NEEDLES_PER_WHORL : 4);
}

void emitFoliage(Builder& b, const Foliage& f, int count, const Plan& p) {
    Rng r{(uint32_t)(f.seed * 9973.f) + 17u};
    if (f.kind == F_FRUIT) { emitFruit(b, f, count, p, r); return; }
    // the shoots: fanning out from the twig in a spiral, reaching for the light, tips bent down by their leaves
    int K = shootCount(p, f.n), nodes = (f.n + K - 1) / K;
    float shootR = std::max(SHOOT_R_MIN, f.size * SHOOT_R_PER_LEAF);
    struct Shoot { V a, m, t; };
    Shoot sh[MAX_SHOOTS];
    V side0 = perpTo(f.dir);
    float phi = r.in(0.f, TAU);
    for (int k = 0; k < K; k++) {
        phi += GOLDEN_ANGLE + r.in(-0.25f, 0.25f);
        float tilt = r.in(SHOOT_TILT_MIN, SHOOT_TILT_MAX);
        V d = norm(f.dir * cosf(tilt) + rot(side0, f.dir, phi) * sinf(tilt) + V{0, 0.4f * p.upturn, 0});
        float L = f.spread * r.in(0.65f, 1.05f);
        V a = f.c + f.dir * (f.spread * SHOOT_SPREAD * r.in(-1.f, 1.f));
        V m = a + d * (0.5f * L);
        V t = m + (shootSegments(p) > 1 ? norm(d - V{0, 0.6f * p.droop, 0}) : d) * (0.5f * L);
        sh[k] = {a, m, t};
        if (shootSegments(p) > 1) {
            tube(b, a, m, shootR, shootR * 0.7f, p.woodMat, SHOOT_SIDES);
            tube(b, m, t, shootR * 0.7f, shootR * 0.35f, p.woodMat, SHOOT_SIDES);
        } else tube(b, a, t, shootR, shootR * 0.35f, p.woodMat, SHOOT_SIDES);
    }
    // leaf j: shoot j % K, node j / K counted from the tip down (the tip keeps its leaves longest)
    for (int j = 0; j < count; j++) {
        int k = j % K, node = j / K;
        const Shoot& s = sh[k];
        float u = 1.f - 0.85f * node / std::max(nodes - 1, 1);
        V q, sd;
        float rs;
        if (u > 0.5f) { float w = 2.f * u - 1.f; q = s.m + (s.t - s.m) * w; sd = norm(s.t - s.m); rs = shootR * (0.7f - 0.35f * w); }
        else { float w = 2.f * u; q = s.a + (s.m - s.a) * w; sd = norm(s.m - s.a); rs = shootR * (1.f - 0.3f * w); }
        float ang = node * GOLDEN_ANGLE + k * 1.3f + r.in(-0.3f, 0.3f);
        V out = rot(perpTo(sd), sd, ang);
        if (p.needles) {  // a whorl of needles around the shoot, angled forward
            float L = f.size * r.in(0.8f, 1.15f), W = std::max(NEEDLE_MIN_WIDTH, f.size * p.leafWidth * 0.5f);
            for (int nn = 0; nn < NEEDLES_PER_WHORL; nn++) {
                V o = rot(out, sd, TAU * nn / NEEDLES_PER_WHORL + r.in(-0.25f, 0.25f));
                float fa = r.in(NEEDLE_ANGLE_MIN, NEEDLE_ANGLE_MAX);
                V nd = norm(sd * cosf(fa) + o * sinf(fa) - V{0, 0.15f * p.droop, 0});
                V side = rot(norm(cross(nd, o)), nd, r.in(-0.8f, 0.8f));
                needle(b, q + o * rs, nd, side, L * r.in(0.85f, 1.1f), W, p.leafMat);
            }
            continue;
        }
        V ld = node == 0 ? sd : norm(sd * cosf(LEAF_ANGLE) + out * sinf(LEAF_ANGLE));
        ld = norm(ld + V{0, LEAF_LIGHT_TURN, 0});
        V up = fabsf(ld.y) > 0.95f ? perpTo(ld) : norm(V{0, 1, 0} - ld * ld.y);  // the blade faces the sky
        up = rot(up, ld, r.in(-0.6f, 0.6f));
        float L = f.size * r.in(0.75f, 1.25f);
        leaf(b, q + ld * rs, ld, up, L, L * p.leafWidth * 0.5f, p.droop * r.in(0.5f, 1.3f), p.leafMat);
    }
}
}  // namespace trees

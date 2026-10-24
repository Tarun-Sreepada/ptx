// Small plants (reeds, flowers, bushes, ferns), debris particles and ground clutter: one mesh per variant.
#include "vegetation/trees.h"

#include <cstring>

#include "core/jobs.h"
#include "vegetation/trees/foliage.h"
#include "vegetation/trees/limb_mesh.h"
#include "vegetation/trees/tree_internal.h"

namespace trees {
namespace {
// plant materials (indices into buildPlants' material list)
enum : uint32_t { P_STEM, P_CATTAIL, P_LEAF, P_YELLOW, P_PUFF, P_ROSE, P_ROSELEAF, P_BUSH_A, P_BUSH_B, P_BLOSSOM, P_SPRAY, P_CLOD, P_WOOD,
                  P_FERN, P_PEBBLE, P_TWIG, P_LITTER, P_BERRYLEAF, P_BERRY, P_EMBER, P_FIREFLY, P_FEATHER, P_WING_A, P_WING_B, P_INSECT };
const float COLORS[][3] = {{0.09f, 0.16f, 0.04f}, {0.13f, 0.06f, 0.025f}, {0.11f, 0.2f, 0.05f}, {0.8f, 0.55f, 0.02f}, {0.75f, 0.75f, 0.7f},
                           {0.45f, 0.012f, 0.02f}, {0.05f, 0.11f, 0.035f}, {0.08f, 0.17f, 0.04f}, {0.12f, 0.2f, 0.06f}, {0.75f, 0.6f, 0.7f},
                           {0.82f, 0.86f, 0.9f}, {0.11f, 0.08f, 0.05f}, {0.32f, 0.22f, 0.12f}, {0.06f, 0.14f, 0.035f}, {0.20f, 0.19f, 0.17f},
                           {0.12f, 0.085f, 0.055f}, {0.30f, 0.16f, 0.06f}, {0.f, 0.f, 0.f}, {0.f, 0.f, 0.f}, {0.05f, 0.02f, 0.01f},
                           {0.1f, 0.12f, 0.05f}, {0.09f, 0.08f, 0.075f}, {0.85f, 0.38f, 0.04f}, {0.85f, 0.85f, 0.8f}, {0.03f, 0.025f, 0.02f}};
const float EMBER_GLOW[3] = {6.f, 1.6f, 0.25f};     // radiance of a glowing ember (scene units)
const float FIREFLY_GLOW[3] = {14.f, 22.f, 3.f};    // a firefly's lantern: yellow-green
constexpr float THIN_TRANSLUCENCY = 0.4f;  // thin leaves, stems and petals pass light

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

// Generated bush (growBush): wood in twig brown, leaves on shoots, fruit on stalks.
void bush(Builder& b, const BushParams& B, uint32_t seed, uint32_t leafMat, uint32_t fruitMat) {
    Plan p = growBush(B, B.seed * 7919u + seed, leafMat, P_TWIG, fruitMat);
    emitLimbs(b, p, p.woodMat);
    for (const Foliage& f : p.leaves) emitFoliage(b, f, f.n, p);
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
    frond(rng.in(0, TAU), rng.in(0.5f, 0.7f), 0.12f, 0.1f, P_LITTER, P_LITTER);  // one dead frond lying on the ground
    int nf = 1 + (int)(rng() * 3);  // fiddleheads: short stalks ending in a tight spiral
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

// A bird, unit wingspan: a slim body along +z and two tapering wings along x (flapped by the vertex shader).
void bird(Builder& b) {
    blob(b, {0, 0, 0.02f}, {CRITTER_BODY, 0.05f, 0.22f}, 3.f, P_FEATHER, 1);
    blob(b, {0, 0.02f, 0.2f}, {0.04f, 0.04f, 0.05f}, 4.f, P_FEATHER, 1);  // head
    for (int s = -1; s <= 1; s += 2) {
        uint32_t a = b.vtx({s * CRITTER_BODY * 0.8f, 0.01f, 0.12f}, 0, 0), c = b.vtx({s * CRITTER_BODY * 0.8f, 0.01f, -0.1f}, 0, 1);
        uint32_t m1 = b.vtx({s * 0.3f, 0.02f, 0.1f}, 0.5f, 0), m2 = b.vtx({s * 0.3f, 0.02f, -0.08f}, 0.5f, 1), tip = b.vtx({s * 0.5f, 0.0f, -0.04f}, 1, 0.5f);
        b.tri(a, m1, c, P_FEATHER); b.tri(c, m1, m2, P_FEATHER); b.tri(m1, tip, m2, P_FEATHER);
        b.endPart();
    }
    uint32_t t0 = b.vtx({-0.03f, 0.f, -0.18f}, 0, 0), t1 = b.vtx({0.03f, 0.f, -0.18f}, 1, 0), t2 = b.vtx({0.f, 0.f, -0.32f}, 0.5f, 1);  // tail
    b.tri(t0, t1, t2, P_FEATHER);
    b.endPart();
}
// A butterfly, unit wingspan: a thin body, a big forewing and a smaller hindwing on each side.
void butterfly(Builder& b, uint32_t wing) {
    blob(b, {0, 0, 0}, {0.025f, 0.025f, 0.18f}, 6.f, P_INSECT, 1);
    for (int s = -1; s <= 1; s += 2) {
        auto quad = [&](V a, V c, V d, V e) {
            uint32_t i0 = b.vtx(a, 0, 0), i1 = b.vtx(c, 1, 0), i2 = b.vtx(d, 1, 1), i3 = b.vtx(e, 0, 1);
            b.tri(i0, i1, i2, wing); b.tri(i0, i2, i3, wing);
            b.endPart();
        };
        quad({s * CRITTER_BODY * 0.5f, 0, 0.1f}, {s * 0.42f, 0, 0.2f}, {s * 0.48f, 0, -0.02f}, {s * CRITTER_BODY * 0.5f, 0, -0.02f});   // forewing
        quad({s * CRITTER_BODY * 0.5f, 0, -0.02f}, {s * 0.36f, 0, -0.06f}, {s * 0.24f, 0, -0.26f}, {s * CRITTER_BODY * 0.5f, 0, -0.14f});  // hindwing
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
}  // namespace

void buildPlants(ModelData& out) {
    out = {};
    out.textures = {leafTexture()};
    auto color = [&](int i) -> const float* {  // bush leaf / fruit colours come from their sliders
        if (i == P_BUSH_A) return bushParams[BUSH_PLAIN].leafColor;
        if (i == P_BUSH_B) return bushParams[BUSH_FLOWER].leafColor;
        if (i == P_BLOSSOM) return bushParams[BUSH_FLOWER].fruitColor;
        if (i == P_ROSELEAF) return bushParams[BUSH_ROSE].leafColor;
        if (i == P_ROSE) return bushParams[BUSH_ROSE].fruitColor;
        if (i == P_BERRYLEAF) return bushParams[BUSH_BERRY].leafColor;
        if (i == P_BERRY) return bushParams[BUSH_BERRY].fruitColor;
        return COLORS[i];
    };
    for (int i = 0; i <= (int)P_INSECT; i++) {
        MaterialData m;
        memcpy(m.color, color(i), 12);
        bool leafy = i == P_LEAF || i == P_ROSELEAF || i == P_BUSH_A || i == P_BUSH_B || i == P_BERRYLEAF;
        m.tex = leafy || i == P_FERN || i == P_LITTER ? 0 : -1;
        if (leafy || i == P_STEM || i == P_BLOSSOM || i == P_FERN || i == P_LITTER) m.translucency = THIN_TRANSLUCENCY;
        if (i == P_PEBBLE) m.rough = 0.7f;  // weathered
        if (i == P_BERRY || i == P_ROSE) { m.rough = 0.25f; m.translucency = 0.25f; }  // glossy, skin lets light in
        if (i == P_SPRAY) { m.transmission = 0.9f; m.ior = 1.33f; m.rough = 0.04f; }  // water: clear drops that refract and sparkle
        if (i == P_EMBER) memcpy(m.emissive, EMBER_GLOW, 12);
        if (i == P_FIREFLY) memcpy(m.emissive, FIREFLY_GLOW, 12);
        if (i == P_WING_A || i == P_WING_B) m.translucency = 0.3f;  // thin wings glow in the sun
        out.materials.push_back(m);
    }
    out.meshes.resize(PLANT_VARIANTS);
    jobs::parallelFor(PLANT_VARIANTS, 1, [&](uint32_t first, uint32_t last) {
        for (int v = (int)first; v < (int)last; v++) {
            MeshData& m = out.meshes[v];
            Builder b{m};
            switch (v) {
            case PLANT_CATTAIL: cattails(b, 41u); break;
            case PLANT_DANDELION: dandelions(b, 53u); break;
            case PLANT_ROSE: bush(b, bushParams[BUSH_ROSE], 67u, P_ROSELEAF, P_ROSE); break;
            case PLANT_BUSH: bush(b, bushParams[BUSH_PLAIN], 79u, P_BUSH_A, 0); break;
            case PLANT_FLOWERBUSH: bush(b, bushParams[BUSH_FLOWER], 97u, P_BUSH_B, P_BLOSSOM); break;
            case PLANT_BERRYBUSH: bush(b, bushParams[BUSH_BERRY], 113u, P_BERRYLEAF, P_BERRY); break;
            case PLANT_SPRAY: blob(b, {0, 0, 0}, {1, 1, 1}, 5.f, P_SPRAY, 1); break;
            case PLANT_CLOD: blob(b, {0, 0, 0}, {1, 0.8f, 1}, 9.f, P_CLOD, 1); break;
            case PLANT_SPLINTER: tube(b, {0, -1, 0}, {0.1f, 1, 0}, 0.22f, 0.08f, P_WOOD, 5); break;
            case PLANT_LEAF: ribbon(b, {0, 0, -1}, {0, 0, 1}, 2.f, 0.5f, 0.15f, P_LEAF); break;
            case PLANT_FERN: fern(b, 131u); break;
            case PLANT_PEBBLES: pebbles(b, 151u); break;
            case PLANT_TWIGS: twigs(b, 173u); break;
            case PLANT_LITTER: litter(b, 191u); break;
            case PLANT_EMBER: blob(b, {0, 0, 0}, {1, 1, 1}, 13.f, P_EMBER, 0); break;
            case PLANT_FIREFLY: blob(b, {0, 0, 0}, {1, 1, 1.4f}, 17.f, P_FIREFLY, 0); break;
            case PLANT_BIRD: bird(b); break;
            case PLANT_BUTTERFLY: butterfly(b, P_WING_A); break;
            default: butterfly(b, P_WING_B); break;  // PLANT_BUTTERFLY2
            }
            finishMesh(m);
        }
    });
}
}  // namespace trees

// The procedural tree model: one mesh per variant (bark, leaf and wood materials, small generated textures).
#include "vegetation/trees.h"

#include "core/jobs.h"
#include "vegetation/trees/foliage.h"
#include "vegetation/trees/limb_mesh.h"
#include "vegetation/trees/tree_internal.h"

namespace trees {
namespace {
constexpr float LEAF_TRANSLUCENCY = 0.45f, LEAF_ROUGH = 0.55f;      // leaves glow when backlit; waxy cuticle
constexpr float NEEDLE_TRANSLUCENCY = 0.2f, NEEDLE_ROUGH = 0.45f;   // needles: thick, waxier
constexpr uint32_t BARK_W = 64, BARK_H = 128, LEAF_TEX = 128;

Plan planCache[VARIANTS];
bool planValid[VARIANTS] = {};
TreeParams grownAs[SPECIES];  // each species as this build grows it (the season applied)

uint32_t variantSeed(int v) { return treeParams[v / 2].seed * 7919u + (uint32_t)v * 101u + 13u; }

uint8_t srgb8(float lin) {
    lin = std::clamp(lin, 0.f, 1.f);
    float s = lin <= 0.0031308f ? lin * 12.92f : 1.055f * powf(lin, 1.f / 2.4f) - 0.055f;
    return (uint8_t)(s * 255.f + 0.5f);
}
// bark: vertical furrows (multiplied by each material's tint)
TextureData barkTex(const float col[3]) {
    TextureData t;
    t.w = BARK_W; t.h = BARK_H;
    t.rgba.resize(t.w * t.h * 4);
    for (uint32_t y = 0; y < t.h; y++)
        for (uint32_t x = 0; x < t.w; x++) {
            float fx = x / (float)BARK_W * 8.f, fy = y / (float)BARK_H * 4.f;
            float f = vnoise(fx * 2.f, fy * 0.6f) * 0.6f + vnoise(fx * 6.f, fy * 2.f) * 0.4f;
            float k = 0.55f + 0.6f * f;
            uint8_t* p = &t.rgba[(y * t.w + x) * 4];
            p[0] = srgb8(col[0] * k); p[1] = srgb8(col[1] * k); p[2] = srgb8(col[2] * k); p[3] = 255;
        }
    return t;
}
}  // namespace

// foliage: mottled with darker gaps (multiplied by each material's tint)
TextureData leafTexture() {
    TextureData t;
    t.w = t.h = LEAF_TEX;
    t.rgba.resize(t.w * t.h * 4);
    for (uint32_t y = 0; y < t.h; y++)
        for (uint32_t x = 0; x < t.w; x++) {
            float fx = x / (float)LEAF_TEX * 16.f, fy = y / (float)LEAF_TEX * 16.f;
            float f = vnoise(fx, fy) * 0.5f + vnoise(fx * 3.1f, fy * 3.1f) * 0.5f;
            float k = 0.45f + 0.8f * f * f;
            uint8_t* p = &t.rgba[(y * t.w + x) * 4];
            p[0] = p[1] = p[2] = srgb8(std::min(k, 1.f));
            p[3] = 255;
        }
    return t;
}

uint32_t barkMat(int v) { return v < 2 ? 0u : MAT_WOOD + (uint32_t)(v / 2); }

const Plan& plan(int v) {
    if (!planValid[v]) { planCache[v] = growTree(grownAs[v / 2], variantSeed(v), 1 + v, barkMat(v)); planValid[v] = true; }
    return planCache[v];
}

void resetPlans() { for (bool& b : planValid) b = false; }

void build(ModelData& out) {
    for (int s = 0; s < SPECIES; s++) grownAs[s] = seasonal(treeParams[s], s);
    resetPlans();
    resetSkeletons();
    out = {};
    out.textures = {barkTex(treeParams[0].barkColor), leafTexture()};  // species s > 0 bark: texture 1 + s
    for (int s = 1; s < SPECIES; s++) out.textures.push_back(barkTex(treeParams[s].barkColor));
    MaterialData bark;  // material 0: species A bark; the others' after the wood (MAT_WOOD + s)
    bark.tex = 0;
    out.materials.push_back(bark);
    for (int v = 0; v < VARIANTS; v++) {  // leaves: material 1 + v
        const TreeParams& P = grownAs[v / 2];
        Rng r{variantSeed(v) * 31u + 5u};
        MaterialData m;
        float h = P.hueVar * r.in(-0.5f, 0.5f);  // shift toward yellow / blue green
        m.color[0] = P.leafColor[0] * (1.f + h);
        m.color[1] = P.leafColor[1] * (1.f + 0.3f * h);
        m.color[2] = P.leafColor[2] * (1.f - h);
        m.color[3] = 1;
        m.tex = 1;
        bool needles = conifer(P);
        m.translucency = needles ? NEEDLE_TRANSLUCENCY : LEAF_TRANSLUCENCY;
        m.rough = needles ? NEEDLE_ROUGH : LEAF_ROUGH;
        out.materials.push_back(m);
    }
    MaterialData wood;  // fresh wood where a trunk or branch snapped (MAT_WOOD)
    wood.color[0] = 0.55f; wood.color[1] = 0.40f; wood.color[2] = 0.24f;
    out.materials.push_back(wood);
    for (int s = 1; s < SPECIES; s++) {
        MaterialData bk;
        bk.tex = 1 + s;
        out.materials.push_back(bk);
    }
    out.meshes.resize(VARIANTS);
    // every variant on its own worker: growth plan (plan(v) fills its own cache slot) and mesh
    jobs::parallelFor(VARIANTS, 1, [&](uint32_t first, uint32_t last) {
        for (int v = (int)first; v < (int)last; v++) {
            MeshData& m = out.meshes[v];
            Builder b{m};
            const Plan& p = plan(v);
            emitLimbs(b, p, p.woodMat);
            for (const Foliage& f : p.leaves) emitFoliage(b, f, f.n, p);
            finishMesh(m);
            skel(v);  // beam model now, not on the first hit
        }
    });
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

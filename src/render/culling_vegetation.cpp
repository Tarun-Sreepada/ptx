#include "render/culling_vegetation.h"
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include "core/prof.h"
#include "vegetation/trees.h"
#include "characters/character.h"
#include "world/world.h"
#include "app/settings.h"
#include "core/limits.h"
#include "core/math.h"
#include "destruction/thrown.h"
#include "destruction/tree_damage.h"
#include "render/culling.h"
#include "render/occlusion.h"
#include "scene/camera.h"
#include "scene/characters.h"
#include "scene/model_instances.h"
#include "scene/models.h"
#include "scene/vegetation.h"
#include "sim/critters.h"
#include "sim/debris.h"

namespace {
// how much each plant gives to the wind (sway.glsl: integer part of the phase); -1 = rigid ground clutter
int plantFlex(int v) {
    switch (v) {
    case trees::PLANT_CATTAIL: return 4;
    case trees::PLANT_DANDELION: return 3;
    case trees::PLANT_FERN: return 2;
    case trees::PLANT_ROSE: case trees::PLANT_BUSH: case trees::PLANT_FLOWERBUSH: case trees::PLANT_BERRYBUSH: return 1;
    default: return -1;
    }
}
float swayPhase(float x, float z) { return hash2((int)floorf(x * 3.f), (int)floorf(z * 3.f)) * 0.999f; }

// indices of list sorted by .v (counting sort: stable, linear); variants >= n are left out
template <class T> void bucketByVariant(const std::vector<T>& list, uint32_t n, std::vector<uint32_t>& order) {
    static std::vector<uint32_t> start;
    start.assign(n + 1, 0);
    for (const T& e : list) if (e.v < n) start[e.v + 1]++;
    for (uint32_t v = 0; v < n; v++) start[v + 1] += start[v];
    order.resize(start[n]);
    for (uint32_t i = 0; i < (uint32_t)list.size(); i++) if (list[i].v < n) order[start[list[i].v]++] = i;
}
}  // namespace

// Trees for the raster G-buffer: frustum-cull bounding spheres; near ones join the model draws (one instanced
// run per variant), far ones become impostors, and the 20 m band in between draws both with a dithered split.
// Trees and plants carry their height and a phase for swaying in the wind (raster only, sway.glsl).
void cullTrees() {
    PROF_FUNC();
    numTreeNear = numImpostors = 0;
    if (!S.rasterPrimary) return;
    auto* g = (GpuModel*)treeModel.gpu;
    float planes[5][4];
    frustumPlanes(planes);
    float* xf = (float*)modelXfBuf.map;
    float* ib = (float*)impBuf.map;
    const float d0 = S.impostorDist, band = 20.f;
    uint32_t first = numModelVisible;
    // instances grouped by variant in one pass (an instanced run per variant), not a pass over the list per variant
    static std::vector<uint32_t> order;
    bucketByVariant(treeList, (uint32_t)numTreeVariants, order);
    for (uint32_t v = 0, oi = 0; v < (uint32_t)numTreeVariants; v++) {
        uint32_t runStart = numModelVisible;
        for (; oi < order.size() && treeList[order[oi]].v == v; oi++) {
            const TreeInst& t = treeList[order[oi]];
            float c = cosf(t.yaw), sn = sinf(t.yaw), sc = t.scale;
            V3 lc{treeC[v][0], treeC[v][1], treeC[v][2]};
            V3 C = t.p + V3{c * lc.x + sn * lc.z, lc.y, -sn * lc.x + c * lc.z} * sc;
            float R = treeR[v] * sc;
            bool vis = true;
            for (int i = 0; i < 5 && vis; i++) vis = planes[i][0] * C.x + planes[i][1] * C.y + planes[i][2] * C.z + planes[i][3] >= -R;
            if (!vis || occluded(C, R)) continue;
            V3 dd = C - cam.pos;
            float fade = std::clamp((sqrtf(dot(dd, dd)) - d0) / band, 0.f, 1.f);
            if (fade < 1.f && numModelVisible < MAX_MODEL_INST) {
                uint32_t k = numModelVisible++;
                float m[12] = {c * sc, 0, sn * sc, t.p.x, 0, sc, 0, t.p.y, -sn * sc, 0, c * sc, t.p.z};
                memcpy(xf + k * 16, m, 48);
                memcpy(xf + k * 16 + 12, &g->meshes[v].slot, 4);
                xf[k * 16 + 13] = fade;
                xf[k * 16 + 14] = treeTop[v] * sc;
                xf[k * 16 + 15] = swayPhase(t.p.x, t.p.z);  // trees: flexibility 0
            }
            if (fade > 0.f && numImpostors < (uint32_t)MAX_TREES) {
                float e[12] = {t.p.x, t.p.y, t.p.z, sc, t.yaw, (float)v, fade, 0, lc.x, lc.y, lc.z, treeR[v]};
                memcpy(ib + numImpostors * 12, e, 48);
                numImpostors++;
            }
        }
        if (numModelVisible > runStart) drawRuns.push_back({g, v, runStart, numModelVisible - runStart});
    }
    numTreeNear = numModelVisible - first;

    // plants: always the real mesh (they're placed only where they still cover a few pixels)
    auto* pg = (GpuModel*)plantModel.gpu;
    first = numModelVisible;
    bucketByVariant(plantList, (uint32_t)trees::PLANT_VARIANTS, order);
    for (uint32_t v = 0, oi = 0; pg && v < (uint32_t)trees::PLANT_VARIANTS; v++) {
        uint32_t runStart = numModelVisible;
        for (; oi < order.size() && plantList[order[oi]].v == v; oi++) {
            const PlantInst& p = plantList[order[oi]];
            if (numModelVisible >= MAX_MODEL_INST) continue;
            const float* m = p.m;
            V3 C{m[0] * plantC[v][0] + m[1] * plantC[v][1] + m[2] * plantC[v][2] + m[3], m[4] * plantC[v][0] + m[5] * plantC[v][1] + m[6] * plantC[v][2] + m[7],
                 m[8] * plantC[v][0] + m[9] * plantC[v][1] + m[10] * plantC[v][2] + m[11]};
            float R = plantR[v] * m[5];
            bool vis = true;
            for (int i = 0; i < 5 && vis; i++) vis = planes[i][0] * C.x + planes[i][1] * C.y + planes[i][2] * C.z + planes[i][3] >= -R;
            if (!vis || occluded(C, R)) continue;
            uint32_t k = numModelVisible++;
            memcpy(xf + k * 16, m, 48);
            memcpy(xf + k * 16 + 12, &pg->meshes[v].slot, 4);
            int flex = plantFlex((int)v);
            xf[k * 16 + 13] = 0;
            xf[k * 16 + 14] = flex >= 0 ? plantTop[v] * m[5] : 0.f;
            xf[k * 16 + 15] = flex >= 0 ? (float)flex + swayPhase(m[3], m[11]) : 0.f;
        }
        if (numModelVisible > runStart) drawRuns.push_back({pg, v, runStart, numModelVisible - runStart});
    }
    numPlantsDrawn = numModelVisible - first;

    // moving things, every frame: debris particles (one run per kind), fragments and stumps (a run each)
    auto push = [&](GpuModel* gm, uint32_t mesh, const float m[3][4], const float (*prev)[4] = nullptr) {
        if (numModelVisible >= MAX_MODEL_INST) return;
        uint32_t k = numModelVisible++;
        memcpy(xf + k * 16, m, 48);
        memcpy(xf + k * 16 + 12, &gm->meshes[mesh].slot, 4);
        xf[k * 16 + 13] = xf[k * 16 + 14] = xf[k * 16 + 15] = 0;
        if (prev) setPrevXf(k, prev, gm->meshes[mesh].slot);
    };
    for (int kind = 0; pg && kind < DEB_KINDS; kind++) {
        uint32_t runStart = numModelVisible;
        for (const Droplet& d : spray) {
            if (d.delay > 0.f || d.kind != kind) continue;
            float m[3][4], pm[3][4];
            dropletXf(d, m);
            Droplet pd = d;
            if (d.hasPrev) pd.p = d.prevP;
            dropletXf(pd, pm);
            push(pg, (uint32_t)debrisMesh[kind], m, pm);
        }
        if (numModelVisible > runStart) drawRuns.push_back({pg, (uint32_t)debrisMesh[kind], runStart, numModelVisible - runStart});
    }
    // small animals (raster only): one run per kind; wings flap in the vertex shader (z < 0: beats per second)
    for (int mesh : {trees::PLANT_FIREFLY, trees::PLANT_BIRD, trees::PLANT_BUTTERFLY, trees::PLANT_BUTTERFLY2}) {
        uint32_t runStart = numModelVisible;
        for (const CritterDraw& c : critterDraws()) {
            if (c.mesh != mesh || !pg) continue;
            uint32_t k = numModelVisible;
            push(pg, (uint32_t)mesh, c.m, c.prev);
            if (numModelVisible == k) break;
            if (c.flapAmp > 0.f) {
                xf[k * 16 + 14] = -std::max(c.flapHz, 1e-4f);
                xf[k * 16 + 15] = c.flapPhase * 0.999f + floorf(c.flapAmp * 10.f + 0.5f);
            }
        }
        if (numModelVisible > runStart) drawRuns.push_back({pg, (uint32_t)mesh, runStart, numModelVisible - runStart});
    }
    // rock fragments: every piece is its own mesh, so one run each
    fragsDrawn = 0;
    for (const Thrown& t : thrown) {
        auto* fg = t.frag ? (GpuModel*)t.frag->model.gpu : nullptr;
        if (!fg) continue;
        uint32_t runStart = numModelVisible;
        push(fg, t.fragMesh, t.xf.m, t.hasPrev ? t.prevXf.m : t.xf.m);
        if (numModelVisible > runStart) { drawRuns.push_back({fg, t.fragMesh, runStart, 1}); fragsDrawn++; }
    }
    for (const Stump& st : stumps) {
        auto* sg = (GpuModel*)st.frag->model.gpu;
        if (!sg) continue;
        uint32_t runStart = numModelVisible;
        push(sg, st.mesh, st.xf.m);
        if (numModelVisible > runStart) drawRuns.push_back({sg, st.mesh, runStart, 1});
    }
    // characters: the model's indices, the character's own skinned vertices (its mesh slot)
    for (character::Character* c : character::all()) {
        auto* gc = (GpuChar*)c->user;
        if (!gc) continue;
        V3 C{0.5f * (c->bmin[0] + c->bmax[0]), 0.5f * (c->bmin[1] + c->bmax[1]), 0.5f * (c->bmin[2] + c->bmax[2])};
        V3 e{c->bmax[0] - C.x, c->bmax[1] - C.y, c->bmax[2] - C.z};
        float R = sqrtf(dot(e, e));
        bool vis = true;
        for (int i = 0; i < 5 && vis; i++) vis = planes[i][0] * C.x + planes[i][1] * C.y + planes[i][2] * C.z + planes[i][3] >= -R;
        if (!vis || occluded(C, R)) continue;
        Xf w = character::worldXf(*c);
        auto* mg = (GpuModel*)gc->asset->model->gpu;
        for (auto& cm : gc->meshes) {
            if (numModelVisible >= MAX_MODEL_INST) break;
            uint32_t k = numModelVisible++;
            memcpy(xf + k * 16, w.m, 48);
            memcpy(xf + k * 16 + 12, &cm.slot, 4);
            xf[k * 16 + 13] = xf[k * 16 + 14] = xf[k * 16 + 15] = 0;
            setPrevXf(k, gc->hasPrev ? gc->prevW.m : w.m, gc->frames > 1 ? cm.prevSlot : cm.slot);
            drawRuns.push_back({mg, cm.mesh, k, 1});
        }
    }
    finishPrevXf();
}

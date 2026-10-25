#include "destruction/thrown.h"
#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <vulkan/vulkan.h>
#include "physics/physics.h"
#include "world/terrain.h"
#include "app/settings.h"
#include "core/limits.h"
#include "core/math.h"
#include "destruction/impacts.h"
#include "destruction/tree_damage.h"
#include "destruction/wood.h"
#include "gpu/accel.h"
#include "gpu/memory.h"
#include "scene/camera.h"
#include "scene/model_instances.h"
#include "scene/models.h"
#include "scene/rocks.h"
#include "scene/scene_gpu.h"
#include "scene/vegetation.h"
#include "sim/debris.h"
#include "sim/water.h"
#include "sim/water_splash.h"
#include "world/terrain_query.h"

std::vector<std::unique_ptr<FragSet>> fragSets;

uint32_t numThrownRock = 0;  // thrown that use the rock mesh (raster rock list)

uint32_t fragsDrawn = 0;

std::vector<Thrown> thrown;

uint32_t numThrown = 0;

int autoThrow = 0;

bool treeLog = false;

int treeThrow = 0;  // test: rocks at the nearest tree (trunk, limbs, crown in turn), hits logged

// One user of a fragment model gone; the model is freed with the last (after the GPU is done with it).
void releaseFrag(FragSet* fs) {
    if (--fs->live > 0) return;
    auto it = std::find_if(fragSets.begin(), fragSets.end(), [&](auto& p) { return p.get() == fs; });
    if (it == fragSets.end()) return;
    std::shared_ptr<FragSet> keep(it->release());
    fragSets.erase(it);
    // instances went away this frame; the last frame using them finishes before retireNext runs
    retireNext.push_back([keep]() { if (keep->model.gpu) releaseModel(keep->model); });
}

// Remove thrown[i]: body, own hull, and its fragment model once no piece uses it (freed after the GPU is done).
void dropThrown(size_t i) {
    Thrown& t = thrown[i];
    physics::removeBodies(&t.body, 1);
    if (t.shape) physics::releaseShape(t.shape);
    if (t.frag) releaseFrag(t.frag);
    thrown.erase(thrown.begin() + i);
}

// Water on thrown bodies before the physics step: buoyancy and drag per element of the body (sim/hydro.h) against
// the local current, so rocks sink and roll along the bed, wood floats and drifts, and a flat fast stone skips.
// Every touch of the surface is a splash event; something moving through the surface leaves a trail of small
// ones (wake). Hard landings go to landImpact.
namespace {
constexpr float SPLASH_ENTRY_FRACTION = 0.02f;  // submerged share that counts as having entered the water
constexpr float WAKE_SPLASH_INTERVAL = 0.07f;   // s between the small splashes of something cutting the surface
constexpr float LEAF_TRANSLUCENCY = 0.2f;       // materials at least this translucent are leaves: area, no volume

constexpr float GREEN_WOOD_DENSITY = 850.f;     // kg/m^3, wood pieces without a limb model

std::shared_ptr<const HydroShape> hydroShapeOf(const Thrown& t, float mass) {
    if (!t.frag) return hydroRockVariant(t.rockVar);
    if (t.woodPart) {  // limbs as tubes, leaf clusters as plates; tree local -> body space: woodScale * p - woodCtr
        std::vector<float> tubes, plates;
        trees::woodHydro(*t.woodPart, tubes, plates);
        const float* ctr = &t.woodCtr.x;
        for (size_t i = 0; i + 8 <= tubes.size(); i += 8) {
            for (int j = 0; j < 6; j++) tubes[i + j] = tubes[i + j] * t.woodScale - ctr[j % 3];
            tubes[i + 6] *= t.woodScale;
            tubes[i + 7] *= t.woodScale;
        }
        for (size_t i = 0; i + 4 <= plates.size(); i += 4) {
            for (int j = 0; j < 3; j++) plates[i + j] = plates[i + j] * t.woodScale - ctr[j];
            plates[i + 3] *= t.woodScale * t.woodScale;
        }
        return hydroFromTubes(tubes, plates);
    }
    const ModelData& md = t.frag->model.data;
    if (t.fragMesh >= md.meshes.size()) return nullptr;
    const MeshData& m = md.meshes[t.fragMesh];
    std::vector<uint8_t> solid(m.idx.size() / 3, 1);
    for (size_t k = 0; k < solid.size() && k < m.triMat.size(); k++)
        if (m.triMat[k] < md.materials.size() && md.materials[m.triMat[k]].translucency >= LEAF_TRANSLUCENCY) solid[k] = 0;
    float solidDensity = t.wood ? GREEN_WOOD_DENSITY : t.density;  // a crown's hull density is mostly air
    return hydroFromMesh(m.verts.data(), 8, m.idx.data(), (uint32_t)(m.idx.size() / 3), mass / std::max(solidDensity, 1.f), &solid);
}
}  // namespace

void updateThrown(float dt) {
    for (size_t i = 0; i < thrown.size();) {
        Thrown& t = thrown[i];
        float v[3], w[3], mass;
        if (!physics::bodyState(t.body, t.sc, t.xf, v, w, mass) || t.xf.m[1][3] < -200.f) {
            dropThrown(i);
            continue;
        }
        // settled debris retires: chips after a few seconds at rest, bigger pieces once out of the way. Every piece
        // is its own body and BLAS, so a few hundred of them cost GI / BVH time for nothing.
        float sp2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
        t.rest = sp2 < 0.04f ? t.rest + dt : 0.f;
        if (t.frag || t.wood) {
            float cdx = t.xf.m[0][3] - cam.pos.x, cdz = t.xf.m[2][3] - cam.pos.z, cd2 = cdx * cdx + cdz * cdz;
            if ((t.radius < 0.15f && t.rest > 3.f) || (t.radius < 0.4f && t.rest > 15.f && cd2 > 30.f * 30.f) ||
                (t.rest > 60.f && cd2 > 60.f * 60.f)) {
                dropThrown(i);
                continue;
            }
        }
        i++;
        t.age += dt;
        float v0[3];
        memcpy(v0, t.v, 12);
        memcpy(t.v, v, 12);
        memcpy(t.w0, t.w, 12);
        memcpy(t.w, w, 12);
        t.cooldown -= dt;
        float x = t.xf.m[0][3], y = t.xf.m[1][3], z = t.xf.m[2][3], r = t.radius;
        if (t.woodPart) {  // a broken limb tumbles no faster than air drag on its leaves and twigs lets it (~2 turns/s)
            float w2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2];
            if (w2 > 12.f * 12.f) { float k = 12.f / sqrtf(w2); for (float& c : w) c *= k; physics::setAngularVelocity(t.body, w); }
        }
        if (sp2 > 4.f && !t.woodPart && t.radius > 0.03f) {  // a rock (or chunk of one) moving: through any thin limbs?
            V3 p1{x, y, z}, vv{v[0], v[1], v[2]};
            sweepThinLimbs(t, p1 - vv * dt, p1, vv, mass);
        }
        float dv[3] = {v[0] - v0[0], v[1] - v0[1], v[2] - v0[2]};
        bool hardHit = dv[0] * dv[0] + (dv[1] + 9.8f * dt) * (dv[1] + 9.8f * dt) + dv[2] * dv[2] > 9.f;
        if (t.woodPart && !hardHit) {  // a falling trunk lands mostly by its spin stopping (the far end slams down)
            float dw[3] = {w[0] - t.w0[0], w[1] - t.w0[1], w[2] - t.w0[2]};
            hardHit = (dw[0] * dw[0] + dw[1] * dw[1] + dw[2] * dw[2]) * r * r > 9.f;
        }
        float level = waterLevelAt(x, z);
        bool overWater = level > -1e8f;
        RiverInfo ri = riverInfo(x, z);
        if (!overWater || y - r > level) {
            if (y - r > level + 0.05f) t.wet = false;
            if (hardHit && t.cooldown <= 0.f) landImpact(t, v0, v, mass, dt);
            continue;
        }
        // water: buoyancy and drag per surface element (sim/hydro.h), against the local current
        float cur[2];
        riverVelocity(x, z, S.flow * (1.f + 1.6f * riverSwell / std::max(gRiver.depth, 0.3f)), cur);
        if (fabsf(ri.d) > ri.w + 1.f) cur[0] = cur[1] = 0.f;  // a crater pond: still water
        if (!t.hydro) t.hydro = hydroShapeOf(t, mass);
        HydroBody hb{t.body, t.xf, t.sc, {v[0], v[1], v[2]}, {w[0], w[1], w[2]}, {}, mass};
        float frac = 0.f;
        if (t.hydro && physics::bodyCom(t.body, hb.com)) frac = applyWater(*t.hydro, hb, {level, {cur[0], cur[1]}, ri.depth}, dt);
        float vh = sqrtf(v[0] * v[0] + v[2] * v[2]);
        float sz = r / 0.85f;
        if (!t.wet && frac > SPLASH_ENTRY_FRACTION) {  // splash-down (again on every skip)
            t.wet = true;
            waterImpact({x, level, z}, sz, v0);
        } else if (t.wet && frac < 0.95f && vh > 1.f && (t.trail -= dt) <= 0.f) {  // cutting through the surface
            t.trail = WAKE_SPLASH_INTERVAL;
            waterImpact({x, level, z}, sz * 0.5f, v, std::min(0.12f * vh, 0.5f));
        }
        if (hardHit && t.cooldown <= 0.f && y - r < heightFn(x, z) + 0.3f) landImpact(t, v0, v, mass, dt);  // hit the bed
    }
}

// Moving things into the TLAS after the placed instances (every frame): thrown rocks (rock-mesh ones also into
// the raster rock list), fragments and tree pieces (their own meshes), debris particles, stumps.
void writeThrown() {
    numThrown = (uint32_t)thrown.size();
    numThrownRock = 0;
    auto* inst = (VkAccelerationStructureInstanceKHR*)instBuf.map;
    float* xf = (float*)rockXf.map;
    uint32_t base = INST_MODELS + numModelInst + numRocks + numTrees + numPlants;
    for (uint32_t i = 0; i < numThrown; i++) {
        const Thrown& t = thrown[i];
        auto* g = t.frag ? (GpuModel*)t.frag->model.gpu : nullptr;
        if (t.frag && g) {
            const GpuMesh& gm = g->meshes[t.fragMesh];
            setInstance(inst[base + i], 3 + gm.slot, gm.blasAddr, t.xf.m);
        } else {
            setRockInstance(inst[base + i], t.rockVar, t.xf.m);
            if (!t.frag) {
                writeRockXf(xf, numRocks + numThrownRock, t.xf.m, t.hasPrev ? t.prevXf.m : t.xf.m, t.rockVar);
                numThrownRock++;
            }
        }
    }
    numSprayInst = numFallen = 0;
    auto* pg = (GpuModel*)plantModel.gpu;
    if (pg)
        for (const Droplet& d : spray) {
            if (d.delay > 0.f) continue;
            const GpuMesh& dm = pg->meshes[debrisMesh[d.kind]];
            float m[3][4];
            dropletXf(d, m);
            setInstance(inst[base + numThrown + numSprayInst++], 3 + dm.slot, dm.blasAddr, m);
        }
    for (const Stump& st : stumps) {
        auto* sg = (GpuModel*)st.frag->model.gpu;
        if (!sg) continue;
        const GpuMesh& sm = sg->meshes[st.mesh];
        setInstance(inst[base + numThrown + numSprayInst + numFallen++], 3 + sm.slot, sm.blasAddr, st.xf.m);
    }
}

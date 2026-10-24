#include "destruction/metal.h"
#include <vector>
#include <unordered_map>
#include <memory>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <utility>
#include "physics/physics.h"
#include "world/world.h"
#include "core/math.h"
#include "core/random.h"
#include "destruction/buildings.h"
#include "gpu/memory.h"
#include "scene/models.h"
#include "scene/stream_sync.h"
#include "sim/debris.h"

// Metal: a blow below tearing energy dents it. The triangles near the contact are subdivided down to ~4 cm and
// pushed in along the blow with a smooth falloff (the whole thickness moves, like a plate bending); the dented
// copy replaces the mesh like a breach does, so dents add up.
bool dentMetal(const Placed& hit, V3 p, V3 dir, float En) {
    const MeshData& src = hit.model->data.meshes[hit.mesh];
    float s = std::max(src.strength, 0.1f);
    float depth = std::clamp(0.015f + 0.3f * sqrtf(En / (s * 1e6f)), 0.01f, 0.5f);
    float R = 0.2f + 1.6f * depth;
    Xf inv = xfInverse(hit.xf);
    float pw[3] = {p.x, p.y, p.z}, pl[3];
    xfPoint(inv, pw, pl);
    float dw[3] = {dir.x, dir.y, dir.z}, dl[3];
    for (int i = 0; i < 3; i++) dl[i] = inv.m[i][0] * dw[0] + inv.m[i][1] * dw[1] + inv.m[i][2] * dw[2];
    float dn = sqrtf(dl[0] * dl[0] + dl[1] * dl[1] + dl[2] * dl[2]);
    for (float& x : dl) x /= std::max(dn, 1e-6f);
    // subdivide near the contact (verts are pos3 nrm3 uv2)
    std::vector<float> V = src.verts;
    std::vector<uint32_t> I = src.idx, M = src.triMat;
    if (M.empty()) M.assign(I.size() / 3, 0u);
    auto P = [&](uint32_t i) { return V3{V[i * 8], V[i * 8 + 1], V[i * 8 + 2]}; };
    V3 c{pl[0], pl[1], pl[2]};
    const float target = 0.04f;
    for (int pass = 0; pass < 10; pass++) {
        std::vector<uint32_t> nI, nM;
        bool any = false;
        std::unordered_map<uint64_t, uint32_t> mids;
        auto mid = [&](uint32_t a, uint32_t b) {
            uint64_t k = ((uint64_t)std::min(a, b) << 32) | std::max(a, b);
            auto it = mids.find(k);
            if (it != mids.end()) return it->second;
            uint32_t n = (uint32_t)(V.size() / 8);
            for (int j = 0; j < 8; j++) V.push_back(0.5f * (V[a * 8 + j] + V[b * 8 + j]));
            mids[k] = n;
            return n;
        };
        for (size_t t = 0; t < I.size(); t += 3) {
            uint32_t a = I[t], b = I[t + 1], d = I[t + 2];
            V3 pa = P(a), pb = P(b), pd = P(d);
            float e = std::max({vlen(pb - pa), vlen(pd - pb), vlen(pa - pd)});
            V3 ctr = (pa + pb + pd) * (1.f / 3.f);
            if (e > target && vlen(ctr - c) < R + e && nI.size() < 600000) {
                uint32_t ab = mid(a, b), bd = mid(b, d), da = mid(d, a);
                nI.insert(nI.end(), {a, ab, da, ab, b, bd, da, bd, d, ab, bd, da});
                nM.insert(nM.end(), {M[t / 3], M[t / 3], M[t / 3], M[t / 3]});
                any = true;
            } else { nI.insert(nI.end(), {a, b, d}); nM.push_back(M[t / 3]); }
        }
        I.swap(nI); M.swap(nM);
        if (!any) break;
    }
    // edges split on one side only leave T-junctions; with every vertex displaced by a smooth field they stay tiny
    std::vector<bool> moved(V.size() / 8, false);
    for (size_t i = 0; i < V.size() / 8; i++) {
        V3 q = P((uint32_t)i), d = q - c;
        float along = d.x * dl[0] + d.y * dl[1] + d.z * dl[2];
        V3 lat = d - V3{dl[0], dl[1], dl[2]} * along;
        float x = vlen(lat) / R;
        if (x >= 1.f) continue;
        float f = (1.f - x * x) * (1.f - x * x);
        for (int k = 0; k < 3; k++) V[i * 8 + k] += dl[k] * depth * f;
        moved[i] = true;
    }
    // flat normals for the dented triangles (shared verts would smear across the plate's corners)
    std::vector<float> outV;
    std::vector<uint32_t> outI;
    outV.reserve(V.size());
    std::vector<uint32_t> remap(V.size() / 8, UINT32_MAX);
    for (size_t t = 0; t < I.size(); t += 3) {
        bool dented = moved[I[t]] || moved[I[t + 1]] || moved[I[t + 2]];
        V3 a = P(I[t]), b = P(I[t + 1]), d = P(I[t + 2]);
        V3 n = normalize(cross(b - a, d - a));
        for (int j = 0; j < 3; j++) {
            uint32_t vi = I[t + j];
            if (!dented && remap[vi] != UINT32_MAX) { outI.push_back(remap[vi]); continue; }
            uint32_t ni = (uint32_t)(outV.size() / 8);
            outV.insert(outV.end(), &V[vi * 8], &V[vi * 8 + 8]);
            if (dented) { outV[ni * 8 + 3] = n.x; outV[ni * 8 + 4] = n.y; outV[ni * 8 + 5] = n.z; }
            else remap[vi] = ni;
            outI.push_back(ni);
        }
    }
    auto copy = std::make_shared<Model>();
    copy->state = 1;
    ModelData& cd = copy->data;
    const ModelData& sd = hit.model->data;
    cd.materials = sd.materials;
    std::unordered_map<int, int> texMap;
    for (auto& mt : cd.materials)
        forEachTex(mt, [&](int t) {
            if (t < 0) return t;
            auto [e, fresh] = texMap.emplace(t, (int)cd.textures.size());
            if (fresh) cd.textures.push_back(sd.textures[t]);
            return e->second;
        });
    MeshData dm;
    dm.verts = std::move(outV);
    dm.idx = std::move(outI);
    dm.triMat = std::move(M);
    dm.strength = src.strength;
    dm.breakKind = src.breakKind;
    for (int k = 0; k < 3; k++) { dm.bmin[k] = 1e30f; dm.bmax[k] = -1e30f; }
    for (size_t i = 0; i < dm.verts.size(); i += 8)
        for (int k = 0; k < 3; k++) { dm.bmin[k] = std::min(dm.bmin[k], dm.verts[i + k]); dm.bmax[k] = std::max(dm.bmax[k], dm.verts[i + k]); }
    dm.shape = physics::meshShape(dm.verts.data(), 8, (uint32_t)(dm.verts.size() / 8), dm.idx.data(), (uint32_t)(dm.idx.size() / 3));
    cd.meshes.push_back(std::move(dm));
    if (!uploadModel(*copy)) {
        if (cd.meshes[0].shape) physics::releaseShape(cd.meshes[0].shape);
        return false;
    }
    std::shared_ptr<Model> old = stream::overridePlaced(stream::placedKey(hit), copy);
    damagedModels.push_back(copy);
    if (old) {
        damagedModels.erase(std::remove(damagedModels.begin(), damagedModels.end(), old), damagedModels.end());
        retireNext.push_back([old]() {
            if (old->gpu) releaseModel(*old);
            if (old->data.meshes[0].shape) physics::releaseShape(old->data.meshes[0].shape);
        });
    }
    onStreamChanged();
    for (int i = 0; i < 12; i++)  // a few sparks / flakes of paint
        addDebris(DEB_DIRT, p - dir * 0.05f, (rndDir() - dir) * (1.f + 2.f * rnd01()), 0.006f + 0.01f * rnd01());
    return true;
}

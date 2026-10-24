// Acoustic geometry: simplified meshes, their instances in the world, and the ground patch around the listener.
#include "audio/audio_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

#include <meshoptimizer.h>

namespace audio {
uint32_t meshCreate(const float* verts, uint32_t stride, uint32_t nverts, const uint32_t* idx, uint32_t ntris, Material mat) {
    if (!ok) return UINT32_MAX;
    uint32_t id;
    if (!freeMesh.empty()) { id = freeMesh.back(); freeMesh.pop_back(); }
    else id = nextMesh++;
    liveMeshes++;
    // copy positions now (the caller's data may go away), simplify + build on the sim thread
    std::vector<float> pos(nverts * 3);
    for (uint32_t i = 0; i < nverts; i++) memcpy(&pos[i * 3], verts + (size_t)i * stride, 12);
    std::vector<uint32_t> ind(idx, idx + ntris * 3);
    post([id, pos = std::move(pos), ind = std::move(ind), mat]() mutable {
        if (meshes.size() <= id) meshes.resize(id + 1);
        // acoustics don't need visual detail: ~25% of the triangles at 2% error keeps occluders and room shapes intact
        size_t target = std::max<size_t>(ind.size() / 4 / 3 * 3, 36);
        if (ind.size() > 300) {
            std::vector<uint32_t> s(ind.size());
            s.resize(meshopt_simplify(s.data(), ind.data(), ind.size(), pos.data(), pos.size() / 3, 12, target, 0.02f, 0, nullptr));
            if (s.size() >= 3) ind.swap(s);
        }
        // compact to the used vertices
        std::vector<uint32_t> remap(pos.size() / 3, UINT32_MAX);
        std::vector<IPLVector3> v;
        std::vector<IPLTriangle> t(ind.size() / 3);
        for (size_t i = 0; i < ind.size(); i++) {
            uint32_t& r = remap[ind[i]];
            if (r == UINT32_MAX) { r = (uint32_t)v.size(); v.push_back({pos[ind[i] * 3], pos[ind[i] * 3 + 1], pos[ind[i] * 3 + 2]}); }
            t[i / 3].indices[i % 3] = (IPLint32)r;
        }
        std::vector<IPLint32> mi(t.size(), 0);
        IPLMaterial m = kMaterials[mat];
        SubMesh sm;
        IPLSceneSettings scs{};
        scs.type = sceneType;
        scs.embreeDevice = embree;
        iplSceneCreate(ctx, &scs, &sm.scene);
        IPLStaticMeshSettings ms{(IPLint32)v.size(), (IPLint32)t.size(), 1, v.data(), t.data(), mi.data(), &m};
        iplStaticMeshCreate(sm.scene, &ms, &sm.mesh);
        iplStaticMeshAdd(sm.mesh, sm.scene);
        iplSceneCommit(sm.scene);
        sm.tris = (uint32_t)t.size();
        meshes[id] = sm;
    });
    return id;
}

void meshRelease(uint32_t id) {
    if (!ok || id == UINT32_MAX) return;
    freeMesh.push_back(id);
    liveMeshes--;
    post([id] {
        SubMesh& m = meshes[id];
        if (m.mesh) iplStaticMeshRelease(&m.mesh);
        if (m.scene) iplSceneRelease(&m.scene);  // instances hold their own reference to the sub-scene
        m = {};
    });
}

uint32_t instanceAdd(uint32_t mesh, const Xf& xf) {
    if (!ok || mesh == UINT32_MAX) return UINT32_MAX;
    uint32_t id;
    if (!freeInst.empty()) { id = freeInst.back(); freeInst.pop_back(); }
    else id = nextInst++;
    liveInsts++;
    post([id, mesh, xf] {
        if (instances.size() <= id) instances.resize(id + 1);
        if (mesh >= meshes.size() || !meshes[mesh].scene) return;
        IPLInstancedMeshSettings is{};
        is.subScene = meshes[mesh].scene;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 4; c++) is.transform.elements[r][c] = xf.m[r][c];
        is.transform.elements[3][3] = 1.f;
        iplInstancedMeshCreate(scene, &is, &instances[id].m);
        iplInstancedMeshAdd(instances[id].m, scene);
        instances[id].tris = meshes[mesh].tris;
        sceneTris += meshes[mesh].tris;
    });
    return id;
}

void instanceRemove(uint32_t id) {
    if (!ok || id == UINT32_MAX) return;
    freeInst.push_back(id);
    liveInsts--;
    post([id] {
        if (id >= instances.size() || !instances[id].m) return;
        iplInstancedMeshRemove(instances[id].m, scene);
        iplInstancedMeshRelease(&instances[id].m);
        sceneTris -= instances[id].tris;
        instances[id] = {};
    });
}

void setTerrain(const float* verts, uint32_t nverts, const uint32_t* idx, uint32_t ntris) {
    if (!ok) return;
    std::vector<IPLVector3> v(nverts);
    for (uint32_t i = 0; i < nverts; i++) v[i] = {verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2]};
    std::vector<IPLTriangle> t(ntris);
    for (uint32_t i = 0; i < ntris; i++) t[i] = {{(IPLint32)idx[i * 3], (IPLint32)idx[i * 3 + 1], (IPLint32)idx[i * 3 + 2]}};
    post([v = std::move(v), t = std::move(t)]() mutable {
        if (terrain.mesh) { iplStaticMeshRemove(terrain.mesh, scene); iplStaticMeshRelease(&terrain.mesh); }
        std::vector<IPLint32> mi(t.size(), 0);
        IPLMaterial m = kMaterials[MAT_GROUND];
        IPLStaticMeshSettings ms{(IPLint32)v.size(), (IPLint32)t.size(), 1, v.data(), t.data(), mi.data(), &m};
        iplStaticMeshCreate(scene, &ms, &terrain.mesh);
        iplStaticMeshAdd(terrain.mesh, scene);
        terrain.tris = (uint32_t)t.size();
    });
}

}  // namespace audio

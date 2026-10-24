#include "scene/forest.h"
#include <vector>
#include <string>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <cstddef>
#include <utility>
#include "vegetation/trees.h"
#include "world/world.h"
#include "scene/vegetation.h"

// Generate + upload the tree variants, build their trunk colliders and bake the impostor atlas by ray tracing
// the real meshes (TLAS temporarily holds only the variants, parked far below the terrain).
// forest.txt -> forestAssets (loaded now: the impostor atlas needs the variant count before buildScene)
void loadForest(const std::string& worldDir) {
    for (int v = 0; v < trees::VARIANTS; v++) treeWeight[v] = 1.f;
    std::ifstream f(worldDir + "forest.txt");
    std::string line;
    float builtin = -1.f;
    while (std::getline(f, line)) {
        std::istringstream ls(line);
        std::string a, name;
        if (!(ls >> a) || a[0] == '#') continue;
        if (a == "builtin") { ls >> builtin; continue; }
        if (a == "spacing") { ls >> forestSpacing; continue; }
        float w = (float)atof(a.c_str());
        std::getline(ls >> std::ws, name);
        if (w <= 0.f || name.empty() || numTreeVariants >= MAX_TREE_VARIANTS) continue;
        ModelData md;
        std::string err;
        auto t0 = std::chrono::steady_clock::now();
        if (!loadGltf(assetFile(worldDir, name), md, err)) { printf("forest: %s: %s\n", name.c_str(), err.c_str()); continue; }
        printf("forest: %s as tree variant %d (%.1f s)\n", name.c_str(), numTreeVariants,
               std::chrono::duration<float>(std::chrono::steady_clock::now() - t0).count());
        forestAssets.push_back(std::move(md));
        treeWeight[numTreeVariants++] = w;
    }
    // listed assets replace the built-in trees unless "builtin <w>" keeps a share of them
    float bw = builtin >= 0.f ? builtin : (forestAssets.empty() ? 1.f : 0.f);
    for (int v = 0; v < trees::VARIANTS; v++) treeWeight[v] = bw / trees::VARIANTS;
    treeWeightSum = 0.f;
    for (int v = 0; v < numTreeVariants; v++) treeWeightSum += treeWeight[v];
}

// one mesh (instances baked in, materials offset into treeModel) for an imported forest asset
MeshData mergeAsset(const ModelData& a, uint32_t matBase) {
    MeshData out;
    for (auto& in : a.instances) {
        if (in.mesh < 0 || in.mesh >= (int)a.meshes.size()) continue;
        const MeshData& m = a.meshes[in.mesh];
        uint32_t base = (uint32_t)out.verts.size() / 8;
        const float(*x)[4] = in.xf.m;
        for (size_t i = 0; i + 8 <= m.verts.size(); i += 8) {
            const float* v = &m.verts[i];
            float n[3];
            for (int k = 0; k < 3; k++) {
                out.verts.push_back(x[k][0] * v[0] + x[k][1] * v[1] + x[k][2] * v[2] + x[k][3]);
                n[k] = x[k][0] * v[3] + x[k][1] * v[4] + x[k][2] * v[5];
            }
            float l = sqrtf(n[0] * n[0] + n[1] * n[1] + n[2] * n[2]);
            for (int k = 0; k < 3; k++) out.verts.push_back(l > 0 ? n[k] / l : 0.f);
            out.verts.push_back(v[6]);
            out.verts.push_back(v[7]);
        }
        for (uint32_t i : m.idx) out.idx.push_back(base + i);
        for (uint32_t t : m.triMat) out.triMat.push_back(t + matBase);
    }
    for (int k = 0; k < 3; k++) { out.bmin[k] = 1e30f; out.bmax[k] = -1e30f; }
    for (size_t i = 0; i < out.verts.size(); i += 8)
        for (int k = 0; k < 3; k++) { out.bmin[k] = std::min(out.bmin[k], out.verts[i + k]); out.bmax[k] = std::max(out.bmax[k], out.verts[i + k]); }
    return out;
}

// Trunk collider for an imported tree: an 8-sided cylinder around the vertices 0.3-1.5 m up (the trunk), to 60% height.
void assetTrunk(const MeshData& m, std::vector<float>& cv, std::vector<uint32_t>& ci) {
    double sx = 0, sz = 0;
    size_t n = 0;
    std::vector<float> d;
    for (size_t i = 0; i < m.verts.size(); i += 8)
        if (m.verts[i + 1] > m.bmin[1] + 0.3f && m.verts[i + 1] < m.bmin[1] + 1.5f) { sx += m.verts[i]; sz += m.verts[i + 2]; n++; }
    float cx = n ? (float)(sx / n) : 0.f, cz = n ? (float)(sz / n) : 0.f;
    for (size_t i = 0; i < m.verts.size(); i += 8)
        if (m.verts[i + 1] > m.bmin[1] + 0.3f && m.verts[i + 1] < m.bmin[1] + 1.5f)
            d.push_back(hypotf(m.verts[i] - cx, m.verts[i + 2] - cz));
    float r = 0.2f;
    if (!d.empty()) { std::nth_element(d.begin(), d.begin() + d.size() * 3 / 4, d.end()); r = std::clamp(d[d.size() * 3 / 4], 0.08f, 1.2f); }
    float y0 = m.bmin[1], y1 = m.bmin[1] + 0.6f * (m.bmax[1] - m.bmin[1]);
    cv.clear(); ci.clear();
    for (int j = 0; j < 2; j++)
        for (int k = 0; k < 8; k++) {
            float a = k * 6.2831853f / 8;
            cv.insert(cv.end(), {cx + r * cosf(a), j ? y1 : y0, cz + r * sinf(a)});
        }
    for (int k = 0; k < 8; k++) {
        uint32_t a = k, b = (k + 1) % 8;
        ci.insert(ci.end(), {a, b, 8 + b, a, 8 + b, 8 + a});
        if (k >= 2) { ci.insert(ci.end(), {0, (uint32_t)k, (uint32_t)k - 1, 8, 8 + (uint32_t)k - 1, 8 + (uint32_t)k}); }
    }
}

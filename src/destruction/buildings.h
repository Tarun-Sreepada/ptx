#pragma once
#include <vector>
#include <unordered_map>
#include <array>
#include <memory>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <utility>
#include "physics/fracture.h"
#include "world/world.h"
#include "core/math.h"

// breakable buildings (breakStatic): connected components of a static mesh (triangles, local AABB)
struct MeshParts { std::vector<std::vector<uint32_t>> tris; std::vector<std::array<float, 6>> box; };

extern std::unordered_map<const MeshData*, MeshParts> meshPartsCache;

extern std::vector<std::shared_ptr<Model>> damagedModels;

struct WallBreakReq { uint32_t body; V3 p, dir, v; float En, reach; };

extern std::vector<WallBreakReq> pendingWallBreaks;

// ---------------------------------------------------------------- breakable buildings
// Static meshes (walls, floors, crates, ...) are unions of convex pieces, mostly boxes: a mesh's connected
// components. A hard enough blow breaches the components it reaches: each is shattered around the contact
// (fracture::shatter); cells inside the breach radius fly off as fragments (thrown, and can break again), cells
// that lost every hold on the rest of the structure fall too, and the remaining cells stay as static geometry in a
// damaged copy of the mesh that replaces it in the world (stream::overridePlaced, survives cell reloads).
// Breach area = En / WALL_GC.
inline constexpr float WALL_GC = 2e5f;  // J per m^2 of breach (brick / plaster wall incl. what goes into dust)

inline const MeshParts& meshParts(const MeshData& m) {
    auto it = meshPartsCache.find(&m);
    if (it != meshPartsCache.end()) return it->second;
    uint32_t nv = (uint32_t)m.verts.size() / 8;
    // weld by position (1 mm), then union-find over triangles
    std::vector<uint32_t> par(nv);
    for (uint32_t i = 0; i < nv; i++) par[i] = i;
    auto find = [&](uint32_t x) { while (par[x] != x) x = par[x] = par[par[x]]; return x; };
    auto unite = [&](uint32_t a, uint32_t b) { a = find(a); b = find(b); if (a != b) par[a] = b; };
    std::unordered_map<uint64_t, uint32_t> weld;
    for (uint32_t i = 0; i < nv; i++) {
        auto q = [&](int k) { return (uint64_t)(uint32_t)(int32_t)lroundf(m.verts[i * 8 + k] * 1000.f) & 0x1FFFFF; };
        auto [w, fresh] = weld.emplace(q(0) | q(1) << 21 | q(2) << 42, i);
        if (!fresh) unite(i, w->second);
    }
    for (size_t t = 0; t < m.idx.size(); t += 3) { unite(m.idx[t], m.idx[t + 1]); unite(m.idx[t], m.idx[t + 2]); }
    MeshParts mp;
    std::unordered_map<uint32_t, uint32_t> comp;
    for (size_t t = 0; t < m.idx.size(); t += 3) {
        uint32_t root = find(m.idx[t]);
        auto [c, fresh] = comp.emplace(root, (uint32_t)mp.tris.size());
        if (fresh) { mp.tris.emplace_back(); mp.box.push_back({1e30f, 1e30f, 1e30f, -1e30f, -1e30f, -1e30f}); }
        mp.tris[c->second].push_back((uint32_t)(t / 3));
        auto& b = mp.box[c->second];
        for (int j = 0; j < 3; j++)
            for (int k = 0; k < 3; k++) {
                float x = m.verts[m.idx[t + j] * 8 + k];
                b[k] = std::min(b[k], x); b[3 + k] = std::max(b[3 + k], x);
            }
    }
    return meshPartsCache.emplace(&m, std::move(mp)).first->second;
}

Xf xfInverse(const Xf& a);

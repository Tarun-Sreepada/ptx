// Light BVH for many-light sampling (lights.glsl): point lights and emissive triangles under one binary tree whose
// nodes hold bounds and summed intensity. The shader walks it from the root, picking each child by its estimated
// contribution at the shading point (intensity / distance^2, zero when the box is behind the surface), so a
// light's selection probability follows how much it matters there instead of 1 / light count.
#pragma once
#include <cstdint>
#include <vector>

namespace lighttree {
struct Prim { float mn[3], mx[3]; float power; };            // power: luminous intensity (point) or radiance * area (triangle)
struct Node { float mn[3]; float power; float mx[3]; uint32_t child; };  // child: right child (left = this + 1), or LEAF | prim
constexpr uint32_t LEAF = 0x80000000u;
static_assert(sizeof(Node) == 32);
// Depth-first node order, root at 0. Empty input gives no nodes.
void build(const std::vector<Prim>& prims, std::vector<Node>& out);
// Vose alias table over prim powers: per entry (probability of keeping it, alias index, this prim's pdf, -).
struct AliasEntry { float prob; uint32_t alias; float pdf; float pad; };
void buildAlias(const std::vector<Prim>& prims, std::vector<AliasEntry>& out);
// size x size tileable blue noise, 4 independent channels interleaved (void-and-cluster), values in (0, 1).
void blueNoise(int size, std::vector<float>& out);
}  // namespace lighttree

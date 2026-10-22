// Runtime fracture of convex solids: impact-aligned Voronoi shattering, after Mueller, Chentanez & Kim,
// "Real Time Dynamic Fracture with Volumetric Approximate Convex Decompositions" (SIGGRAPH 2013): crack seeds
// cluster around the contact point and spread in a cone along the impact direction, each seed's Voronoi cell is
// clipped out of the solid, and every piece is again convex (so it can be a hull body and break again).
#pragma once
#include <cstdint>
#include <vector>

#include "world.h"

namespace fracture {
struct Vec { float x, y, z; };
// Convex polyhedron as outward-facing (CCW from outside) polygons. inner = surface created by a fracture.
struct Face { std::vector<Vec> pts; bool inner = false; };
struct Poly { std::vector<Face> faces; };

// k-DOP approximation of a point cloud's convex hull (support planes in `dirs` directions, at least 26).
Poly fromPoints(const float* pts, uint32_t strideFloats, uint32_t n, int dirs = 64);
// Keep the part with dot(nrm, x) <= d; the cut face is marked inner.
Poly clip(const Poly& p, Vec nrm, float d);
float volume(const Poly& p, Vec* centroid = nullptr);
float radius(const Poly& p, Vec c);  // bounding radius about c
void translate(Poly& p, Vec by);

// Break p into Voronoi cells of `count` seeds. impact = contact point, dir = direction the blow travels into the
// solid (unit). spread 0..1: 0 = seeds tight in the crush zone and an impact cone, 1 = spread through the solid.
void shatter(const Poly& p, Vec impact, Vec dir, int count, float spread, uint32_t seed, std::vector<Poly>& out);

// Flat-shaded triangle mesh (pos3 nrm3 uv2), planar UVs at uvScale per metre; outer / inner faces get their own
// material index.
void toMesh(const Poly& p, MeshData& m, uint32_t matOuter, uint32_t matInner, float uvScale = 1.5f);
}  // namespace fracture

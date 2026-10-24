// Appending triangle meshes (pos3 nrm3 uv2 vertices) for the vegetation generators, and the small shapes they are
// made of. Normals are accumulated per part from its faces and normalised when the part ends. Leaves and needles
// are single sided: the shaders turn their normals to the viewer.
#pragma once
#include "vegetation/trees/tree_math.h"

namespace trees {
struct Builder {
    MeshData& m;
    size_t partStart = 0;
    uint32_t vtx(V p, float u, float v);
    V pos(uint32_t i) const { return {m.verts[i * 8], m.verts[i * 8 + 1], m.verts[i * 8 + 2]}; }
    void tri(uint32_t a, uint32_t b, uint32_t c, uint32_t mat);
    void endPart();
};

// Tapered tube a -> c (open ends: caps hide in the canopy / ground).
void tube(Builder& b, V a, V c, float ra, float rc, uint32_t mat, int sides = 8);
// Lumpy closed blob (subdivided icosahedron), flattened underneath.
void blob(Builder& b, V c, V r, float seed, uint32_t mat, int subdiv = 2);
// One leaf: a pointed oval from base along d, its face toward up, curling down by droop (fraction of its length).
void leaf(Builder& b, V base, V d, V up, float L, float W, float droop, uint32_t mat);
// One needle: a thin triangle from base along d, width W across side.
void needle(Builder& b, V base, V d, V side, float L, float W, uint32_t mat);
// Flat tapering strip from base along dir, sagging by droop * len at the tip; jag notches every other edge segment.
void ribbon(Builder& b, V base, V dir, float len, float w, float droop, uint32_t mat, bool jag = false);
// Bounding box of the finished mesh.
void finishMesh(MeshData& m);
}  // namespace trees

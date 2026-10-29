// Vegetation meshlets (render/veg_gpu.cpp buildMeshlets): per meshlet two uvec4, (vertex offset, vertex count |
// triangle count << 8, offset of its triangles' indices in the mesh, packed triangle offset) and (bounding sphere
// centre, radius as float bits); meshletData holds the vertex ids (into the mesh), the triangles' local corners
// (8 bits each) and each triangle's index in the mesh.
#ifndef SCENE_MESHLET
#define SCENE_MESHLET
layout(binding = 107, std430) readonly buffer VegCmdR { uint vegCmd[]; };
layout(binding = 110, std430) readonly buffer Meshlets { uvec4 meshlets[]; };
layout(binding = 111, std430) readonly buffer MeshletData { uint meshletData[]; };
layout(binding = 112, std430) readonly buffer VegBuckets { uvec4 vegBuckets[]; };

struct MeshletPayload { uint inst; uint ids[32]; };

vec4 meshletSphere(uint m) { return uintBitsToFloat(meshlets[2u * m + 1u]); }
#endif

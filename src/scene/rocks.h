#pragma once
#include <vector>
#include <cstdint>
#include <vulkan/vulkan.h>

extern std::vector<float> rockPts;        // rock mesh vertices of all variants (pos3 nrm3, rockVerts each)
extern std::vector<uint32_t> rockTriIdx;  // their shared triangle list (CPU copy: hulls, hydrodynamics)

extern void* rockHull[16];

// Every rock variant has its own seed, so its own cleavage planes, wear, proportions, bedding and erosion. All
// share the subdivided-icosahedron topology (one index list, ROCK_VERTS vertices each); instances pick a variant.
inline constexpr int ROCK_VARIANTS = 16;

void buildRockMesh(uint32_t seed, std::vector<float>& verts, std::vector<uint32_t>& idx);

uint32_t rockVarOf(float x, float z);

void setRockInstance(VkAccelerationStructureInstanceKHR& in, uint32_t var, const float m[3][4]);

void writeRockXf(float* xf, uint32_t slot, const float m[3][4], const float prev[3][4], uint32_t var);

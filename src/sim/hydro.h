// Water acting on loose bodies (thrown rocks, fragments, tree pieces). A body is described once by
//  - volume elements: where its volume is (chunks along its length and around its axis, or along each limb)
//  - face elements: its surface, binned by position and by the way it faces
// Every physics step each volume element gets the buoyancy of its submerged share, at its own position (a log
// floats flat, a rock rolls to sit upright), and each submerged face the push of the water: pressure on faces
// moving into it (~1/2 rho v_n^2 per m^2, Newtonian flow: Cd ~0.5 for a sphere, ~1.15 for a plate face on), a
// little suction behind, skin friction along it. Velocities are relative to the local current (fastest at the
// surface), so wood drifts downstream; drag on spinning faces brakes tumbling; a flat stone meeting the water at a
// shallow angle gets the lift that makes it skip. Drag is clamped so it can stop a body but never throw it back.
#pragma once
#include <cstdint>
#include <memory>
#include <vector>

#include "world/world.h"

struct HydroShape {
    struct Vol { float p[3], v, size; };  // volume v (m^3) centred at p; size: its height (m), for partial submersion
    struct Face { float p[3], n[3], a; }; // area a (m^2) facing n (outward), centred at p
    std::vector<Vol> vols;
    std::vector<Face> faces;
};

// From a triangle mesh (pos3 first in each vertex of stride floats) in body space, whose solid holds volume (m^3,
// body space units). solid (optional, per triangle): 0 for thin parts (leaves) that catch water but hold none.
std::shared_ptr<const HydroShape> hydroFromMesh(const float* verts, uint32_t stride, const uint32_t* idx, uint32_t ntris,
                                                float volume, const std::vector<uint8_t>* solid = nullptr);
// From tapered tubes (wood: a3 b3 ra rb each) and thin plates (leaf clusters: centre3 one-sided area each), body space.
std::shared_ptr<const HydroShape> hydroFromTubes(const std::vector<float>& tubes, const std::vector<float>& plates);
// The shared rock mesh variant's (unit scale).
std::shared_ptr<const HydroShape> hydroRockVariant(uint32_t var);

struct HydroBody {                 // a body this step, world space
    uint32_t id;
    Xf xf;                         // body space -> world: rotation x scale, translation
    float scale;                   // the uniform scale in xf
    float v[3], w[3], com[3], mass;
};
struct HydroWater { float level, cur[2], depth; };  // surface height, surface current (m/s, xz), channel depth
// Applies the water's forces for a step of dt; returns the submerged share of the volume (0..1).
float applyWater(const HydroShape& shape, const HydroBody& b, const HydroWater& w, float dt);

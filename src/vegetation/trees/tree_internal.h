// Shared by the tree build, the wood mechanics and the wood meshes: every variant's growth plan and beam skeleton,
// generated once per build() (impacts query the limbs, breaks need the parts).
#pragma once
#include <vector>

#include "vegetation/trees/growth.h"

namespace trees {
inline constexpr float WOOD_KG = 700.f, LEAF_KG = 6.f;  // per m^3: green wood; a crown's twigs + leaves per m^3 of envelope
inline constexpr float LEAF_AREA = 0.55f;               // a leaf's area / (length x width): pointed oval

const Plan& plan(int v);
uint32_t barkMat(int v);
void resetPlans();      // after the parameters changed: regrow on next use
void resetSkeletons();
TextureData leafTexture();  // mottled grey, tinted by each leaf material

// The tree as beams (wood mechanics): TRUNK_ELEMS trunk elements, then one per limb segment (parents before
// children). t0..t1: the part of the trunk; leafMass: per leaf of each cluster (kg).
struct Elem { V a, b; float ra, rb; int parent, level; float t0, t1, mass; };
struct Skel { std::vector<Elem> e; std::vector<int> leafElem; std::vector<float> leafMass; };
const Skel& skel(int v);

float tubeVolume(V a, V b, float ra, float rb);
}  // namespace trees

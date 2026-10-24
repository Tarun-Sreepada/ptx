// Geometry helpers shared by the wood mechanics and the wood meshes.
#pragma once
#include <vector>

#include "vegetation/trees/tree_internal.h"

namespace trees {
// closed prism along a..c, appended (pos3)
void prism(V a, V c, float ra, float rc, int S, std::vector<float>& verts, std::vector<uint32_t>& idx);
// limbs thick enough for a physics collider (thrown things hit them; thinner ones are swept, woodSweep)
inline constexpr float COLLIDER_MIN_R = 0.035f;
bool colliderLimb(const Limb& l);
bool isCollider(const Skel& S, int k);
void colliderOf(const Plan& p, const std::vector<bool>* skip, std::vector<float>& verts, std::vector<uint32_t>& idx);
// distance of q from segment a..b, t: the closest point's parameter
float segDist(V q, V a, V b, float& t);
}  // namespace trees

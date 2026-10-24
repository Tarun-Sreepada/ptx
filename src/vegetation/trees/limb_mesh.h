// Limb surfaces. A chain of limb segments (growth.h) is one tube swept along a smooth curve through its joints, so
// it bends without gaps or creases; its rings are carried along by parallel transport so they don't twist; its
// cross-section is lumpy (fluted, changing along it) as a function of world position. The trunk path (trunk +
// leader) flares into root buttresses at the ground and its centre line wanders a little; a branch swells into a
// collar where it grows out of its parent. Any stretch of a path can be built alone (the pieces of a broken limb)
// and meets the rest exactly.
#pragma once
#include <vector>

#include "vegetation/trees/growth.h"
#include "vegetation/trees/mesh_builder.h"

namespace trees {
// A path's parameter s runs over its segments: segment i from s = i to i + 1. The trunk path's segment 0 is the
// trunk, its segment 1 + i the leader's segment i.
struct LimbPath {
    std::vector<V> pts;        // joints, base to tip
    std::vector<float> r;      // radius at each joint
    std::vector<float> arc;    // distance along the limb at each joint
    std::vector<float> rings;  // where the rings go (s), base to tip
    int sides = 6;
    float wave = 0.f;                    // lumpy cross-section
    float flare = 0.f, flareR = 0.f;     // root buttresses: amount, trunk base radius
    float wander = 0.f;                  // trunk centre line wander (m), over segment 0
    float collarR = 0.f, collarExit = 0.f, collarSwell = 0.f;  // parent radius, distance to where it leaves the parent, swell
    float seed = 0.f;
    bool trunk = false;
};
LimbPath limbPath(const Plan& p, int chain);
// The surface of path from s0 to s1.
void sweepLimb(Builder& b, const LimbPath& path, float s0, float s1, uint32_t mat);
// Every limb of the plan, or (has set) the stretches whose wood elements are present: has is in the wood mechanics'
// element order, TRUNK_ELEMS pieces of the trunk then one per p.branches entry.
void emitLimbs(Builder& b, const Plan& p, uint32_t mat, const std::vector<uint8_t>* has = nullptr);
}  // namespace trees

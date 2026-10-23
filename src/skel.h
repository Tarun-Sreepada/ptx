// Skeletal animation on the CPU: sample glTF clips into a local pose (per joint t3 r4 s3), blend poses, and turn
// a pose into model-space joint transforms and skinning matrices (joint global x inverse bind). The GPU applies
// the matrices (skin.comp); the ragdoll (character.h) replaces the globals of its bones while it's limp.
#pragma once
#include <vector>

#include "world.h"

namespace skel {
using Pose = std::vector<float>;  // 10 floats per joint: translation, rotation (xyzw), scale

void restPose(const SkinData& s, Pose& out);
// overwrites the joints the clip animates (others keep what's in p); t wraps when loop
void sample(const ClipData& c, float t, bool loop, Pose& p);
// out = a..b by w (nlerp for rotations); out may alias a
void blend(const Pose& a, const Pose& b, float w, Pose& out);
Xf localXf(const float* trs);
// model-space joint transforms, parents first (s.order)
void globals(const SkinData& s, const Pose& p, std::vector<Xf>& out);
// per joint 3x4 rows (12 floats): global x invBind, the GPU's skinning matrices
void skinMatrices(const SkinData& s, const std::vector<Xf>& globals, float* out);
Xf inverse(const Xf& a);  // affine inverse (any invertible 3x3)
}  // namespace skel

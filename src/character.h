// Animated characters: a glTF skin posed from its clips (crossfaded), with a ragdoll built from the same skeleton.
// While animated the ragdoll's bodies are kinematic and follow the pose (they push rocks and the player); a hard
// hit (or ragdoll()) makes them dynamic, and the physics then poses the skeleton until the character gets up.
// Renderer-agnostic: main.cpp uploads the model, applies skinMats on the GPU (skin.comp) and refits the BLAS.
#pragma once
#include <cstdint>
#include <functional>
#include <vector>

#include "physics.h"
#include "skel.h"
#include "world.h"

namespace character {
enum Mode { ANIMATED, RAGDOLL };

struct Character {
    const ModelData* data = nullptr;
    int skin = 0;
    void* user = nullptr;            // the renderer's per-character state
    float pos[3] = {}, yaw = 0.f, scale = 1.f, footOffset = 0.f;  // world placement: feet on the ground at pos
    // animation: clip (playing) crossfading into next over fadeLen seconds
    int clip = -1, next = -1;
    float t = 0.f, tNext = 0.f, fade = 0.f, fadeLen = 0.3f, rate = 1.f;
    float walkSpeed = 0.f;           // m/s along yaw while animated, wandering over the terrain (0 = stand)
    float wanderPhase = 0.f;
    Mode mode = ANIMATED;
    float restTime = 0.f, getUpAfter = 4.f;  // limp: back to the animation after resting this long (0 = never)
    // ragdoll: one physics part per significant bone
    uint32_t ragdoll = 0;
    std::vector<physics::RagdollPart> parts;  // bind-pose capsules (in each bone's model-space frame, model units)
    std::vector<int> partJoint, jointPart;    // part -> joint, joint -> part (-1: follows its parent)
    std::vector<float> partScale;             // 3 per part: the bone's own scale (the bodies are rigid frames)
    float mass = 70.f;
    bool teleport = true;                     // place the bodies on the pose instead of driving them there
    // this frame
    skel::Pose pose;
    std::vector<Xf> globals;         // model space
    std::vector<float> skinMats;     // 12 per joint
    float bmin[3] = {}, bmax[3] = {};  // world bounds (culling)
};

using HeightFn = std::function<float(float x, float z)>;
void init(HeightFn height);
// height > 0: scale the model to this standing height (m); 0: model units (centimetre rigs are detected)
Character* spawn(const ModelData& d, const float pos[3], float yaw, float height = 0.f);
void remove(Character* c);
std::vector<Character*>& all();
int findClip(const ModelData& d, const char* name);  // case-insensitive substring, -1 = none
void play(Character& c, int clip, float fade = 0.3f);
void ragdoll(Character& c, const float impulse[3] = nullptr, const float at[3] = nullptr, int part = 0);
void getUp(Character& c);  // back to the animation where the body lies
Xf worldXf(const Character& c);
// before physics::step: animate, drive the kinematic ragdolls, turn rock hits into ragdolls
void preStep(float dt);
// after physics::step: limp poses from the ragdolls, skinning matrices and bounds
void postStep(float dt);
}  // namespace character

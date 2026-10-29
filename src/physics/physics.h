// Thin wrapper over Jolt so the rest of the code doesn't include Jolt headers.
// Shapes are refcounted Jolt shapes passed around as void*; bodies are raw Jolt BodyIDs.
#pragma once
#include <cstddef>
#include <cstdint>
#include <vector>

#include "world/world.h"

namespace physics {
void init();  // uses the shared jobs:: scheduler, call jobs::init first
void shutdown();
// Thread-safe: shapes are built on the loader threads.
void* meshShape(const float* verts, uint32_t strideFloats, uint32_t nverts, const uint32_t* idx, uint32_t ntris);
void* heightShape(const float* heights, uint32_t n, float x0, float z0, float spacing);  // heights[iz * n + ix]
void releaseShape(void* shape);
// Main thread: static bodies, added as one broadphase batch.
void addStatic(void* const* shapes, const Xf* xfs, size_t n, uint32_t* outIds);
void removeBodies(const uint32_t* ids, size_t n);
void step(float dt);
// Ground surface at a contact on the terrain (heightfield): friction and restitution for a dynamic body touching it
// there (snow, wet soil, paving, ...). Called from the physics worker threads: must be thread safe.
using SurfaceFn = void (*)(const float p[3], float& friction, float& restitution);
void setSurfaceFn(SurfaceFn fn);
uint32_t bodyCount();

// Dynamic bodies (thrown rocks). convexShape builds a hull from a point cloud; addDynamic scales it uniformly.
void* convexShape(const float* verts, uint32_t strideFloats, uint32_t nverts);
// Compound of convex hulls (counts[i] points each, pos3 back to back): a broken-off limb with its branches.
void* compoundShape(const float* pts, const uint32_t* counts, uint32_t nHulls);
float shapeVolume(void* shape);  // m^3
uint32_t addDynamic(void* shape, float scale, const float pos[3], const float vel[3], const float angVel[3], float density,
                    float yaw = 0.f);
// World transform (rotation * scale + translation, scale folded in), linear / angular velocity and mass.
// False if the body is gone.
bool bodyState(uint32_t id, float scale, Xf& xf, float vel[3], float angVel[3], float& mass);
// Centre of mass in world space (forces / torques act about it). False if the body is gone.
bool bodyCom(uint32_t id, float com[3]);
// Force / torque for the next step (N, N m), e.g. buoyancy and drag.
void addForce(uint32_t id, const float force[3], const float torque[3]);
// Same with a full pose: xf rotation must be orthonormal (no scale), translation = body origin.
uint32_t addDynamicPose(void* shape, const Xf& xf, const float vel[3], const float angVel[3], float density);
// Same with the mass (kg), its centre (body space) and inertia about that (kg m^2, body axes, row major) given;
// shifted to the shape's centre of mass, which the body rotates about.
uint32_t addDynamicPoseInertia(void* shape, const Xf& xf, const float vel[3], const float angVel[3], float mass, const float com[3],
                               const float inertia[9]);
void addImpulse(uint32_t id, const float impulse[3], const float at[3]);  // N s, applied at a world point
void setVelocity(uint32_t id, const float vel[3]);
void setAngularVelocity(uint32_t id, const float w[3]);
// Velocity decay per second (Jolt: dv/dt = -c v), e.g. air drag on a leafy branch.
void setDamping(uint32_t id, float linear, float angular);

// Ragdolls (character.h): one capsule body per physics bone, a swing-twist joint to its parent bone. Kinematic while
// the character is animated (the bodies follow the pose and push what they touch), dynamic when it goes limp.
struct RagdollPart {
    int parent = -1;              // part index, parents first; -1 = the root
    Xf xf;                        // the bone's world frame (orthonormal rotation + origin): the body transform
    float a[3], b[3], radius;     // capsule segment in that frame (m)
    float swing = 0.7f, twist = 0.4f;  // joint limits to the parent (half angles, radians)
};
uint32_t ragdollCreate(const RagdollPart* parts, size_t n, float totalMass);  // 0 = out of bodies
void ragdollDestroy(uint32_t r);
void ragdollDrive(uint32_t r, const Xf* xfs, float dt);  // kinematic: reach these frames in dt (contacts see the velocity)
void ragdollTeleport(uint32_t r, const Xf* xfs);
void ragdollSetLimp(uint32_t r, bool limp);              // dynamic, keeping the bodies' current velocity / kinematic
void ragdollGetPose(uint32_t r, Xf* xfs);                // world frames of the bodies
void ragdollImpulse(uint32_t r, int part, const float impulse[3], const float at[3]);
void ragdollAddVelocity(uint32_t r, const float vel[3]);  // every body
bool ragdollResting(uint32_t r);                         // all bodies asleep
// Fast dynamic bodies (rocks) that touched an animated (kinematic) ragdoll since the last call. The kinematic body
// doesn't give way, so the caller turns the hit into a limp ragdoll + impulse.
struct RagdollHit { uint32_t ragdoll; int part; float at[3], momentum[3], mass; };  // momentum: the other body's, relative
void takeRagdollHits(std::vector<RagdollHit>& out);

// First-person character (capsule, 1.8 m), walks stairs and slopes.
void spawnPlayer(const float feet[3]);
// wish = desired horizontal velocity (m/s). Returns feet position.
void movePlayer(float dt, const float wish[2], bool jump, float maxSpeed, float outFeet[3], bool& grounded);
}  // namespace physics

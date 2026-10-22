// Thin wrapper over Jolt so the rest of the code doesn't include Jolt headers.
// Shapes are refcounted Jolt shapes passed around as void*; bodies are raw Jolt BodyIDs.
#pragma once
#include <cstddef>
#include <cstdint>

#include "world.h"

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
uint32_t addDynamic(void* shape, float scale, const float pos[3], const float vel[3], const float angVel[3], float density,
                    float yaw = 0.f);
// World transform (rotation * scale + translation, scale folded in), linear / angular velocity and mass.
// False if the body is gone.
bool bodyState(uint32_t id, float scale, Xf& xf, float vel[3], float angVel[3], float& mass);
// Force / torque for the next step (N, N m), e.g. buoyancy and drag.
void addForce(uint32_t id, const float force[3], const float torque[3]);
// Same with a full pose: xf rotation must be orthonormal (no scale), translation = body origin.
uint32_t addDynamicPose(void* shape, const Xf& xf, const float vel[3], const float angVel[3], float density);
void addImpulse(uint32_t id, const float impulse[3], const float at[3]);  // N s, applied at a world point
void setVelocity(uint32_t id, const float vel[3]);

// First-person character (capsule, 1.8 m), walks stairs and slopes.
void spawnPlayer(const float feet[3]);
// wish = desired horizontal velocity (m/s). Returns feet position.
void movePlayer(float dt, const float wish[2], bool jump, float outFeet[3], bool& grounded);
}  // namespace physics

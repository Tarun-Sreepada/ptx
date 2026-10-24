// Jolt internals shared by the physics files: the physics system, object layers, ragdoll bookkeeping, the player.
#pragma once
#include <Jolt/Jolt.h>

#include <Jolt/Core/Factory.h>
#include <Jolt/Core/FixedSizeFreeList.h>
#include <Jolt/Core/JobSystemWithBarrier.h>
#include <Jolt/Core/TempAllocator.h>
#include <TaskScheduler.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Character/CharacterVirtual.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayerInterfaceTable.h>
#include <Jolt/Physics/Collision/BroadPhase/ObjectVsBroadPhaseLayerFilterTable.h>
#include <Jolt/Physics/Collision/ObjectLayerPairFilterTable.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/HeightFieldShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <Jolt/Physics/Collision/Shape/ScaledShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <Jolt/Physics/Collision/ContactListener.h>
#include <Jolt/Physics/Constraints/SwingTwistConstraint.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Ragdoll/Ragdoll.h>
#include <Jolt/Skeleton/Skeleton.h>
#include <Jolt/RegisterTypes.h>

#include <cstdio>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "physics/physics.h"
#include "core/constants.h"
#include "core/jobs.h"


#include "physics/physics.h"

namespace physics {
using namespace JPH;

namespace Layers { inline constexpr ObjectLayer STATIC = 0, MOVING = 1, COUNT = 2; }
namespace BPLayers { inline constexpr BroadPhaseLayer STATIC(0), MOVING(1); inline constexpr uint COUNT = 2; }

extern std::unique_ptr<PhysicsSystem> sys;
extern std::unique_ptr<TempAllocatorImpl> temp;  // scratch memory for steps and character moves
extern bool addedSinceOptimize;  // bodies were added since the broadphase was last optimised
// ragdolls: body user data = RAGDOLL_TAG | handle
inline constexpr uint64 RAGDOLL_TAG = 1ull << 63;
extern std::unordered_map<uint32_t, Ref<Ragdoll>> ragdolls;
extern uint32_t nextRagdoll;
extern std::mutex hitMu;
extern std::vector<RagdollHit> hits;
extern std::unordered_map<uint32_t, std::pair<uint32_t, int>> ragdollOfBody;  // body id -> (ragdoll, part), main thread writes only between steps
extern Ref<CharacterVirtual> player;
}  // namespace physics

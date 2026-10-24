#pragma once
#include "core/limits.h"
#include <vector>
#include <string>
#include <memory>
#include <cstdint>
#include <vulkan/vulkan.h>
#include "characters/character.h"
#include "world/world.h"
#include "gpu/vk.h"


extern uint32_t numCharInst;

struct CharAsset { std::shared_ptr<Model> model; std::vector<Buffer> skinW; };  // skinW per mesh (empty: unskinned)

struct GpuCharMesh {
    uint32_t mesh = 0, nv = 0, ntri = 0, slot = UINT32_MAX, srcSlot = UINT32_MAX, prevSlot = UINT32_MAX;
    Buffer verts, prevVerts;  // prevVerts: last frame's skinned vertices (copied before skinning; motion vectors)
    Accel blas;
    VkAccelerationStructureGeometryKHR geom{};
};

struct GpuChar { CharAsset* asset = nullptr; Buffer mats; std::vector<GpuCharMesh> meshes; uint32_t frames = 0; Xf prevW{}; bool hasPrev = false; };

struct CharSpawn { std::string asset, clip; float pos[3], yaw, height, walk; bool wild = false; float forward = 1.f; };

extern std::vector<CharSpawn> charSpawns;

extern std::string charWorldDir;

void spawnCharacters();

void releaseCharacter(character::Character* c);

void writeCharacters();

// Characters walking leave footprints and trodden grass, or splash with every stride through water.
void characterSteps(float dt);

void recordSkinning(VkCommandBuffer cb);

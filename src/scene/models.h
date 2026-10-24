#pragma once
#include <vector>
#include <cstdint>
#include <cstddef>
#include <vulkan/vulkan.h>
#include "world/world.h"
#include "gpu/vk.h"

// model.glsl Material (std430, 128 bytes)
struct GpuMat {
    float color[4];
    float emissive[3]; uint32_t tex;
    float rough, metal, normalScale; uint32_t nrmTex;
    uint32_t ormTex, emisTex, occTex; float occStrength;
    float heightScale, clearcoat, ccRough, sheen;
    float sheenColor[3]; float aniso;
    float anisoRot, transmission, ior, specular;
    float transl, sss, pad[2];
};

static_assert(sizeof(GpuMat) == 128);

extern Buffer meshInfoBuf, materialBuf, modelXfBuf, zoneBuf;

struct SlotPool {  // free-list allocator for bindless slots
    std::vector<uint32_t> freeList;
    uint32_t next = 0, cap = 0;
    uint32_t alloc() {
        if (!freeList.empty()) { uint32_t s = freeList.back(); freeList.pop_back(); return s; }
        return next < cap ? next++ : UINT32_MAX;
    }
    void release(uint32_t s) { freeList.push_back(s); }
    uint32_t used() const { return next - (uint32_t)freeList.size(); }
};

extern SlotPool meshSlots, matSlots, texSlots;

struct EmisTri { float v[3][3]; float Le[3]; };  // object-space emissive triangle and its radiance (light list)

struct GpuMesh {
    uint32_t slot = UINT32_MAX; uint32_t firstIndex = 0, indexCount = 0; VkAccelerationStructureKHR blas = VK_NULL_HANDLE; VkDeviceAddress blasAddr = 0;
    std::vector<EmisTri> emis;
};

struct GpuModel {
    Buffer geom, blasMem;               // all meshes' vertices | indices | triangle materials; all BLASes
    std::vector<GpuMesh> meshes;
    std::vector<uint32_t> mats, texs;   // slots
    std::vector<Image> images;
    std::vector<uint32_t> audioMeshes;  // acoustic (simplified) copies, see audio.h
    size_t bytes = 0;
};

extern bool uploadsRecorded;

extern size_t gpuModelBytes;

extern float uploadBudgetMB;

void beginUploads();

extern size_t frameUploadBytes;

bool uploadModel(Model& m);

void releaseModel(Model& m);

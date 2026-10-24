#include "scene/models.h"
#include <vector>
#include <array>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include <cstddef>
#include <vulkan/vulkan.h>
#include "core/prof.h"
#include "audio/audio.h"
#include "world/world.h"
#include "core/limits.h"
#include "gpu/accel.h"
#include "gpu/context.h"
#include "gpu/memory.h"
#include "gpu/textures.h"
#include "gpu/vk.h"
#include "scene/audio_sync.h"

// ---------------------------------------------------------------- streamed models (GPU side)
Buffer meshInfoBuf, materialBuf, modelXfBuf, zoneBuf;

SlotPool meshSlots{{}, 0, MAX_MESHES}, matSlots{{}, 0, MAX_MATERIALS}, texSlots{{}, 1, MAX_TEXTURES};  // tex slot 0 = white

bool uploadsRecorded = false;

size_t gpuModelBytes = 0;

float uploadBudgetMB = 96.f;

void beginUploads() {
    if (uploadsRecorded) return;
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkResetCommandBuffer(uploadCmd, 0);
    vkBeginCommandBuffer(uploadCmd, &bi);
    uploadsRecorded = true;
}

template <class F> static GpuMat gpuMaterial(const MaterialData& mt, F slot) {
    GpuMat g{};
    memcpy(g.color, mt.color, sizeof g.color);
    memcpy(g.emissive, mt.emissive, sizeof g.emissive);
    g.tex = slot(mt.tex);
    g.rough = mt.rough; g.metal = mt.metal; g.normalScale = mt.normalScale; g.nrmTex = slot(mt.normalTex);
    g.ormTex = slot(mt.ormTex); g.emisTex = slot(mt.emisTex); g.occTex = slot(mt.occTex); g.occStrength = mt.occStrength;
    g.heightScale = mt.normalTex >= 0 ? mt.heightScale : 0.f;
    g.clearcoat = mt.clearcoat; g.ccRough = mt.clearcoatRough; g.sheen = mt.sheen;
    memcpy(g.sheenColor, mt.sheenColor, sizeof g.sheenColor);
    g.aniso = mt.anisotropy; g.anisoRot = mt.anisoRot; g.transmission = mt.transmission; g.ior = mt.ior; g.specular = mt.specular;
    g.transl = mt.translucency;
    g.sss = mt.sss;
    return g;
}

static bool isGlass(const MaterialData& m) { return m.transmission > 0.5f; }

size_t frameUploadBytes = 0;

// Model CPU data -> one geometry buffer, one BLAS buffer (a BLAS per mesh), mip-mapped textures, bindless slots.
// Recorded into uploadCmd; returns false once this frame's upload budget is spent.
bool uploadModel(Model& m) {
    PROF_FUNC();
    const ModelData& d = m.data;
    if (meshSlots.cap - meshSlots.used() < d.meshes.size() || matSlots.cap - matSlots.used() < d.materials.size() ||
        texSlots.cap - texSlots.used() < d.textures.size()) {
        static bool warned = false;
        if (!warned) fprintf(stderr, "stream: out of mesh/material/texture slots, %s not uploaded\n", m.path.c_str());
        warned = true;
        return false;
    }
    beginUploads();
    VkCommandBuffer cb = uploadCmd;
    auto* g = new GpuModel;

    // slots: textures, then materials (pointing at texture slots), then per-triangle material slots
    for (auto& t : d.textures) {
        g->texs.push_back(texSlots.alloc());
        if (!t.bc7.empty()) g->images.push_back(mkTexture(t.w, t.h, t.bcMips, t.linear ? VK_FORMAT_BC7_UNORM_BLOCK : VK_FORMAT_BC7_SRGB_BLOCK));
        else g->images.push_back(mkTexture(t.w, t.h, mipCount(t.w, t.h), t.linear ? VK_FORMAT_R8G8B8A8_UNORM : VK_FORMAT_R8G8B8A8_SRGB));
    }
    for (auto& mt : d.materials) {
        uint32_t slot = matSlots.alloc();
        GpuMat gm = gpuMaterial(mt, [&](int t) { return t >= 0 && t < (int)g->texs.size() ? g->texs[t] : 0xFFFFFFFFu; });
        memcpy((char*)materialBuf.map + slot * sizeof(GpuMat), &gm, sizeof gm);
        g->mats.push_back(slot);
    }

    // layout: per mesh [verts | indices | triangle material slots], 16-byte aligned
    auto al = [](VkDeviceSize v, VkDeviceSize a) { return (v + a - 1) & ~(a - 1); };
    std::vector<VkDeviceSize> vOff(d.meshes.size()), iOff(d.meshes.size()), tOff(d.meshes.size());
    VkDeviceSize geomSize = 0;
    for (size_t i = 0; i < d.meshes.size(); i++) {
        auto& md = d.meshes[i];
        vOff[i] = geomSize = al(geomSize, 16); geomSize += md.verts.size() * 4;
        iOff[i] = geomSize = al(geomSize, 16); geomSize += md.idx.size() * 4;
        tOff[i] = geomSize = al(geomSize, 16); geomSize += md.triMat.size() * 4;
    }
    geomSize = std::max<VkDeviceSize>(al(geomSize, 16), 16);
    VkDeviceSize texSize = 0;
    std::vector<VkDeviceSize> texOff;
    auto texBytes = [](const TextureData& t) -> const std::vector<uint8_t>& { return t.bc7.empty() ? t.rgba : t.bc7; };
    for (auto& t : d.textures) { texOff.push_back(geomSize + texSize); texSize += al(texBytes(t).size(), 16); }

    Buffer st = mkBuf(geomSize + texSize, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    char* sp = (char*)st.map;
    for (size_t i = 0; i < d.meshes.size(); i++) {
        auto& md = d.meshes[i];
        memcpy(sp + vOff[i], md.verts.data(), md.verts.size() * 4);
        memcpy(sp + iOff[i], md.idx.data(), md.idx.size() * 4);
        uint32_t* tm = (uint32_t*)(sp + tOff[i]);
        for (size_t t = 0; t < md.triMat.size(); t++) tm[t] = g->mats[md.triMat[t]];
    }
    for (size_t i = 0; i < d.textures.size(); i++) memcpy(sp + texOff[i], texBytes(d.textures[i]).data(), texBytes(d.textures[i]).size());

    g->geom = mkBuf(geomSize, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR |
                                  VK_BUFFER_USAGE_INDEX_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                    VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VkBufferCopy cp{0, 0, geomSize};
    vkCmdCopyBuffer(cb, st.b, g->geom.b, 1, &cp);
    for (size_t i = 0; i < d.textures.size(); i++) {
        auto& t = d.textures[i];
        if (!t.bc7.empty()) recordCompressedUpload(cb, g->images[i], t, st.b, texOff[i]);
        else recordTextureUpload(cb, g->images[i], t.w, t.h, mipCount(t.w, t.h), st.b, texOff[i]);
        writeTexDescriptor(g->texs[i], g->images[i].view);
    }
    memBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_ACCESS_TRANSFER_WRITE_BIT,
               VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_MEMORY_READ_BIT);

    // one BLAS per mesh, all in one buffer and one build call
    size_t n = d.meshes.size();
    g->meshes.resize(n);
    std::vector<VkAccelerationStructureGeometryKHR> geoms(n);
    std::vector<VkAccelerationStructureBuildGeometryInfoKHR> infos;
    std::vector<VkAccelerationStructureBuildRangeInfoKHR> ranges;
    std::vector<size_t> which;
    std::vector<VkDeviceSize> asOff, asSize, scOff;
    VkDeviceSize asTotal = 0, scTotal = 0, scAlign = asProps.minAccelerationStructureScratchOffsetAlignment;
    for (size_t i = 0; i < n; i++) {
        auto& md = d.meshes[i];
        uint32_t tris = (uint32_t)md.idx.size() / 3;
        if (!tris) continue;
        auto& gg = geoms[i];
        gg = {VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR};
        gg.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
        // meshes with glass are non-opaque: shadow rays (common.glsl visible()) pass through the glass triangles
        bool glass = false;
        for (uint32_t tm : md.triMat) glass |= tm < d.materials.size() && isGlass(d.materials[tm]);
        gg.flags = glass ? 0 : VK_GEOMETRY_OPAQUE_BIT_KHR;
        auto& t = gg.geometry.triangles;
        t.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
        t.vertexFormat = VK_FORMAT_R32G32B32_SFLOAT;
        t.vertexData.deviceAddress = g->geom.addr + vOff[i];
        t.vertexStride = 32;
        t.maxVertex = (uint32_t)(md.verts.size() / 8) - 1;
        t.indexType = VK_INDEX_TYPE_UINT32;
        t.indexData.deviceAddress = g->geom.addr + iOff[i];
        auto bi = buildInfo(VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR, VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR, &gg);
        VkAccelerationStructureBuildSizesInfoKHR sz{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR};
        pGetASSizes(dev, VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &bi, &tris, &sz);
        asOff.push_back(asTotal = al(asTotal, 256));
        asSize.push_back(sz.accelerationStructureSize);
        asTotal += sz.accelerationStructureSize;
        scOff.push_back(scTotal = al(scTotal, scAlign));
        scTotal += sz.buildScratchSize;
        infos.push_back(bi);
        ranges.push_back({tris, 0, 0, 0});
        which.push_back(i);
    }
    Buffer scratch;
    if (!infos.empty()) {
        g->blasMem = mkBuf(asTotal, VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        scratch = mkBuf(scTotal + scAlign, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
        VkDeviceAddress scBase = al(scratch.addr, scAlign);
        std::vector<const VkAccelerationStructureBuildRangeInfoKHR*> pr;
        for (size_t k = 0; k < infos.size(); k++) {
            GpuMesh& gm = g->meshes[which[k]];
            VkAccelerationStructureCreateInfoKHR ci{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR};
            ci.buffer = g->blasMem.b;
            ci.offset = asOff[k];
            ci.size = asSize[k];
            ci.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
            VK_CHECK(pCreateAS(dev, &ci, nullptr, &gm.blas));
            VkAccelerationStructureDeviceAddressInfoKHR ai{VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR};
            ai.accelerationStructure = gm.blas;
            gm.blasAddr = pGetASAddr(dev, &ai);
            infos[k].pGeometries = &geoms[which[k]];
            infos[k].dstAccelerationStructure = gm.blas;
            infos[k].scratchData.deviceAddress = scBase + scOff[k];
            pr.push_back(&ranges[k]);
        }
        pCmdBuildAS(cb, (uint32_t)infos.size(), infos.data(), pr.data());
    }
    memBarrier(cb, VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR, VK_ACCESS_ACCELERATION_STRUCTURE_WRITE_BIT_KHR,
               VK_PIPELINE_STAGE_ACCELERATION_STRUCTURE_BUILD_BIT_KHR | VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
               VK_ACCESS_ACCELERATION_STRUCTURE_READ_BIT_KHR);

    // emissive triangles (radiance = factor x the emissive texture's mean colour) for the light list
    std::vector<std::array<float, 3>> emisAvg(d.materials.size(), {1, 1, 1});
    for (size_t k = 0; k < d.materials.size(); k++) {
        int t = d.materials[k].emisTex;
        if (t < 0 || t >= (int)d.textures.size() || d.textures[t].rgba.empty()) continue;
        double sum[3] = {0, 0, 0};
        const auto& px = d.textures[t].rgba;
        size_t np = px.size() / 4, step = std::max<size_t>(1, np / 4096);
        size_t cnt = 0;
        for (size_t q = 0; q < np; q += step, cnt++)
            for (int c = 0; c < 3; c++) { float v = px[q * 4 + c] / 255.f; sum[c] += v <= 0.04045f ? v / 12.92f : powf((v + 0.055f) / 1.055f, 2.4f); }
        for (int c = 0; c < 3; c++) emisAvg[k][c] = (float)(sum[c] / std::max<size_t>(cnt, 1));
    }
    for (size_t i = 0; i < n; i++) {
        const MeshData& md = d.meshes[i];
        for (size_t t = 0; t < md.triMat.size(); t++) {
            uint32_t mi = md.triMat[t];
            if (mi >= d.materials.size()) continue;
            const MaterialData& mt = d.materials[mi];
            float Le[3] = {mt.emissive[0] * emisAvg[mi][0], mt.emissive[1] * emisAvg[mi][1], mt.emissive[2] * emisAvg[mi][2]};
            if (Le[0] + Le[1] + Le[2] <= 0.f) continue;
            EmisTri e;
            for (int v = 0; v < 3; v++) memcpy(e.v[v], &md.verts[md.idx[t * 3 + v] * 8], 12);
            memcpy(e.Le, Le, 12);
            g->meshes[i].emis.push_back(e);
        }
    }
    for (size_t i = 0; i < n; i++) {
        GpuMesh& gm = g->meshes[i];
        if (!gm.blas) continue;
        gm.slot = meshSlots.alloc();
        gm.indexCount = (uint32_t)d.meshes[i].idx.size();
        gm.firstIndex = (uint32_t)(iOff[i] / 4);  // raster binds the whole geometry buffer as the index buffer
        VkDeviceAddress info[4] = {g->geom.addr + vOff[i], g->geom.addr + iOff[i], g->geom.addr + tOff[i], 0};
        memcpy((char*)meshInfoBuf.map + gm.slot * 32, info, 32);
    }
    retireNext.push_back([st, scratch]() mutable { freeBuf(st); freeBuf(scratch); });
    for (auto& md : d.meshes)
        g->audioMeshes.push_back(audio::meshCreate(md.verts.data(), 8, (uint32_t)(md.verts.size() / 8), md.idx.data(),
                                                   (uint32_t)(md.idx.size() / 3), audio::MAT_CONCRETE));
    m.gpu = g;
    g->bytes = geomSize + asTotal + texSize * 4 / 3;  // textures + mips
    gpuModelBytes += g->bytes;
    frameUploadBytes += geomSize + texSize;
    return frameUploadBytes < (size_t)(uploadBudgetMB * 1048576.f);
}

void releaseModel(Model& m) {
    auto* g = (GpuModel*)m.gpu;
    for (auto& gm : g->meshes) {
        if (gm.blas) pDestroyAS(dev, gm.blas, nullptr);
        if (gm.slot != UINT32_MAX) meshSlots.release(gm.slot);
    }
    for (uint32_t s : g->mats) matSlots.release(s);
    for (uint32_t s : g->texs) { writeTexDescriptor(s, defaultTex.view); texSlots.release(s); }
    for (auto& im : g->images) freeImage(im);
    // instances of these meshes are removed in onStreamChanged (same frame), release the meshes after that
    audioMeshRetire.insert(audioMeshRetire.end(), g->audioMeshes.begin(), g->audioMeshes.end());
    gpuModelBytes -= g->bytes;
    freeBuf(g->geom);
    freeBuf(g->blasMem);
    delete g;
    m.gpu = nullptr;
}

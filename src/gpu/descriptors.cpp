#include "gpu/descriptors.h"
#include <vector>
#include <algorithm>
#include <cstdint>
#include <vulkan/vulkan.h>
#include "gpu/context.h"
#include "gpu/pipelines.h"
#include "gpu/swapchain.h"
#include "gpu/textures.h"
#include "render/bake.h"
#include "render/crater_bins.h"
#include "render/ground_vt.h"
#include "render/lights.h"
#include "render/targets.h"
#include "scene/models.h"
#include "scene/scene_gpu.h"
#include "scene/vegetation.h"
#include "sim/world_sim.h"

void writeDescriptors() {
    std::vector<VkWriteDescriptorSet> w;
    VkWriteDescriptorSetAccelerationStructureKHR asw{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR};
    asw.accelerationStructureCount = 1;
    asw.pAccelerationStructures = &tlas.as;
    VkWriteDescriptorSet ws{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    ws.dstSet = descSet;
    ws.descriptorCount = 1;

    VkWriteDescriptorSet a = ws;
    a.pNext = &asw;
    a.dstBinding = 0;
    a.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
    w.push_back(a);

    VkDescriptorBufferInfo bufs[9] = {{uboBuf.b, 0, VK_WHOLE_SIZE},   {terrainV.b, 0, VK_WHOLE_SIZE}, {terrainI.b, 0, VK_WHOLE_SIZE},
                                      {grassV.b, 0, VK_WHOLE_SIZE},   {grassI.b, 0, VK_WHOLE_SIZE},   {bladeBuf.b, 0, VK_WHOLE_SIZE},
                                      {lightBuf.b, 0, VK_WHOLE_SIZE}, {resCur.b, 0, VK_WHOLE_SIZE},   {resPrev.b, 0, VK_WHOLE_SIZE}};
    for (uint32_t i = 0; i < 9; i++) {
        VkWriteDescriptorSet x = ws;
        x.dstBinding = 1 + i;
        x.descriptorType = i == 0 ? VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER : VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        x.pBufferInfo = &bufs[i];
        w.push_back(x);
    }
    VkDescriptorImageInfo ii[I_COUNT];
    for (uint32_t i = 0; i < I_COUNT; i++) {
        ii[i] = {VK_NULL_HANDLE, imgs[i].view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet x = ws;
        x.dstBinding = 10 + i;
        x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        x.pImageInfo = &ii[i];
        w.push_back(x);
    }
    VkDescriptorBufferInfo gib[7] = {{giCur.b, 0, VK_WHOLE_SIZE}, {giPrev.b, 0, VK_WHOLE_SIZE}, {grassCounter.b, 0, VK_WHOLE_SIZE},
                                     {}, {rockV.b, 0, VK_WHOLE_SIZE}, {rockI.b, 0, VK_WHOLE_SIZE}, {rockXf.b, 0, VK_WHOLE_SIZE}};
    VkDescriptorImageInfo skyInfo{VK_NULL_HANDLE, skyImg.view, VK_IMAGE_LAYOUT_GENERAL};
    for (uint32_t i = 0; i < 7; i++) {
        if (i == 3) {
            VkWriteDescriptorSet x = ws;
            x.dstBinding = 28;
            x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
            x.pImageInfo = &skyInfo;
            w.push_back(x);
            continue;
        }
        VkWriteDescriptorSet x = ws;
        x.dstBinding = 25 + i;
        x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        x.pBufferInfo = &gib[i];
        w.push_back(x);
    }
    VkDescriptorBufferInfo mb[4] = {{meshInfoBuf.b, 0, VK_WHOLE_SIZE}, {materialBuf.b, 0, VK_WHOLE_SIZE}, {modelXfBuf.b, 0, VK_WHOLE_SIZE},
                                    {zoneBuf.b, 0, VK_WHOLE_SIZE}};
    for (uint32_t i = 0; i < 4; i++) {
        VkWriteDescriptorSet x = ws;
        x.dstBinding = 33 + i;
        x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        x.pBufferInfo = &mb[i];
        w.push_back(x);
    }
    VkDescriptorImageInfo ri[RR_COUNT];
    for (uint32_t i = 0; i < RR_COUNT; i++) {
        ri[i] = {VK_NULL_HANDLE, rrImgs[i].view, VK_IMAGE_LAYOUT_GENERAL};
        VkWriteDescriptorSet x = ws;
        x.dstBinding = 37 + i;
        x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        x.pImageInfo = &ri[i];
        w.push_back(x);
    }
    VkDescriptorImageInfo windInfo[2] = {{VK_NULL_HANDLE, windImg[0].view, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, windImg[1].view, VK_IMAGE_LAYOUT_GENERAL}};
    for (uint32_t i = 0; i < 2; i++) {
        VkWriteDescriptorSet x = ws;
        x.dstBinding = 52 + i;
        x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        x.pImageInfo = &windInfo[i];
        w.push_back(x);
    }
    VkDescriptorBufferInfo impInfo[4] = {{impBuf.b, 0, VK_WHOLE_SIZE}, {volBuf.b, 0, VK_WHOLE_SIZE}, {probeBuf.b, 0, VK_WHOLE_SIZE},
                                         {probeTmp.b, 0, VK_WHOLE_SIZE}};
    VkWriteDescriptorSet x = ws;
    for (uint32_t i = 0; i < 4; i++) {
        x = ws;
        x.dstBinding = i == 0 ? 44 : 48 + i;
        x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        x.pBufferInfo = &impInfo[i];
        w.push_back(x);
    }
    VkDescriptorImageInfo impImg[4] = {{texSampler, impAlbedoImg.view, VK_IMAGE_LAYOUT_GENERAL}, {texSampler, impNormalImg.view, VK_IMAGE_LAYOUT_GENERAL},
                                       {VK_NULL_HANDLE, impAlbedoImg.view, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, impNormalImg.view, VK_IMAGE_LAYOUT_GENERAL}};
    for (uint32_t i = 0; i < 4; i++) {
        x = ws;
        x.dstBinding = 45 + i;
        x.descriptorType = i < 2 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        x.pImageInfo = &impImg[i];
        w.push_back(x);
    }
    VkDescriptorImageInfo mapInfo[7] = {{postSampler, mapHImg.view, VK_IMAGE_LAYOUT_GENERAL}, {postSampler, mapPaintImg.view, VK_IMAGE_LAYOUT_GENERAL},
                                        {postSampler, mapWaterImg.view, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, simImg.view, VK_IMAGE_LAYOUT_GENERAL},
                                        {VK_NULL_HANDLE, fluxWImg.view, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, fluxSImg.view, VK_IMAGE_LAYOUT_GENERAL},
                                        {VK_NULL_HANDLE, simGroundImg.view, VK_IMAGE_LAYOUT_GENERAL}};
    for (uint32_t i = 0; i < 7; i++) {
        x = ws;
        x.dstBinding = 90 + i;
        x.descriptorType = i < 3 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        x.pImageInfo = &mapInfo[i];
        w.push_back(x);
    }
    VkDescriptorBufferInfo cbInfo{craterBinBuf.b, 0, VK_WHOLE_SIZE};
    x = ws;
    x.dstBinding = 97;
    x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    x.pBufferInfo = &cbInfo;
    w.push_back(x);
    VkDescriptorBufferInfo rcInfo{radCacheBuf.b, 0, VK_WHOLE_SIZE};
    x.dstBinding = 98;
    x.pBufferInfo = &rcInfo;
    w.push_back(x);
    VkDescriptorImageInfo vtInfo[6] = {{texSampler, vtImgA.view, VK_IMAGE_LAYOUT_GENERAL}, {texSampler, vtImgB.view, VK_IMAGE_LAYOUT_GENERAL},
                                       {texSampler, vtImgH.view, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, vtImgA.view, VK_IMAGE_LAYOUT_GENERAL},
                                       {VK_NULL_HANDLE, vtImgB.view, VK_IMAGE_LAYOUT_GENERAL}, {VK_NULL_HANDLE, vtImgH.view, VK_IMAGE_LAYOUT_GENERAL}};
    for (uint32_t i = 0; i < 6; i++) {
        x = ws;
        x.dstBinding = i < 3 ? 99 + i : 100 + i;
        x.descriptorType = i < 3 ? VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER : VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        x.pImageInfo = &vtInfo[i];
        w.push_back(x);
    }
    VkDescriptorBufferInfo vtTiles{vtTileBuf.b, 0, VK_WHOLE_SIZE};
    x = ws;
    x.dstBinding = 102;
    x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    x.pBufferInfo = &vtTiles;
    w.push_back(x);
    VkDescriptorBufferInfo lb[3] = {{triLightBuf.b, 0, VK_WHOLE_SIZE}, {lightTreeBuf.b, 0, VK_WHOLE_SIZE}, {blueNoiseBuf.b, 0, VK_WHOLE_SIZE}};
    for (uint32_t i = 0; i < 3; i++) {
        x = ws;
        x.dstBinding = 80 + i;
        x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        x.pBufferInfo = &lb[i];
        w.push_back(x);
    }
    VkDescriptorBufferInfo eb{exposureBuf.b, 0, VK_WHOLE_SIZE}, ab{lightAliasBuf.b, 0, VK_WHOLE_SIZE};
    x = ws;
    x.dstBinding = 83;
    x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
    x.pBufferInfo = &eb;
    w.push_back(x);
    x.dstBinding = 87;
    x.pBufferInfo = &ab;
    w.push_back(x);
    VkDescriptorImageInfo pi[2] = {{postSampler, ximgs[X_HDR].view, VK_IMAGE_LAYOUT_GENERAL}, {postSampler, bloomImg.view, VK_IMAGE_LAYOUT_GENERAL}};
    for (uint32_t i = 0; i < 2; i++) {
        x = ws;
        x.dstBinding = 84 + i;
        x.descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER;
        x.pImageInfo = &pi[i];
        w.push_back(x);
    }
    VkDescriptorImageInfo bm[8];
    for (uint32_t i = 0; i < 8; i++) bm[i] = {VK_NULL_HANDLE, bloomViews[std::min(i, BLOOM_MIPS - 1)], VK_IMAGE_LAYOUT_GENERAL};
    x = ws;
    x.dstBinding = 86;
    x.descriptorCount = 8;
    x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    x.pImageInfo = bm;
    w.push_back(x);
    VkDescriptorImageInfo xi[X_COUNT];
    for (uint32_t i = 0; i < X_COUNT; i++) {
        xi[i] = {VK_NULL_HANDLE, ximgs[i].view, VK_IMAGE_LAYOUT_GENERAL};
        x = ws;
        x.dstBinding = X_BINDING + i;
        x.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
        x.pImageInfo = &xi[i];
        w.push_back(x);
    }
    vkUpdateDescriptorSets(dev, (uint32_t)w.size(), w.data(), 0, nullptr);
    writePresentDescriptors();
}

void writePresentDescriptors() {
    VkDescriptorImageInfo ui{postSampler, uiImg.view, VK_IMAGE_LAYOUT_GENERAL}, pr{VK_NULL_HANDLE, presentImg.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w[2] = {{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
    for (int i = 0; i < 2; i++) { w[i].dstSet = descSet; w[i].descriptorCount = 1; }
    w[0].dstBinding = 88; w[0].descriptorType = VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER; w[0].pImageInfo = &ui;
    w[1].dstBinding = 89; w[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; w[1].pImageInfo = &pr;
    vkUpdateDescriptorSets(dev, 2, w, 0, nullptr);
}

#pragma once
#include <cstdint>
#include <vulkan/vulkan.h>
#include "gpu/dlss.h"
#include "gpu/vk.h"

// Screen images, in binding order 10..24 (must match common.glsl)
enum { I_GPOS, I_GNORM, I_GALBEDO, I_GEMIS, I_PREVPOS, I_PREVNORM, I_DIRECT, I_INDIRECT, I_ACCUM, I_ACCUMMOM,
       I_HISTCOLOR, I_HISTMOM, I_PINGA, I_PINGB, I_OUT, I_COUNT };

extern Image imgs[I_COUNT];

enum { X_GMAT, X_GMAT2, X_GTAN, X_SPEC, X_HDR, X_HDROUT, X_FOG, X_FOGHIST, X_GVEL, X_FOGAGE, X_FOGAGEHIST, X_CLOUD, X_CLOUD2, X_COUNT };

// post-processing (post.glsl)
inline constexpr uint32_t BLOOM_MIPS = 7;

extern VkSampler postSampler;

extern Image bloomImg;

extern VkImageView bloomViews[8];

extern Buffer exposureBuf;

inline constexpr uint32_t X_BINDING = 54;

extern Image ximgs[X_COUNT];

// raster G-buffer targets: gDepth, gNorm, gAlbedo, gEmis, gMat, gMat2, gTan, gVel
inline constexpr uint32_t GBUF_TARGETS = 8;

VkFormat gbufFormat(int i);

Image& gbufImage(int i);

extern Buffer resCur, resPrev, giCur, giPrev;

extern Image depthImg;

inline const VkFormat depthFmt = VK_FORMAT_D32_SFLOAT;

extern uint32_t W, H;

extern uint32_t OW, OH;

extern dlss::Mode upscaler;

extern bool rrActive;

// DLSS-RR inputs (bindings 37..42, render res) and output (43, output res); see dlss_prep.comp
enum { RR_MV, RR_VIEWZ, RR_DIFF, RR_SPEC, RR_NRM, RR_COLOR, RR_OUT, RR_COUNT };

inline const VkFormat rrFormats[RR_COUNT] = {VK_FORMAT_R16G16_SFLOAT, VK_FORMAT_R32_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
                                              VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT, VK_FORMAT_R16G16B16A16_SFLOAT,
                                              VK_FORMAT_R16G16B16A16_SFLOAT};

extern Image rrImgs[RR_COUNT];

inline constexpr uint32_t RC_ENTRIES = 1u << 20, RC_STRIDE = 64;  // radiance cache (radcache.glsl RC_SIZE, RC_STRIDE * 4)

extern Buffer radCacheBuf;

void createScreenResources();

void destroyScreenResources();

extern int pendingRes;

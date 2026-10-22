// DLSS Ray Reconstruction (NGX "DLSSD"): one network that denoises the noisy path-traced radiance and
// upscales it from render to output resolution, replacing the temporal + a-trous denoiser.
// Built only when the SDK is present (external/dlss, PTX_DLSS); otherwise every call is a no-op.
#pragma once
#include <vulkan/vulkan.h>

#include <cstdint>
#include <vector>

namespace dlss {
enum Mode : int { OFF, DLAA, QUALITY, BALANCED, PERFORMANCE, ULTRA_PERFORMANCE, MODE_COUNT };
const char* modeName(Mode m);
float renderScale(Mode m);  // render resolution / output resolution, per axis

// Extensions NGX needs, to append before vkCreateInstance / vkCreateDevice.
void requiredExtensions(std::vector<const char*>& instance, std::vector<const char*>& device);
bool init(VkInstance inst, VkPhysicalDevice phys, VkDevice dev);  // false: RR unavailable (driver, GPU, no SDK)
bool available();
void shutdown();

// (Re)create the feature for a resolution pair; records into cb (submit + wait before evaluate).
bool create(VkCommandBuffer cb, Mode m, uint32_t renderW, uint32_t renderH, uint32_t outW, uint32_t outH);
void release();

struct Img { VkImage image; VkImageView view; VkFormat format; uint32_t w, h; };
struct Inputs {
    Img color, diffuseAlbedo, specularAlbedo, normalRoughness, motion, viewZ, output;
    float jitterX, jitterY;    // render pixels, same convention as the motion vectors
    float worldToView[16], viewToClip[16];  // row-major, row-vector (left multiplication) as NGX expects
    float frameMs;
    bool reset;
};
// Inputs must be in SHADER_READ_ONLY_OPTIMAL, output in GENERAL.
bool evaluate(VkCommandBuffer cb, const Inputs& in);
}  // namespace dlss

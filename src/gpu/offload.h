// A second, compute-only Vulkan device on the integrated GPU (when there is one besides the discrete GPU that
// renders). It runs latency-tolerant generators whose results are small enough to hand over through host memory a
// frame or two later: they come off the main GPU's frame entirely. Plain Vulkan, no VMA, no ray tracing.
#pragma once
#include <cstdint>
#include <vector>
#include <vulkan/vulkan.h>

namespace igpu {
struct Buf { VkBuffer b = VK_NULL_HANDLE; VkDeviceMemory m = VK_NULL_HANDLE; void* map = nullptr; VkDeviceSize size = 0; };
struct Img { VkImage img = VK_NULL_HANDLE; VkDeviceMemory m = VK_NULL_HANDLE; VkImageView view = VK_NULL_HANDLE; };

extern bool enabled;  // false: --no-igpu (everything stays on the main GPU)

// after the main device: picks an integrated GPU other than `main` (none, or no fp64: never ready); on a thread
void init(VkInstance instance, VkPhysicalDevice main);
bool ready();
const char* name();
VkDevice device();
void shutdown();

Buf mkBuf(VkDeviceSize size, VkBufferUsageFlags usage);  // host visible, coherent, mapped (host cached if offered)
Img mkImageArray(uint32_t w, uint32_t h, uint32_t layers, VkFormat fmt, VkImageUsageFlags usage);

// a compute pipeline from build/shaders/<name>.spv (name = "vtgen.comp"); one descriptor set of the given (binding, type) pairs, 16 bytes
// of push constants
struct Pipe { VkPipeline p = VK_NULL_HANDLE; VkPipelineLayout layout = VK_NULL_HANDLE; VkDescriptorSetLayout dsl = VK_NULL_HANDLE; VkDescriptorSet set = VK_NULL_HANDLE; };
Pipe mkPipe(const char* spv, const std::vector<std::pair<uint32_t, VkDescriptorType>>& bindings);
void writeBuf(const Pipe& p, uint32_t binding, VkDescriptorType t, const Buf& b);
void writeImg(const Pipe& p, uint32_t binding, const Img& im);

// one job in flight at a time: record into the returned command buffer, then submit(); poll done()
VkCommandBuffer begin();
void submit();
bool busy();  // submitted and not finished yet
}  // namespace igpu

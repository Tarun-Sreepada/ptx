#include "gpu/offload.h"
#include <cstdio>
#include <atomic>
#include <fstream>
#include <iterator>
#include <string>
#include <thread>
#include <vector>
#include "gpu/vk.h"


namespace igpu {
bool enabled = true;

static VkPhysicalDevice pd = VK_NULL_HANDLE;
static VkDevice d = VK_NULL_HANDLE;
static VkQueue q = VK_NULL_HANDLE;
static uint32_t fam = 0;
static VkPhysicalDeviceMemoryProperties mp;
static VkCommandPool pool = VK_NULL_HANDLE;
static VkCommandBuffer cb = VK_NULL_HANDLE;
static VkFence fence = VK_NULL_HANDLE;
static VkDescriptorPool dpool = VK_NULL_HANDLE;
static bool inFlight = false;
static char devName[256] = "";
static std::atomic<bool> up{false};  // set once everything below exists (init runs on its own thread)
static std::thread starter;

static bool initNow(VkInstance instance, VkPhysicalDevice main) {
    if (!enabled) return false;
    uint32_t n = 0;
    vkEnumeratePhysicalDevices(instance, &n, nullptr);
    std::vector<VkPhysicalDevice> pds(n);
    vkEnumeratePhysicalDevices(instance, &n, pds.data());
    for (auto p : pds) {
        if (p == main) continue;
        VkPhysicalDeviceProperties pr;
        vkGetPhysicalDeviceProperties(p, &pr);
        VkPhysicalDeviceFeatures f;
        vkGetPhysicalDeviceFeatures(p, &f);
        if (pr.deviceType != VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU || !f.shaderFloat64 || !f.shaderStorageImageExtendedFormats) continue;
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(p, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qp(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(p, &qn, qp.data());
        for (uint32_t i = 0; i < qn && !pd; i++)
            if (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { pd = p; fam = i; snprintf(devName, sizeof devName, "%s", pr.deviceName); }
        if (pd) break;
    }
    if (!pd) return false;
    float prio = 1.f;
    VkDeviceQueueCreateInfo qci{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qci.queueFamilyIndex = fam;
    qci.queueCount = 1;
    qci.pQueuePriorities = &prio;
    VkPhysicalDeviceFeatures feats{};
    feats.shaderFloat64 = VK_TRUE;
    feats.shaderStorageImageExtendedFormats = VK_TRUE;
    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.queueCreateInfoCount = 1;
    dci.pQueueCreateInfos = &qci;
    dci.pEnabledFeatures = &feats;
    if (vkCreateDevice(pd, &dci, nullptr, &d) != VK_SUCCESS) { pd = VK_NULL_HANDLE; return false; }
    vkGetDeviceQueue(d, fam, 0, &q);
    vkGetPhysicalDeviceMemoryProperties(pd, &mp);
    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = fam;
    VK_CHECK(vkCreateCommandPool(d, &cpi, nullptr, &pool));
    VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    ai.commandPool = pool;
    ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    ai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(d, &ai, &cb));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(d, &fci, nullptr, &fence));
    VkDescriptorPoolSize sizes[3] = {{VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, 8}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 8}, {VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 16}};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 8;
    dpi.poolSizeCount = 3;
    dpi.pPoolSizes = sizes;
    VK_CHECK(vkCreateDescriptorPool(d, &dpi, nullptr, &dpool));
    fprintf(stderr, "igpu: offload device %s\n", devName);
    up = true;
    return true;
}

// creating a second device takes a while: it happens beside the rest of startup, the generators wait for ready()
void init(VkInstance instance, VkPhysicalDevice main) {
    if (enabled) starter = std::thread([=] { initNow(instance, main); });
}

bool ready() { return up; }
const char* name() { return devName; }
VkDevice device() { return d; }

void shutdown() {
    if (starter.joinable()) starter.join();
    if (!up) return;
    up = false;
    vkDeviceWaitIdle(d);
    // not destroyed: the process is exiting and the OS frees it (destroying it would first need every child object freed)
}

static uint32_t memType(uint32_t bits, VkMemoryPropertyFlags want, VkMemoryPropertyFlags nice) {
    for (VkMemoryPropertyFlags f : {want | nice, want})
        for (uint32_t i = 0; i < mp.memoryTypeCount; i++)
            if ((bits >> i & 1) && (mp.memoryTypes[i].propertyFlags & f) == f) return i;
    fprintf(stderr, "igpu: no memory type\n");
    abort();
}

Buf mkBuf(VkDeviceSize size, VkBufferUsageFlags usage) {
    Buf b;
    b.size = size;
    VkBufferCreateInfo ci{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    ci.size = size;
    ci.usage = usage;
    VK_CHECK(vkCreateBuffer(d, &ci, nullptr, &b.b));
    VkMemoryRequirements r;
    vkGetBufferMemoryRequirements(d, b.b, &r);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = r.size;
    ai.memoryTypeIndex = memType(r.memoryTypeBits, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                 VK_MEMORY_PROPERTY_HOST_CACHED_BIT);  // the CPU reads the results back
    VK_CHECK(vkAllocateMemory(d, &ai, nullptr, &b.m));
    VK_CHECK(vkBindBufferMemory(d, b.b, b.m, 0));
    VK_CHECK(vkMapMemory(d, b.m, 0, VK_WHOLE_SIZE, 0, &b.map));
    return b;
}

Img mkImageArray(uint32_t w, uint32_t h, uint32_t layers, VkFormat fmt, VkImageUsageFlags usage) {
    Img im;
    VkImageCreateInfo ci{VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO};
    ci.imageType = VK_IMAGE_TYPE_2D;
    ci.format = fmt;
    ci.extent = {w, h, 1};
    ci.mipLevels = 1;
    ci.arrayLayers = layers;
    ci.samples = VK_SAMPLE_COUNT_1_BIT;
    ci.tiling = VK_IMAGE_TILING_OPTIMAL;
    ci.usage = usage;
    VK_CHECK(vkCreateImage(d, &ci, nullptr, &im.img));
    VkMemoryRequirements r;
    vkGetImageMemoryRequirements(d, im.img, &r);
    VkMemoryAllocateInfo ai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ai.allocationSize = r.size;
    ai.memoryTypeIndex = memType(r.memoryTypeBits, 0, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    VK_CHECK(vkAllocateMemory(d, &ai, nullptr, &im.m));
    VK_CHECK(vkBindImageMemory(d, im.img, im.m, 0));
    VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
    vi.image = im.img;
    vi.viewType = VK_IMAGE_VIEW_TYPE_2D_ARRAY;
    vi.format = fmt;
    vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, layers};
    VK_CHECK(vkCreateImageView(d, &vi, nullptr, &im.view));
    return im;
}

Pipe mkPipe(const char* spv, const std::vector<std::pair<uint32_t, VkDescriptorType>>& bindings) {
    Pipe p;
    std::vector<VkDescriptorSetLayoutBinding> lb;
    for (auto& [b, t] : bindings) lb.push_back({b, t, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr});
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = (uint32_t)lb.size();
    li.pBindings = lb.data();
    VK_CHECK(vkCreateDescriptorSetLayout(d, &li, nullptr, &p.dsl));
    VkPushConstantRange pr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 16};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &p.dsl;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pr;
    VK_CHECK(vkCreatePipelineLayout(d, &pli, nullptr, &p.layout));
    std::ifstream f(std::string(SHADER_DIR) + spv + ".spv", std::ios::binary);
    std::vector<char> code((std::istreambuf_iterator<char>(f)), {});
    if (code.empty()) { fprintf(stderr, "igpu: no %s.spv\n", spv); abort(); }
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = code.size();
    mi.pCode = (const uint32_t*)code.data();
    VkShaderModule m;
    VK_CHECK(vkCreateShaderModule(d, &mi, nullptr, &m));
    VkComputePipelineCreateInfo ci{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    ci.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, m, "main", nullptr};
    ci.layout = p.layout;
    VK_CHECK(vkCreateComputePipelines(d, VK_NULL_HANDLE, 1, &ci, nullptr, &p.p));
    vkDestroyShaderModule(d, m, nullptr);
    VkDescriptorSetAllocateInfo ai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    ai.descriptorPool = dpool;
    ai.descriptorSetCount = 1;
    ai.pSetLayouts = &p.dsl;
    VK_CHECK(vkAllocateDescriptorSets(d, &ai, &p.set));
    return p;
}

void writeBuf(const Pipe& p, uint32_t binding, VkDescriptorType t, const Buf& b) {
    VkDescriptorBufferInfo bi{b.b, 0, VK_WHOLE_SIZE};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = p.set;
    w.dstBinding = binding;
    w.descriptorCount = 1;
    w.descriptorType = t;
    w.pBufferInfo = &bi;
    vkUpdateDescriptorSets(d, 1, &w, 0, nullptr);
}

void writeImg(const Pipe& p, uint32_t binding, const Img& im) {
    VkDescriptorImageInfo ii{VK_NULL_HANDLE, im.view, VK_IMAGE_LAYOUT_GENERAL};
    VkWriteDescriptorSet w{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
    w.dstSet = p.set;
    w.dstBinding = binding;
    w.descriptorCount = 1;
    w.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE;
    w.pImageInfo = &ii;
    vkUpdateDescriptorSets(d, 1, &w, 0, nullptr);
}

VkCommandBuffer begin() {
    VK_CHECK(vkResetCommandBuffer(cb, 0));
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &bi));
    return cb;
}

void submit() {
    VK_CHECK(vkEndCommandBuffer(cb));
    VK_CHECK(vkResetFences(d, 1, &fence));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1;
    si.pCommandBuffers = &cb;
    VK_CHECK(vkQueueSubmit(q, 1, &si, fence));
    inFlight = true;
}

bool busy() {
    if (inFlight && vkGetFenceStatus(d, fence) == VK_SUCCESS) inFlight = false;
    return inFlight;
}
}  // namespace igpu

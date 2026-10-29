#include "gpu/context.h"
#include <vector>
#include <unordered_map>
#include <algorithm>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <atomic>
#include <fstream>
#include <cstdint>
#include <cstddef>
#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>
#include <vk_mem_alloc.h>
#include "gpu/dlss.h"
#include "gpu/rendergraph.h"
#include "gpu/pipelines.h"
#include "gpu/swapchain.h"
#include "render/frame_graph.h"
#include "render/targets.h"

// ---------------------------------------------------------------- vulkan globals
GLFWwindow* window;

VkInstance instance;

VkSurfaceKHR surface;

VkPhysicalDevice phys;

VkDevice dev;

VkQueue queue, computeQueue;

uint32_t qfam, computeFam;    // computeFam == qfam when there is no separate async compute family

uint32_t shareFams[2];       // buffers are CONCURRENT across both families (no ownership transfers)

static VkPhysicalDeviceMemoryProperties memProps;

VmaAllocator vma;

VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_PROPERTIES_KHR};

static float timestampPeriod;

char gpuName[256];

VkCommandPool cmdPool;

VkCommandBuffer uploadCmd;  // streamed model uploads + BLAS builds, submitted ahead of the frame graph

VkFence frameFence;

VkSemaphore acquireSem;

PFN_vkCreateAccelerationStructureKHR pCreateAS;

PFN_vkGetAccelerationStructureBuildSizesKHR pGetASSizes;

PFN_vkCmdBuildAccelerationStructuresKHR pCmdBuildAS;

PFN_vkGetAccelerationStructureDeviceAddressKHR pGetASAddr;

PFN_vkDestroyAccelerationStructureKHR pDestroyAS;
PFN_vkCmdDrawMeshTasksIndirectEXT pCmdDrawMeshTasksIndirect = nullptr;
bool hasMeshShaders = false;

// ---------------------------------------------------------------- device setup
static bool hasExt(VkPhysicalDevice pd, const char* name) {
    uint32_t n = 0;
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, nullptr);
    std::vector<VkExtensionProperties> e(n);
    vkEnumerateDeviceExtensionProperties(pd, nullptr, &n, e.data());
    for (auto& x : e)
        if (!strcmp(x.extensionName, name)) return true;
    return false;
}

// --validate: validation layer messages (warnings and errors), each distinct message id printed only a few times
static VkDebugUtilsMessengerEXT debugMessenger = VK_NULL_HANDLE;

static std::atomic<uint32_t> validationErrors{0};

static VKAPI_ATTR VkBool32 VKAPI_CALL validationCb(VkDebugUtilsMessageSeverityFlagBitsEXT sev, VkDebugUtilsMessageTypeFlagsEXT,
                                                   const VkDebugUtilsMessengerCallbackDataEXT* d, void*) {
    static std::mutex m;
    static std::unordered_map<int32_t, int> seen;
    bool err = sev & VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    if (err) validationErrors++;
    std::lock_guard<std::mutex> lk(m);
    if (++seen[d->messageIdNumber] > 3) return VK_FALSE;
    fprintf(stderr, "[vk %s] %s\n", err ? "ERROR" : "warn", d->pMessage);
    return VK_FALSE;
}

void initVulkan(bool validate) {
    VkApplicationInfo app{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    app.pApplicationName = "ptx";
    app.apiVersion = VK_API_VERSION_1_3;
    uint32_t n = 0;
    const char** gext = glfwGetRequiredInstanceExtensions(&n);
    std::vector<const char*> exts(gext, gext + n);
    std::vector<const char*> ngxInst, ngxDev;
    dlss::requiredExtensions(ngxInst, ngxDev);
    {
        uint32_t ne = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &ne, nullptr);
        std::vector<VkExtensionProperties> av(ne);
        vkEnumerateInstanceExtensionProperties(nullptr, &ne, av.data());
        for (auto e : ngxInst) {
            bool have = std::any_of(av.begin(), av.end(), [&](auto& x) { return !strcmp(x.extensionName, e); });
            bool dup = std::any_of(exts.begin(), exts.end(), [&](auto x) { return !strcmp(x, e); });
            if (have && !dup) exts.push_back(e);
        }
    }
    {   // HDR swapchain colour spaces
        uint32_t ne = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &ne, nullptr);
        std::vector<VkExtensionProperties> av(ne);
        vkEnumerateInstanceExtensionProperties(nullptr, &ne, av.data());
        for (auto& x : av)
            if (!strcmp(x.extensionName, VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME)) haveColorspaceExt = true;
        if (haveColorspaceExt) exts.push_back(VK_EXT_SWAPCHAIN_COLOR_SPACE_EXTENSION_NAME);
    }
    bool haveDebugUtils = false;
    {   // debug labels: Nsight Systems / RenderDoc show GPU passes by name
        uint32_t ne = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &ne, nullptr);
        std::vector<VkExtensionProperties> av(ne);
        vkEnumerateInstanceExtensionProperties(nullptr, &ne, av.data());
        for (auto& x : av) if (!strcmp(x.extensionName, VK_EXT_DEBUG_UTILS_EXTENSION_NAME)) haveDebugUtils = true;
        if (haveDebugUtils && !std::any_of(exts.begin(), exts.end(), [](auto x) { return !strcmp(x, VK_EXT_DEBUG_UTILS_EXTENSION_NAME); }))
            exts.push_back(VK_EXT_DEBUG_UTILS_EXTENSION_NAME);
    }
    const char* layer = "VK_LAYER_KHRONOS_validation";
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &app;
    ici.enabledExtensionCount = (uint32_t)exts.size();
    ici.ppEnabledExtensionNames = exts.data();
    if (validate) { ici.enabledLayerCount = 1; ici.ppEnabledLayerNames = &layer; }
    VkDebugUtilsMessengerCreateInfoEXT dmi{VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT};
    dmi.messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT;
    dmi.messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_TYPE_GENERAL_BIT_EXT;
    dmi.pfnUserCallback = validationCb;
    if (validate && haveDebugUtils) ici.pNext = &dmi;  // also covers vkCreateInstance / vkDestroyInstance themselves
    VK_CHECK(vkCreateInstance(&ici, nullptr, &instance));
    if (validate && haveDebugUtils)
        if (auto f = (PFN_vkCreateDebugUtilsMessengerEXT)vkGetInstanceProcAddr(instance, "vkCreateDebugUtilsMessengerEXT"))
            f(instance, &dmi, nullptr, &debugMessenger);
    if (validate) fprintf(stderr, "validation: %s\n", debugMessenger ? "on" : "layer on, no debug utils messenger");
    if (haveDebugUtils)
        rg::setDebugLabels((PFN_vkCmdBeginDebugUtilsLabelEXT)vkGetInstanceProcAddr(instance, "vkCmdBeginDebugUtilsLabelEXT"),
                           (PFN_vkCmdEndDebugUtilsLabelEXT)vkGetInstanceProcAddr(instance, "vkCmdEndDebugUtilsLabelEXT"));
    VK_CHECK(glfwCreateWindowSurface(instance, window, nullptr, &surface));

    uint32_t pdc = 0;
    vkEnumeratePhysicalDevices(instance, &pdc, nullptr);
    std::vector<VkPhysicalDevice> pds(pdc);
    vkEnumeratePhysicalDevices(instance, &pdc, pds.data());
    for (auto pd : pds) {
        if (!hasExt(pd, VK_KHR_RAY_QUERY_EXTENSION_NAME) || !hasExt(pd, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME)) continue;
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qp(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(pd, &qn, qp.data());
        for (uint32_t i = 0; i < qn; i++) {
            VkBool32 present = VK_FALSE;
            vkGetPhysicalDeviceSurfaceSupportKHR(pd, i, surface, &present);
            if ((qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && (qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && present) {
                VkPhysicalDeviceProperties pr;
                vkGetPhysicalDeviceProperties(pd, &pr);
                if (!phys || pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU) { phys = pd; qfam = i; }
                break;
            }
        }
    }
    if (!phys) { fprintf(stderr, "No GPU with VK_KHR_ray_query found\n"); exit(1); }

    VkPhysicalDeviceProperties2 p2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
    p2.pNext = &asProps;
    vkGetPhysicalDeviceProperties2(phys, &p2);
    timestampPeriod = p2.properties.limits.timestampPeriod;
    snprintf(gpuName, sizeof gpuName, "%s", p2.properties.deviceName);
    vkGetPhysicalDeviceMemoryProperties(phys, &memProps);

    // async compute: a compute-only family (NVIDIA: 8 queues, runs alongside the graphics queue)
    computeFam = qfam;
    {
        uint32_t qn = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, nullptr);
        std::vector<VkQueueFamilyProperties> qp(qn);
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &qn, qp.data());
        for (uint32_t i = 0; i < qn; i++)
            if ((qp[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && !(qp[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) && qp[i].timestampValidBits) { computeFam = i; break; }
    }
    shareFams[0] = qfam;
    shareFams[1] = computeFam;
    float prio = 1.f;
    VkDeviceQueueCreateInfo qcis[2] = {{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}, {VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO}};
    for (int i = 0; i < 2; i++) {
        qcis[i].queueFamilyIndex = i ? computeFam : qfam;
        qcis[i].queueCount = 1;
        qcis[i].pQueuePriorities = &prio;
    }

    VkPhysicalDeviceRayQueryFeaturesKHR rqf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_RAY_QUERY_FEATURES_KHR};
    rqf.rayQuery = VK_TRUE;
    VkPhysicalDeviceAccelerationStructureFeaturesKHR asf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_ACCELERATION_STRUCTURE_FEATURES_KHR};
    asf.accelerationStructure = VK_TRUE;
    asf.pNext = &rqf;
    VkPhysicalDeviceVulkan13Features f13{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_3_FEATURES};
    f13.dynamicRendering = VK_TRUE;
    f13.synchronization2 = VK_TRUE;
    f13.shaderDemoteToHelperInvocation = VK_TRUE;  // NGX (DLSS-RR) shader modules declare it
    f13.pNext = &asf;
    VkPhysicalDeviceVulkan12Features f12{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_VULKAN_1_2_FEATURES};
    f12.bufferDeviceAddress = VK_TRUE;
    f12.timelineSemaphore = VK_TRUE;   // render graph cross-queue sync
    f12.hostQueryReset = VK_TRUE;      // timestamps reset from the CPU, so any queue can write them
    f12.shaderSampledImageArrayNonUniformIndexing = VK_TRUE;  // model textures
    f12.pNext = &f13;
    VkPhysicalDeviceFeatures2 f2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    f2.pNext = &f12;
    // mesh shaders (VK_EXT_mesh_shader): vegetation meshlets culled per cluster (raster_model.task / .mesh)
    VkPhysicalDeviceMeshShaderFeaturesEXT msf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
    if (hasExt(phys, VK_EXT_MESH_SHADER_EXTENSION_NAME)) {
        VkPhysicalDeviceMeshShaderFeaturesEXT q{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_MESH_SHADER_FEATURES_EXT};
        VkPhysicalDeviceFeatures2 qf{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
        qf.pNext = &q;
        vkGetPhysicalDeviceFeatures2(phys, &qf);
        hasMeshShaders = q.taskShader && q.meshShader;
    }
    if (hasMeshShaders) {
        msf.taskShader = msf.meshShader = VK_TRUE;
        msf.pNext = f13.pNext;
        f13.pNext = &msf;
    }
    f2.features.geometryShader = VK_TRUE;     // gl_PrimitiveID in raster_model.frag (per-triangle material)
    f2.features.samplerAnisotropy = VK_TRUE;
    f2.features.shaderFloat64 = VK_TRUE;                       // groundgen.glsl: kilometre positions at sub-mm precision
    f2.features.shaderStorageImageExtendedFormats = VK_TRUE;   // vtgen.comp writes r16
    f2.features.fragmentStoresAndAtomics = VK_TRUE;  // material.glsl declares the world sim image (binding 93) writable for worldsim.comp
    f2.features.vertexPipelineStoresAndAtomics = VK_TRUE;  // raster_model.vert reads the wind map (written by wind.comp)

    std::vector<const char*> dext = {VK_KHR_SWAPCHAIN_EXTENSION_NAME, VK_KHR_ACCELERATION_STRUCTURE_EXTENSION_NAME,
                                     VK_KHR_RAY_QUERY_EXTENSION_NAME, VK_KHR_DEFERRED_HOST_OPERATIONS_EXTENSION_NAME};
    // ImGui's Vulkan backend loads the KHR alias of vkCmdBeginRendering
    if (hasExt(phys, VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME)) dext.push_back(VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME);
    for (auto e : ngxDev)  // DLSS-RR (NGX) device extensions, where the driver has them (BDA is core 1.2, enabled above)
        if (strcmp(e, VK_EXT_BUFFER_DEVICE_ADDRESS_EXTENSION_NAME) && hasExt(phys, e) && std::none_of(dext.begin(), dext.end(), [&](auto x) { return !strcmp(x, e); })) dext.push_back(e);
    if (hasMeshShaders) dext.push_back(VK_EXT_MESH_SHADER_EXTENSION_NAME);
    bool memBudget = hasExt(phys, VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);
    if (memBudget) dext.push_back(VK_EXT_MEMORY_BUDGET_EXTENSION_NAME);

    VkDeviceCreateInfo dci{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    dci.pNext = &f2;
    dci.queueCreateInfoCount = computeFam != qfam ? 2 : 1;
    dci.pQueueCreateInfos = qcis;
    dci.enabledExtensionCount = (uint32_t)dext.size();
    dci.ppEnabledExtensionNames = dext.data();
    VK_CHECK(vkCreateDevice(phys, &dci, nullptr, &dev));
    vkGetDeviceQueue(dev, qfam, 0, &queue);
    vkGetDeviceQueue(dev, computeFam, 0, &computeQueue);

    VmaAllocatorCreateInfo vci{};
    vci.flags = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT | (memBudget ? VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT : 0);
    vci.physicalDevice = phys;
    vci.device = dev;
    vci.instance = instance;
    vci.vulkanApiVersion = VK_API_VERSION_1_3;
    VK_CHECK(vmaCreateAllocator(&vci, &vma));

    pCreateAS = (PFN_vkCreateAccelerationStructureKHR)vkGetDeviceProcAddr(dev, "vkCreateAccelerationStructureKHR");
    pGetASSizes = (PFN_vkGetAccelerationStructureBuildSizesKHR)vkGetDeviceProcAddr(dev, "vkGetAccelerationStructureBuildSizesKHR");
    pCmdBuildAS = (PFN_vkCmdBuildAccelerationStructuresKHR)vkGetDeviceProcAddr(dev, "vkCmdBuildAccelerationStructuresKHR");
    pGetASAddr = (PFN_vkGetAccelerationStructureDeviceAddressKHR)vkGetDeviceProcAddr(dev, "vkGetAccelerationStructureDeviceAddressKHR");
    pDestroyAS = (PFN_vkDestroyAccelerationStructureKHR)vkGetDeviceProcAddr(dev, "vkDestroyAccelerationStructureKHR");
    if (hasMeshShaders) pCmdDrawMeshTasksIndirect = (PFN_vkCmdDrawMeshTasksIndirectEXT)vkGetDeviceProcAddr(dev, "vkCmdDrawMeshTasksIndirectEXT");
    hasMeshShaders = hasMeshShaders && pCmdDrawMeshTasksIndirect;

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = qfam;
    VK_CHECK(vkCreateCommandPool(dev, &cpi, nullptr, &cmdPool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = cmdPool;
    cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    cai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(dev, &cai, &uploadCmd));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = VK_FENCE_CREATE_SIGNALED_BIT;
    VK_CHECK(vkCreateFence(dev, &fci, nullptr, &frameFence));
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VK_CHECK(vkCreateSemaphore(dev, &sci, nullptr, &acquireSem));
    if (!dlss::init(instance, phys, dev)) upscaler = dlss::OFF;
    rg::QueueInfo qs[rg::QUEUE_COUNT] = {{queue, qfam}, {computeQueue, computeFam}};
    graph.init(phys, dev, qs, timestampPeriod);
}

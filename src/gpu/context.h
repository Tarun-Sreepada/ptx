#pragma once
#include <cstdint>
#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <vk_mem_alloc.h>

extern GLFWwindow* window;

extern VkInstance instance;

extern VkSurfaceKHR surface;

extern VkPhysicalDevice phys;

extern VkDevice dev;

extern VkQueue queue, computeQueue;

extern uint32_t qfam, computeFam;

extern uint32_t shareFams[2];

extern VmaAllocator vma;

extern VkPhysicalDeviceAccelerationStructurePropertiesKHR asProps;

extern char gpuName[256];

extern VkCommandPool cmdPool;

extern VkCommandBuffer uploadCmd;

extern VkFence frameFence;

extern VkSemaphore acquireSem;

extern PFN_vkCreateAccelerationStructureKHR pCreateAS;

extern PFN_vkGetAccelerationStructureBuildSizesKHR pGetASSizes;

extern PFN_vkCmdBuildAccelerationStructuresKHR pCmdBuildAS;

extern PFN_vkGetAccelerationStructureDeviceAddressKHR pGetASAddr;

extern PFN_vkDestroyAccelerationStructureKHR pDestroyAS;

extern PFN_vkCmdDrawMeshTasksIndirectEXT pCmdDrawMeshTasksIndirect;
extern bool hasMeshShaders;  // VK_EXT_mesh_shader with task + mesh stages

void initVulkan(bool validate);

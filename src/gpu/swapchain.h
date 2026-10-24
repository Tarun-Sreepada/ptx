#pragma once
#include <vector>
#include <vulkan/vulkan.h>
#include "gpu/vk.h"

extern VkSwapchainKHR swapchain;

extern bool swapCanCopy, swapBGR;

extern VkExtent2D swapExtent;

extern std::vector<VkImage> swapImages;

extern std::vector<VkImageView> swapViews;

extern std::vector<VkSemaphore> renderDone;

extern bool swapDirty;

extern Image uiImg, presentImg;

extern VkFormat uiFmt;

extern bool hdrModeAvail[3];

extern int hdrActive;

extern bool haveColorspaceExt;

extern bool vsync;

void createSwapchain();

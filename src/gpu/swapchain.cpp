#include "gpu/swapchain.h"
#include <vector>
#include <algorithm>
#include <cstdint>
#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>
#include "app/settings.h"
#include "gpu/context.h"
#include "gpu/descriptors.h"
#include "gpu/memory.h"
#include "gpu/pipelines.h"
#include "gpu/vk.h"

VkSwapchainKHR swapchain;

static VkFormat swapFmt;

// screenshots (F12 scene, Shift+F12 with the UI): <repo>/screenshots/*.png, path copied to the clipboard
bool swapCanCopy = false, swapBGR = true;

VkExtent2D swapExtent;

std::vector<VkImage> swapImages;

std::vector<VkImageView> swapViews;

std::vector<VkSemaphore> renderDone;

bool swapDirty = false;

Image uiImg, presentImg;        // HDR present: UI rendered SDR, then composed into presentImg (swapchain size)

VkFormat uiFmt = VK_FORMAT_B8G8R8A8_UNORM;

bool hdrModeAvail[3] = {true, false, false};  // SDR, HDR10 (PQ), scRGB offered by the surface

int hdrActive = 0;                            // what the swapchain currently is

bool haveColorspaceExt = false;

bool vsync = true;          // FIFO: frames paced to the display, no uneven-present judder

// Swapchain-sized images for the HDR present (UI target + composed frame); always present so descriptors stay valid.
static void createPresentImages() {
    if (uiImg.img) { vkDeviceWaitIdle(dev); freeImage(uiImg); freeImage(presentImg); }
    uiImg = mkImageUsage(swapExtent.width, swapExtent.height, uiFmt, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_SAMPLED_BIT);
    presentImg = mkImageUsage(swapExtent.width, swapExtent.height, VK_FORMAT_R16G16B16A16_SFLOAT,
                              VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT);
    oneShot([](VkCommandBuffer cb) {
        for (Image* im : {&uiImg, &presentImg})
            imgBarrier(cb, im->img, VK_IMAGE_LAYOUT_UNDEFINED, VK_IMAGE_LAYOUT_GENERAL, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, 0,
                       VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_ACCESS_MEMORY_READ_BIT | VK_ACCESS_MEMORY_WRITE_BIT);
    });
    if (descSet) writePresentDescriptors();
}

void createSwapchain() {
    VkSurfaceCapabilitiesKHR caps;
    vkGetPhysicalDeviceSurfaceCapabilitiesKHR(phys, surface, &caps);
    int fw, fh;
    glfwGetFramebufferSize(window, &fw, &fh);
    swapExtent = caps.currentExtent;
    if (swapExtent.width == 0xFFFFFFFF) {
        swapExtent.width = std::clamp((uint32_t)fw, caps.minImageExtent.width, caps.maxImageExtent.width);
        swapExtent.height = std::clamp((uint32_t)fh, caps.minImageExtent.height, caps.maxImageExtent.height);
    }
    uint32_t n = 0;
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface, &n, nullptr);
    std::vector<VkSurfaceFormatKHR> fmts(n);
    vkGetPhysicalDeviceSurfaceFormatsKHR(phys, surface, &n, fmts.data());
    VkSurfaceFormatKHR fmt = fmts[0];
    for (auto& f : fmts)
        if ((f.format == VK_FORMAT_B8G8R8A8_UNORM || f.format == VK_FORMAT_R8G8B8A8_UNORM) && f.colorSpace == VK_COLOR_SPACE_SRGB_NONLINEAR_KHR) { fmt = f; break; }
    uiFmt = fmt.format;  // ImGui's target format: the SDR swapchain, or the UI image in HDR mode
    // HDR: HDR10 (10-bit PQ, Rec.2020) or scRGB (fp16 linear, extended sRGB), when the surface offers them
    hdrModeAvail[1] = hdrModeAvail[2] = false;
    VkSurfaceFormatKHR hdrFmt[3] = {fmt, fmt, fmt};
    for (auto& f : fmts) {
        if (f.colorSpace == VK_COLOR_SPACE_HDR10_ST2084_EXT && (f.format == VK_FORMAT_A2B10G10R10_UNORM_PACK32 || f.format == VK_FORMAT_A2R10G10B10_UNORM_PACK32)) { hdrModeAvail[1] = true; hdrFmt[1] = f; }
        if (f.colorSpace == VK_COLOR_SPACE_EXTENDED_SRGB_LINEAR_EXT && f.format == VK_FORMAT_R16G16B16A16_SFLOAT) { hdrModeAvail[2] = true; hdrFmt[2] = f; }
    }
    hdrActive = S.hdrMode > 0 && S.hdrMode < 3 && hdrModeAvail[S.hdrMode] ? S.hdrMode : 0;
    fmt = hdrFmt[hdrActive];
    swapFmt = fmt.format;

    vkGetPhysicalDeviceSurfacePresentModesKHR(phys, surface, &n, nullptr);
    std::vector<VkPresentModeKHR> modes(n);
    vkGetPhysicalDeviceSurfacePresentModesKHR(phys, surface, &n, modes.data());
    VkPresentModeKHR mode = VK_PRESENT_MODE_FIFO_KHR;
    if (!vsync) {
        for (auto m : modes) if (m == VK_PRESENT_MODE_IMMEDIATE_KHR) mode = m;
        for (auto m : modes) if (m == VK_PRESENT_MODE_MAILBOX_KHR) mode = m;
    }

    uint32_t count = caps.minImageCount + 1;
    if (caps.maxImageCount && count > caps.maxImageCount) count = caps.maxImageCount;

    VkSwapchainCreateInfoKHR ci{VK_STRUCTURE_TYPE_SWAPCHAIN_CREATE_INFO_KHR};
    ci.surface = surface;
    ci.minImageCount = count;
    ci.imageFormat = fmt.format;
    ci.imageColorSpace = fmt.colorSpace;
    ci.imageExtent = swapExtent;
    ci.imageArrayLayers = 1;
    ci.imageUsage = VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT | VK_IMAGE_USAGE_TRANSFER_DST_BIT;
    swapCanCopy = (caps.supportedUsageFlags & VK_IMAGE_USAGE_TRANSFER_SRC_BIT) != 0;  // screenshots with the UI
    if (swapCanCopy) ci.imageUsage |= VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    swapBGR = fmt.format == VK_FORMAT_B8G8R8A8_UNORM || fmt.format == VK_FORMAT_B8G8R8A8_SRGB;
    ci.imageSharingMode = VK_SHARING_MODE_EXCLUSIVE;
    ci.preTransform = caps.currentTransform;
    ci.compositeAlpha = VK_COMPOSITE_ALPHA_OPAQUE_BIT_KHR;
    ci.presentMode = mode;
    ci.clipped = VK_TRUE;
    ci.oldSwapchain = swapchain;
    VkSwapchainKHR sc;
    VK_CHECK(vkCreateSwapchainKHR(dev, &ci, nullptr, &sc));
    for (auto v : swapViews) vkDestroyImageView(dev, v, nullptr);
    for (auto s : renderDone) vkDestroySemaphore(dev, s, nullptr);
    if (swapchain) vkDestroySwapchainKHR(dev, swapchain, nullptr);
    swapchain = sc;

    vkGetSwapchainImagesKHR(dev, swapchain, &n, nullptr);
    swapImages.resize(n);
    vkGetSwapchainImagesKHR(dev, swapchain, &n, swapImages.data());
    swapViews.resize(n);
    renderDone.resize(n);
    for (uint32_t i = 0; i < n; i++) {
        VkImageViewCreateInfo vi{VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO};
        vi.image = swapImages[i];
        vi.viewType = VK_IMAGE_VIEW_TYPE_2D;
        vi.format = swapFmt;
        vi.subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1};
        VK_CHECK(vkCreateImageView(dev, &vi, nullptr, &swapViews[i]));
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(dev, &sci, nullptr, &renderDone[i]));
    }
    createPresentImages();
}

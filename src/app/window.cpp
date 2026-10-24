#include "app/window.h"

#include <string>
#include <vector>

#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>

#include "editor/editor.h"
#include "gpu/context.h"
#include "gpu/swapchain.h"
#include "render/targets.h"

uint32_t resOptions[5][2] = {{1280, 720}, {1600, 900}, {1920, 1080}, {2560, 1440}, {0, 0}};
bool resFollowWindow = true;
int windowMode = 0;
int pendingWindowMode = -1;

namespace {
constexpr int FALLBACK_W = 1600, FALLBACK_H = 900;  // no monitor reported (headless / nested compositors)
constexpr const char* TITLE = "ptx - path traced grass";
}

void createWindow(bool fixedOutput) {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    GLFWmonitor* mon = glfwGetPrimaryMonitor();
    const GLFWvidmode* vm = mon ? glfwGetVideoMode(mon) : nullptr;
    int ww = vm ? vm->width : FALLBACK_W, wh = vm ? vm->height : FALLBACK_H;
    if (windowMode == 1) glfwWindowHint(GLFW_DECORATED, GLFW_FALSE);
    if (windowMode == 0) glfwWindowHint(GLFW_MAXIMIZED, GLFW_TRUE);
    window = glfwCreateWindow(ww, wh, TITLE, windowMode == 2 ? mon : nullptr, nullptr);
    if (fixedOutput) resFollowWindow = false;
    else { OW = (uint32_t)ww; OH = (uint32_t)wh; }
    glfwSetFramebufferSizeCallback(window, [](GLFWwindow*, int, int) { swapDirty = true; });
    glfwSetDropCallback(window, [](GLFWwindow*, int n, const char** paths) { editor::dropFiles(std::vector<std::string>(paths, paths + n)); });
}

void applyWindowMode(int mode) {
    GLFWmonitor* mon = glfwGetPrimaryMonitor();
    const GLFWvidmode* vm = mon ? glfwGetVideoMode(mon) : nullptr;
    if (!vm) return;
    if (mode == 2) {
        glfwSetWindowMonitor(window, mon, 0, 0, vm->width, vm->height, vm->refreshRate);
    } else {
        if (glfwGetWindowMonitor(window)) glfwSetWindowMonitor(window, nullptr, 0, 0, vm->width, vm->height, 0);
        glfwSetWindowAttrib(window, GLFW_DECORATED, mode == 1 ? GLFW_FALSE : GLFW_TRUE);
        if (mode == 1) { glfwSetWindowPos(window, 0, 0); glfwSetWindowSize(window, vm->width, vm->height); }
        else glfwMaximizeWindow(window);
    }
    windowMode = mode;
    swapDirty = true;
}

#include "tools/screenshot.h"
#include <vector>
#include <string>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <cstdint>
#include <ctime>
#include <cstddef>
#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <stb_image_write.h>
#include "audio/audio.h"
#include "gpu/rendergraph.h"
#include "app/frame_state.h"
#include "gpu/context.h"
#include "gpu/memory.h"
#include "gpu/vk.h"
#include "render/frame_graph.h"
#include "render/targets.h"
#include "tools/movelog.h"

bool shotUIReq = false, shotUIPending = false, shotSceneReq = false;

uint32_t uiShotW = 0, uiShotH = 0;

std::string shotMsg;

double shotMsgT = -100.0;

Buffer uiShotBuf;  // screenshot with the UI: swapchain readback

// --shot <file.ppm>: save the internal-res output after a warm-up and exit (for testing)
std::string newShotPath() {
    std::error_code ec;
    std::filesystem::create_directories("screenshots", ec);
    char name[96];
    time_t t = time(nullptr);
    strftime(name, sizeof name, "screenshots/ptx_%Y%m%d_%H%M%S", localtime(&t));
    std::string base = name, p = base + ".png";
    for (int i = 2; std::filesystem::exists(p, ec); i++) p = base + "_" + std::to_string(i) + ".png";
    return std::filesystem::absolute(p).string();
}

void shotDone(const std::string& p, bool ok) {
    if (ok) writeShotInfo(p);
    shotMsg = ok ? "saved " + p + " + .txt scene (path copied: paste it to Claude)" : "screenshot failed: " + p;
    shotMsgT = glfwGetTime();
    if (ok) glfwSetClipboardString(window, p.c_str());
    printf("%s\n", shotMsg.c_str());
}

// rgba (or bgra) rows -> PNG, alpha forced opaque
bool writePng(const std::string& p, const uint8_t* px, uint32_t w, uint32_t h, bool bgr) {
    std::vector<uint8_t> rgb((size_t)w * h * 3);
    for (size_t i = 0; i < (size_t)w * h; i++) {
        rgb[i * 3 + 0] = px[i * 4 + (bgr ? 2 : 0)];
        rgb[i * 3 + 1] = px[i * 4 + 1];
        rgb[i * 3 + 2] = px[i * 4 + (bgr ? 0 : 2)];
    }
    return stbi_write_png(p.c_str(), (int)w, (int)h, 3, rgb.data(), (int)w * 3) != 0;
}

// the scene without UI (the output image), as PNG
void saveScenePng() {
    vkDeviceWaitIdle(dev);
    Buffer st = mkBuf((VkDeviceSize)OW * OH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    oneShot([&](VkCommandBuffer cb) {
        VkBufferImageCopy r{};
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.imageExtent = {OW, OH, 1};
        vkCmdCopyImageToBuffer(cb, imgs[I_OUT].img, VK_IMAGE_LAYOUT_GENERAL, st.b, 1, &r);
    });
    std::string p = newShotPath();
    shotDone(p, writePng(p, (const uint8_t*)st.map, OW, OH, false));
    freeBuf(st);
}

void saveShot(const char* path) {
    vkDeviceWaitIdle(dev);
    Buffer st = mkBuf((VkDeviceSize)OW * OH * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    oneShot([&](VkCommandBuffer cb) {
        VkBufferImageCopy r{};
        r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 0, 1};
        r.imageExtent = {OW, OH, 1};
        vkCmdCopyImageToBuffer(cb, imgs[I_OUT].img, VK_IMAGE_LAYOUT_GENERAL, st.b, 1, &r);
    });
    const uint8_t* px = (const uint8_t*)st.map;
    std::string ps(path);
    if (ps.size() > 4 && ps.compare(ps.size() - 4, 4, ".png") == 0) {  // PNG + the scene beside it
        if (writePng(ps, px, OW, OH, false)) writeShotInfo(ps);
    } else {
        FILE* f = fopen(path, "wb");
        fprintf(f, "P6\n%u %u\n255\n", OW, OH);
        for (uint32_t i = 0; i < OW * OH; i++) fwrite(px + i * 4, 1, 3, f);
        fclose(f);
    }
    freeBuf(st);
    printf("GPU %.2f ms:", gpuEmaTotal);
    for (auto& t : graph.timings()) printf(" %s%s=%.2f", t.queue == rg::COMPUTE ? "[async] " : "", t.name.c_str(), t.ms);
    printf("\n");
    auto as = audio::stats();
    printf("audio: %u emitters, %u meshes, %u instances, %u scene tris, sim %.1f ms, dsp %.2f ms\n", as.emitters, as.meshes, as.instances,
           as.sceneTris, as.simMs, as.dspMs);
}

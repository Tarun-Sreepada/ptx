#include "gpu/dlss.h"

#include <cstdio>
#include <cstring>
#include <string>
#include <unistd.h>

#ifdef PTX_DLSS
#include <nvsdk_ngx_vk.h>
#include <nvsdk_ngx_helpers_vk.h>  // before the RR helper, which uses its types
#include <nvsdk_ngx_helpers_dlssd_vk.h>
#endif

namespace dlss {
const char* modeName(Mode m) {
    static const char* n[MODE_COUNT] = {"Off", "DLAA", "Quality", "Balanced", "Performance", "Ultra Performance"};
    return n[m];
}
float renderScale(Mode m) {
    static const float s[MODE_COUNT] = {1.f, 1.f, 2.f / 3.f, 0.58f, 0.5f, 1.f / 3.f};
    return s[m];
}

#ifdef PTX_DLSS
namespace {
VkDevice device = VK_NULL_HANDLE;
NVSDK_NGX_Parameter* params = nullptr;
NVSDK_NGX_Handle* feature = nullptr;
bool ok = false;

NVSDK_NGX_PerfQuality_Value quality(Mode m) {
    switch (m) {
        case DLAA: return NVSDK_NGX_PerfQuality_Value_DLAA;
        case QUALITY: return NVSDK_NGX_PerfQuality_Value_MaxQuality;
        case BALANCED: return NVSDK_NGX_PerfQuality_Value_Balanced;
        case ULTRA_PERFORMANCE: return NVSDK_NGX_PerfQuality_Value_UltraPerformance;
        default: return NVSDK_NGX_PerfQuality_Value_MaxPerf;
    }
}
NVSDK_NGX_Resource_VK res(const Img& i, bool rw) {
    return NVSDK_NGX_Create_ImageView_Resource_VK(i.view, i.image, {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, 1}, i.format, i.w, i.h, rw);
}
}  // namespace

void requiredExtensions(std::vector<const char*>& instance, std::vector<const char*>& device) {
    unsigned ni = 0, nd = 0;
    const char **ie = nullptr, **de = nullptr;
    if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_RequiredExtensions(&ni, &ie, &nd, &de))) return;
    auto add = [](std::vector<const char*>& v, const char* e) {
        for (auto x : v) if (!strcmp(x, e)) return;
        v.push_back(e);
    };
    for (unsigned i = 0; i < ni; i++) add(instance, ie[i]);
    for (unsigned i = 0; i < nd; i++) add(device, de[i]);
}

bool init(VkInstance inst, VkPhysicalDevice phys, VkDevice dev) {
    device = dev;
    // the feature libraries (libnvidia-ngx-dlssd.so.*) are copied next to the executable at build time
    static std::wstring exeDir;
    char buf[4096];
    ssize_t n = readlink("/proc/self/exe", buf, sizeof buf - 1);
    std::string dir = n > 0 ? std::string(buf, n) : std::string(".");
    dir = dir.substr(0, dir.find_last_of('/'));
    exeDir.assign(dir.begin(), dir.end());
    static const wchar_t* paths[1];
    paths[0] = exeDir.c_str();
    NVSDK_NGX_FeatureCommonInfo info{};
    info.PathListInfo.Path = paths;
    info.PathListInfo.Length = 1;
    info.LoggingInfo.MinimumLoggingLevel = NVSDK_NGX_LOGGING_LEVEL_OFF;
    NVSDK_NGX_Result r = NVSDK_NGX_VULKAN_Init_with_ProjectID("5b8a2f6e-3c1d-4e7a-9f42-7d0c1e6a9b3f", NVSDK_NGX_ENGINE_TYPE_CUSTOM, "1.0",
                                                             L"/tmp", inst, phys, dev, vkGetInstanceProcAddr, vkGetDeviceProcAddr, &info);
    if (NVSDK_NGX_FAILED(r)) { fprintf(stderr, "dlss: NGX init failed (0x%x)\n", (unsigned)r); return false; }
    if (NVSDK_NGX_FAILED(NVSDK_NGX_VULKAN_GetCapabilityParameters(&params))) { fprintf(stderr, "dlss: no capability parameters\n"); return false; }
    int avail = 0, needsDriver = 0;
    NVSDK_NGX_Parameter_GetI(params, NVSDK_NGX_Parameter_SuperSamplingDenoising_Available, &avail);
    NVSDK_NGX_Parameter_GetI(params, NVSDK_NGX_Parameter_SuperSamplingDenoising_NeedsUpdatedDriver, &needsDriver);
    if (!avail) {
        fprintf(stderr, "dlss: Ray Reconstruction not available%s\n", needsDriver ? " (driver update needed)" : "");
        return false;
    }
    ok = true;
    printf("dlss: Ray Reconstruction available\n");
    return true;
}

bool available() { return ok; }

void release() {
    if (feature) NVSDK_NGX_VULKAN_ReleaseFeature(feature);
    feature = nullptr;
}

void shutdown() {
    release();
    if (params) NVSDK_NGX_VULKAN_DestroyParameters(params);
    params = nullptr;
    if (device) NVSDK_NGX_VULKAN_Shutdown1(device);
    device = VK_NULL_HANDLE;
    ok = false;
}

bool create(VkCommandBuffer cb, Mode m, uint32_t rw, uint32_t rh, uint32_t ow, uint32_t oh) {
    release();
    if (!ok || m == OFF) return false;
    NVSDK_NGX_DLSSD_Create_Params cp{};
    cp.InDenoiseMode = NVSDK_NGX_DLSS_Denoise_Mode_DLUnified;
    cp.InRoughnessMode = NVSDK_NGX_DLSS_Roughness_Mode_Packed;  // normals.w
    cp.InUseHWDepth = NVSDK_NGX_DLSS_Depth_Type_Linear;         // view-space z
    cp.InWidth = rw;
    cp.InHeight = rh;
    cp.InTargetWidth = ow;
    cp.InTargetHeight = oh;
    cp.InPerfQualityValue = quality(m);
    cp.InFeatureCreateFlags = NVSDK_NGX_DLSS_Feature_Flags_IsHDR | NVSDK_NGX_DLSS_Feature_Flags_MVLowRes;
    NVSDK_NGX_Result r = NGX_VULKAN_CREATE_DLSSD_EXT1(device, cb, 1, 1, &feature, params, &cp);
    if (NVSDK_NGX_FAILED(r)) { fprintf(stderr, "dlss: create failed (0x%x)\n", (unsigned)r); feature = nullptr; return false; }
    return true;
}

bool evaluate(VkCommandBuffer cb, const Inputs& in) {
    if (!feature) return false;
    NVSDK_NGX_Resource_VK color = res(in.color, false), diff = res(in.diffuseAlbedo, false), spec = res(in.specularAlbedo, false),
                          nrm = res(in.normalRoughness, false), mv = res(in.motion, false), z = res(in.viewZ, false), out = res(in.output, true);
    NVSDK_NGX_VK_DLSSD_Eval_Params ep{};
    ep.pInColor = &color;
    ep.pInOutput = &out;
    ep.pInDiffuseAlbedo = &diff;
    ep.pInSpecularAlbedo = &spec;
    ep.pInNormals = &nrm;
    ep.pInRoughness = &nrm;
    ep.pInDepth = &z;
    ep.pInMotionVectors = &mv;
    ep.InJitterOffsetX = in.jitterX;
    ep.InJitterOffsetY = in.jitterY;
    ep.InRenderSubrectDimensions = {in.color.w, in.color.h};
    ep.InReset = in.reset ? 1 : 0;
    ep.InMVScaleX = ep.InMVScaleY = 1.f;
    ep.pInWorldToViewMatrix = const_cast<float*>(in.worldToView);
    ep.pInViewToClipMatrix = const_cast<float*>(in.viewToClip);
    ep.InFrameTimeDeltaInMsec = in.frameMs;
    NVSDK_NGX_Result r = NGX_VULKAN_EVALUATE_DLSSD_EXT(cb, feature, params, &ep);
    if (NVSDK_NGX_FAILED(r)) {
        static bool warned = false;
        if (!warned) fprintf(stderr, "dlss: evaluate failed (0x%x)\n", (unsigned)r);
        warned = true;
        return false;
    }
    return true;
}
#else
void requiredExtensions(std::vector<const char*>&, std::vector<const char*>&) {}
bool init(VkInstance, VkPhysicalDevice, VkDevice) { return false; }
bool available() { return false; }
void shutdown() {}
bool create(VkCommandBuffer, Mode, uint32_t, uint32_t, uint32_t, uint32_t) { return false; }
void release() {}
bool evaluate(VkCommandBuffer, const Inputs&) { return false; }
#endif
}  // namespace dlss

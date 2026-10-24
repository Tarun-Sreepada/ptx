// Main panel sections: Path tracing / ReSTIR / GI settings, surface shading, the denoiser and debug views.
#include "ui/panels/panels.h"

#include <vector>
#include <memory>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cfloat>
#include <cstddef>
#include <vulkan/vulkan.h>
#include <imgui.h>
#include "audio/audio.h"
#include "gpu/dlss.h"
#include "core/input.h"
#include "physics/physics.h"
#include "gpu/rendergraph.h"
#include "characters/character.h"
#include "world/terrain.h"
#include "world/world.h"
#include "app/frame_state.h"
#include "app/settings.h"
#include "app/window.h"
#include "core/limits.h"
#include "core/math.h"
#include "destruction/buildings.h"
#include "destruction/thrown.h"
#include "destruction/tree_damage.h"
#include "gpu/context.h"
#include "gpu/memory.h"
#include "gpu/swapchain.h"
#include "render/bake.h"
#include "render/frame_graph.h"
#include "render/ground_textures.h"
#include "render/ground_vt.h"
#include "render/lights.h"
#include "render/targets.h"
#include "scene/camera.h"
#include "scene/characters.h"
#include "scene/grass.h"
#include "scene/model_instances.h"
#include "scene/models.h"
#include "scene/rocks.h"
#include "scene/scene_gpu.h"
#include "scene/stream_sync.h"
#include "scene/vegetation.h"
#include "sim/debris.h"
#include "sim/effects.h"
#include "sim/water.h"
#include "sim/world_sim.h"
#include "sky/lightning.h"
#include "tools/bench.h"
#include "ui/vegetation_panel.h"
#include "world/terrain_query.h"

void panelRendering() {
    if (!ImGui::CollapsingHeader("Rendering")) return;
    ImGui::BeginDisabled(bench.running);
    int resIdx = 2;
    for (int i = 0; i < 4; i++) if (resOptions[i][0] == OW && resOptions[i][1] == OH) resIdx = i;
    if (resFollowWindow) resIdx = 4;
    char winRes[32];
    snprintf(winRes, sizeof winRes, "Window (%ux%u)", swapExtent.width, swapExtent.height);
    const char* resNames[] = {"720p", "900p", "1080p", "1440p", winRes};
    if (ImGui::Combo("Output res", &resIdx, resNames, 5)) {
        resFollowWindow = resIdx == 4;
        if (resFollowWindow) { resOptions[4][0] = swapExtent.width; resOptions[4][1] = swapExtent.height; }
        if (resOptions[resIdx][0] != OW || resOptions[resIdx][1] != OH) pendingRes = resIdx;
    }
    ImGui::BeginDisabled(!dlss::available());
    int um = upscaler;
    const char* umNames[dlss::MODE_COUNT];
    for (int i = 0; i < dlss::MODE_COUNT; i++) umNames[i] = dlss::modeName((dlss::Mode)i);
    if (ImGui::Combo("DLSS Ray Reconstruction", &um, umNames, dlss::MODE_COUNT) && um != upscaler) { upscaler = (dlss::Mode)um; pendingRes = resIdx; }
    ImGui::EndDisabled();
    if (!dlss::available()) ImGui::TextDisabled("DLSS-RR unavailable (needs RTX GPU + driver, SDK in external/dlss)");
    else if (rrActive) ImGui::TextDisabled("RR denoises + upscales; denoiser settings below are bypassed");
    ImGui::Checkbox("Raster primary (G-buffer)", &S.rasterPrimary);
    ImGui::SameLine();
    if (ImGui::Checkbox("Global illumination (I)", &S.giEnabled)) resetHistory = true;
    ImGui::SliderInt("Bounces", &S.bounces, 1, 4);
    ImGui::Checkbox("ReSTIR DI", &S.restir);
    ImGui::BeginDisabled(!S.restir);
    ImGui::SameLine(); ImGui::Checkbox("temporal", &S.restirTemporal);
    ImGui::SameLine(); ImGui::Checkbox("spatial", &S.restirSpatial);
    ImGui::SliderInt("Candidates (M)", &S.candidates, 1, 32);
    ImGui::SliderInt("Spatial samples", &S.spatialSamples, 1, 8);
    ImGui::SliderFloat("Spatial radius px", &S.spatialRadius, 2, 40);
    ImGui::EndDisabled();
    ImGui::Checkbox("ReSTIR GI", &S.restirGI);
    ImGui::BeginDisabled(!S.restirGI);
    ImGui::SameLine(); ImGui::Checkbox("temporal##gi", &S.giTemporal);
    ImGui::SameLine(); ImGui::Checkbox("spatial##gi", &S.giSpatial);
    ImGui::SliderInt("GI spatial samples", &S.giSpatialSamples, 1, 8);
    if (ImGui::Checkbox("radiance cache", &S.giCache)) resetHistory = true;
    ImGui::SameLine(); if (ImGui::Checkbox("checkerboard", &S.giChecker)) resetHistory = true;
    ImGui::SameLine(); if (ImGui::Checkbox("probes at bounce 2", &S.giProbes)) resetHistory = true;
    ImGui::SliderFloat("GI spatial radius px", &S.giRadius, 2, 60);
    ImGui::SliderFloat("GI history cap (M)", &S.giMaxM, 1, 100);
    ImGui::EndDisabled();

    ImGui::EndDisabled();
}

void panelSurfaceShading() {
    if (!ImGui::CollapsingHeader("Surface shading")) return;
    if (ImGui::Checkbox("Subsurface scattering (random walk)", &S.sss)) resetHistory = true;
    ImGui::SetItemTooltip("Skin (characters), wax, marble and lying snow: light walks through the material and comes out\n"
                          "elsewhere (soft shadows edges, glowing thin parts). Leaves and grass use a thin-slab solution.");
    ImGui::BeginDisabled(!S.sss);
    ImGui::SliderInt("Walk steps", &S.sssSteps, 4, 64);
    ImGui::SetItemTooltip("Scattering events per walk before it surfaces where it is (cost: one short ray each)");
    ImGui::SliderFloat("Scattering distance", &S.sssScale, 0.2f, 5.f, "%.2fx", ImGuiSliderFlags_Logarithmic);
    ImGui::Checkbox("Snow", &S.snowSSS);
    ImGui::EndDisabled();
    if (ImGui::Checkbox("Micro-shadowing + specular occlusion", &S.microShadow)) resetHistory = true;
    ImGui::SetItemTooltip("Material occlusion (cavities in rock, ground relief, AO textures) shadows direct light at grazing\n"
                          "angles and occludes glossy reflections");
}

void panelDenoiserDebugView() {
    if (!ImGui::CollapsingHeader("Denoiser & debug view")) return;
    ImGui::BeginDisabled(bench.running);
    ImGui::Checkbox("Temporal accumulation", &S.accumulate);
    ImGui::SameLine(); ImGui::Checkbox("A-trous filter", &S.denoise);
    ImGui::SliderInt("History max", &S.historyMax, 1, 128);
    ImGui::SliderInt("A-trous iterations", &S.atrousIters, 0, 5);
    ImGui::SliderFloat("Firefly clamp", &S.fireflyClamp, 1, 100, "%.0f", ImGuiSliderFlags_Logarithmic);
    if (ImGui::Button("Reset history")) resetHistory = true;
    const char* views[] = {"Final", "Raw (no denoise)", "Direct only", "Indirect only", "Albedo", "Normals", "History length", "Variance"};
    ImGui::Combo("View", &S.viewMode, views, 8);

    ImGui::EndDisabled();
}

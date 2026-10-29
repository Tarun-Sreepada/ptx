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
#include "ui/search.h"
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
    if (!ui::CollapsingHeader("Rendering")) return;
    ImGui::BeginDisabled(bench.running);
    int resIdx = 2;
    for (int i = 0; i < 4; i++) if (resOptions[i][0] == OW && resOptions[i][1] == OH) resIdx = i;
    if (resFollowWindow) resIdx = 4;
    char winRes[32];
    snprintf(winRes, sizeof winRes, "Window (%ux%u)", swapExtent.width, swapExtent.height);
    const char* resNames[] = {"720p", "900p", "1080p", "1440p", winRes};
    if (ui::Combo("Output res", &resIdx, resNames, 5)) {
        resFollowWindow = resIdx == 4;
        if (resFollowWindow) { resOptions[4][0] = swapExtent.width; resOptions[4][1] = swapExtent.height; }
        if (resOptions[resIdx][0] != OW || resOptions[resIdx][1] != OH) pendingRes = resIdx;
    }
    ImGui::BeginDisabled(!dlss::available());
    int um = upscaler;
    const char* umNames[dlss::MODE_COUNT];
    for (int i = 0; i < dlss::MODE_COUNT; i++) umNames[i] = dlss::modeName((dlss::Mode)i);
    if (ui::Combo("DLSS Ray Reconstruction", &um, umNames, dlss::MODE_COUNT) && um != upscaler) { upscaler = (dlss::Mode)um; pendingRes = resIdx; }
    ImGui::EndDisabled();
    if (!dlss::available()) ui::TextDisabled("DLSS-RR unavailable (needs RTX GPU + driver, SDK in external/dlss)");
    else if (rrActive) ui::TextDisabled("RR denoises + upscales; denoiser settings below are bypassed");
    ui::Checkbox("Raster primary (G-buffer)", &S.rasterPrimary);
    ui::SameLine();
    ui::Checkbox("Occlusion culling", &S.occlusionCull);
    ui::SameLine();
    if (ui::Checkbox("Global illumination (I)", &S.giEnabled)) resetHistory = true;
    ui::SliderInt("Bounces", &S.bounces, 1, 4);
    ui::Checkbox("ReSTIR DI", &S.restir);
    ImGui::BeginDisabled(!S.restir);
    ui::SameLine(); ui::Checkbox("temporal", &S.restirTemporal);
    ui::SameLine(); ui::Checkbox("spatial", &S.restirSpatial);
    ui::SliderInt("Candidates (M)", &S.candidates, 1, 32);
    ui::SliderInt("Spatial samples", &S.spatialSamples, 1, 8);
    ui::SliderFloat("Spatial radius px", &S.spatialRadius, 2, 40);
    ImGui::EndDisabled();
    ui::Checkbox("ReSTIR GI", &S.restirGI);
    ImGui::BeginDisabled(!S.restirGI);
    ui::SameLine(); ui::Checkbox("temporal##gi", &S.giTemporal);
    ui::SameLine(); ui::Checkbox("spatial##gi", &S.giSpatial);
    ui::SliderInt("GI spatial samples", &S.giSpatialSamples, 1, 8);
    if (ui::Checkbox("radiance cache", &S.giCache)) resetHistory = true;
    ui::SameLine(); if (ui::Checkbox("checkerboard", &S.giChecker)) resetHistory = true;
    ui::SameLine(); if (ui::Checkbox("probes at bounce 2", &S.giProbes)) resetHistory = true;
    ImGui::BeginDisabled(!S.giChecker);
    if (ui::SliderInt("checker square px", &S.giCheckerCell, 1, 32)) resetHistory = true;
    if (ui::Checkbox("quarter rate (1 in 4 traced per frame)", &S.giQuarter)) resetHistory = true;
    ui::SetItemTooltip("Size of the checkerboard squares that skip new GI rays each frame.\n"
                          "8 or more: whole GPU warps skip, which is where the time saving comes from.\n"
                          "Below 8: finer, smoother pattern, but most of the saving is lost (idle lanes still cost).\n"
                          "Large squares need a GI spatial radius bigger than the square to fill the gaps.");
    ImGui::EndDisabled();
    ui::SliderFloat("GI spatial radius px", &S.giRadius, 2, 60);
    ui::SliderFloat("GI history cap (M)", &S.giMaxM, 1, 100);
    ImGui::EndDisabled();

    ImGui::EndDisabled();
}

void panelSurfaceShading() {
    if (!ui::CollapsingHeader("Surface shading")) return;
    if (ui::Checkbox("Subsurface scattering (random walk)", &S.sss)) resetHistory = true;
    ui::SetItemTooltip("Skin (characters), wax, marble and lying snow: light walks through the material and comes out\n"
                          "elsewhere (soft shadows edges, glowing thin parts). Leaves and grass use a thin-slab solution.");
    ImGui::BeginDisabled(!S.sss);
    ui::SliderInt("Walk steps", &S.sssSteps, 4, 64);
    ui::SetItemTooltip("Scattering events per walk before it surfaces where it is (cost: one short ray each)");
    ui::SliderFloat("Scattering distance", &S.sssScale, 0.2f, 5.f, "%.2fx", ImGuiSliderFlags_Logarithmic);
    ui::Checkbox("Snow", &S.snowSSS);
    ImGui::EndDisabled();
    if (ui::Checkbox("Micro-shadowing + specular occlusion", &S.microShadow)) resetHistory = true;
    ui::SetItemTooltip("Material occlusion (cavities in rock, ground relief, AO textures) shadows direct light at grazing\n"
                          "angles and occludes glossy reflections");
}

void panelDenoiserDebugView() {
    if (!ui::CollapsingHeader("Denoiser & debug view")) return;
    ImGui::BeginDisabled(bench.running);
    ui::Checkbox("Temporal accumulation", &S.accumulate);
    ui::SameLine(); ui::Checkbox("A-trous filter", &S.denoise);
    ui::SliderInt("History max", &S.historyMax, 1, 128);
    ui::SliderInt("A-trous iterations", &S.atrousIters, 0, 5);
    ui::SliderFloat("Firefly clamp", &S.fireflyClamp, 1, 100, "%.0f", ImGuiSliderFlags_Logarithmic);
    if (ui::Button("Reset history")) resetHistory = true;
    const char* views[] = {"Final", "Raw (no denoise)", "Direct only", "Indirect only", "Albedo", "Normals", "History length", "Variance"};
    ui::Combo("View", &S.viewMode, views, 8);

    ImGui::EndDisabled();
}

// Main panel sections: Display & window, exposure / tonemap / bloom, lens effects, camera & movement.
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

void panelDisplayWindow() {
    if (!ui::CollapsingHeader("Display & window")) return;
    const char* wm[] = {"Windowed (maximised)", "Borderless (fills the screen)", "Fullscreen (exclusive)"};
    int m = windowMode;
    if (ui::Combo("Window", &m, wm, 3) && m != windowMode) pendingWindowMode = m;
    if (ui::Checkbox("VSync##disp", &vsync)) swapDirty = true;
    const char* hm[] = {"SDR", "HDR10 (PQ)", "scRGB (fp16)"};
    int h = S.hdrMode;
    if (ui::Combo("Output", &h, hm, 3) && h != S.hdrMode) { S.hdrMode = h; swapDirty = true; }
    if (S.hdrMode && !hdrModeAvail[S.hdrMode])
        ui::TextColored({1, 0.6f, 0.3f, 1}, haveColorspaceExt ? "The display / compositor doesn't offer this HDR mode: showing SDR"
                                                                 : "No VK_EXT_swapchain_colorspace: HDR unavailable, showing SDR");
    ImGui::BeginDisabled(hdrActive == 0);
    ui::SliderFloat("Paper white (nits)", &S.paperWhite, 80, 500, "%.0f");
    ui::SliderFloat("Peak brightness (nits)", &S.peakNits, 400, 4000, "%.0f");
    ImGui::EndDisabled();
}

void panelExposureTonemapBloom() {
    if (!ui::CollapsingHeader("Exposure, tonemap & bloom")) return;
    ui::Checkbox("Physical camera (f-stop / shutter / ISO)", &S.physCam);
    if (S.physCam) {
        ui::SliderFloat("f-stop", &S.fStop, 1.f, 22.f, "f/%.1f", ImGuiSliderFlags_Logarithmic);
        ui::SliderFloat("Shutter", &S.shutterDen, 1.f, 8000.f, "1/%.0f s", ImGuiSliderFlags_Logarithmic);
        ui::SliderFloat("ISO", &S.iso, 50.f, 25600.f, "%.0f", ImGuiSliderFlags_Logarithmic);
        float ev100 = log2f(S.fStop * S.fStop * S.shutterDen) - log2f(S.iso / 100.f);
        ui::Text("EV100 %.1f (sunny day ~15, overcast ~12, indoors ~7, night ~0)", ev100);
        ui::SliderFloat("Scene luminance scale", &S.sceneLum, 100.f, 1e6f, "%.0f cd/m2", ImGuiSliderFlags_Logarithmic);
        ui::SetItemTooltip("cd/m^2 per unit of scene radiance (calibration of the renderer's units)");
    } else ui::Checkbox("Auto exposure (eye adaptation)", &S.autoExposure);
    if (S.physCam) {
    } else if (S.autoExposure) {
        ui::SliderFloat("Exposure compensation (EV)", &S.evComp, -4, 4, "%.1f");
        ui::SliderFloat("Adapt to brighter (1/s)", &S.adaptUp, 0.1f, 10, "%.1f", ImGuiSliderFlags_Logarithmic);
        ui::SliderFloat("Adapt to darker (1/s)", &S.adaptDown, 0.1f, 10, "%.1f", ImGuiSliderFlags_Logarithmic);
        const float* ex = (const float*)((const char*)exposureBuf.map + 512);
        ui::Text("Metered %.1f EV, exposure x%.2f", ex[1], ex[0]);
    } else {
        ui::SliderFloat("Exposure", &S.exposure, 0.01f, 64, "%.2f", ImGuiSliderFlags_Logarithmic);
    }
    const char* tm[] = {"AgX (neutral, filmic)", "AgX punchy", "ACES (fitted)", "None (clip)"};
    ui::Combo("Tonemapper", &S.tonemapper, tm, 4);
    ui::SliderFloat("Bloom", &S.bloom, 0, 0.3f, "%.3f");
    ui::SliderFloat("Sharpen (CAS)", &S.sharpen, 0, 1, "%.2f");
}

void panelLensCameraEffects() {
    if (!ui::CollapsingHeader("Lens & camera effects")) return;
    ui::Checkbox("Depth of field", &S.dof);
    ui::SliderFloat("Focus distance", &S.focusDist, 0.f, 200.f, S.focusDist <= 0.f ? "auto (centre)" : "%.1f m", ImGuiSliderFlags_Logarithmic);
    if (!S.physCam) ui::SliderFloat("Aperture", &S.fStop, 1.f, 22.f, "f/%.1f", ImGuiSliderFlags_Logarithmic);
    ui::Text("Focal length %.0f mm (from the FOV, full frame)", 12.f / tanf(cam.fovDeg * 0.5f * 3.14159265f / 180.f));
    ui::Checkbox("Motion blur", &S.motionBlur);
    if (!S.physCam) ui::SliderFloat("Shutter angle", &S.mbShutter, 0.f, 1.f, "%.2f of the frame");
    else ui::TextDisabled("Shutter time from the physical camera");
    ui::SliderFloat("Vignetting", &S.vignette, 0.f, 1.f, "%.2f");
    ui::SliderFloat("Chromatic aberration", &S.chromAb, 0.f, 8.f, "%.1f px");
    ui::SliderFloat("Sensor noise", &S.grain, 0.f, 3.f, "%.2f");
    if (S.physCam) ui::SameLine(), ui::TextDisabled("x ISO");
}

void panelCameraMovement() {
    if (!ui::CollapsingHeader("Camera & movement")) return;
    ui::SliderFloat("FOV", &cam.fovDeg, 30, 110, "%.0f deg");
    if (ui::Checkbox("Walk (F)", &walkMode) && walkMode) {
        V3 feet = cam.pos - V3{0, 1.65f, 0};
        feet.y = std::max(feet.y, heightFn(feet.x, feet.z) + 0.05f);
        physics::spawnPlayer(&feet.x);
    }
    if (walkMode) { ui::SameLine(); ui::Text("%s  feet %.1f %.1f %.1f", grounded ? "grounded" : "airborne", playerFeet.x, playerFeet.y, playerFeet.z); }
    ui::TextDisabled("RMB + mouse: look   WASD/QE: move   Shift: fast   Ctrl: turbo (fly)   F: walk/fly   Space: jump");
    ui::SliderFloat("Fly speed", &flySpeed, 0.25f, 20.f, "%.2fx", ImGuiSliderFlags_Logarithmic);
    ui::SliderFloat("Head bob (walking)", &S.headBob, 0.f, 1.5f, "%.2f");

}

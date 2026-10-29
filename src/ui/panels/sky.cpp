// Main panel sections: Sun & sky, clouds, wind, night sky & lightning, light orbs.
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
#include "sky/day_cycle.h"
#include "sky/lightning.h"
#include "tools/bench.h"
#include "ui/vegetation_panel.h"
#include "world/terrain_query.h"

void panelSunSkyExposure() {
    if (!ui::CollapsingHeader("Sun, sky & exposure")) return;
    ImGui::BeginDisabled(bench.running);
    ui::SliderFloat("Haze", &S.fogScale, 0, 100, "%.0f");
    ui::Checkbox("Day / night cycle", &S.dayCycle);
    ui::SetItemTooltip("The clock runs and the sun follows its real path for the latitude and the day of the year");
    if (S.dayCycle) {
        int hh = (int)S.timeOfDay, mm = (int)((S.timeOfDay - hh) * 60.f);
        char lbl[32];
        snprintf(lbl, sizeof lbl, "%02d:%02d", hh, mm);
        if (ui::SliderFloat("Time of day", &S.timeOfDay, 0.f, 24.f, lbl)) sunAt(S.timeOfDay, S.dayOfYear, S.latitude, S.sunElev, S.sunAzim);
        ui::SliderFloat("Day length", &S.dayMinutes, 1.f, 120.f, "%.0f min", ImGuiSliderFlags_Logarithmic);
        ui::SliderFloat("Day of year", &S.dayOfYear, 0.f, 365.f, "%.0f");
        ui::SetItemTooltip("Season: the sun's height and the length of the day (172 = midsummer, 355 = midwinter)");
    } else {
        ui::SliderFloat("Sun elevation", &S.sunElev, -10, 90);
        ui::SliderFloat("Sun azimuth", &S.sunAzim, 0, 360);
    }
    ui::SliderFloat("Sun intensity", &S.sunIntensity, 0, 20);
    ui::SliderFloat("Sun size (deg)", &S.sunRadiusDeg, 0.1f, 5);
    ui::SliderFloat("Sky intensity", &S.skyIntensity, 0, 3);
    ui::SliderFloat("Exposure", &S.exposure, 0.1f, 8, "%.2f", ImGuiSliderFlags_Logarithmic);
    if (ui::Button("Night")) { S.sunElev = -8; S.lightIntensity = 3; S.exposure = 3; }
    ui::SameLine();
    if (ui::Button("Sunset")) { S.sunElev = 6; S.sunIntensity = 6; S.exposure = 1.8f; }
    ui::SameLine();
    if (ui::Button("Day")) { S.sunElev = 45; S.sunIntensity = 6; S.exposure = 1.2f; }
    ImGui::EndDisabled();
}

void panelClouds() {
    if (!ui::CollapsingHeader("Clouds")) return;
    ImGui::BeginDisabled(bench.running);
    ui::Checkbox("Enabled##clouds", &S.clouds);
    {
        static const int divs[] = {1, 2, 3, 4, 6, 8, 12, 16};
        constexpr int ND = (int)(sizeof divs / sizeof divs[0]);
        int ci = 0;
        while (ci + 1 < ND && divs[ci + 1] <= S.cloudDiv) ci++;
        char name[16];
        snprintf(name, sizeof name, "1/%d", divs[ci]);
        if (ui::SliderInt("Cloud resolution", &ci, 0, ND - 1, name)) S.cloudDiv = divs[ci];
        ui::SetItemTooltip("Sky clouds are marched at this fraction of the render resolution and upsampled.\nLooking up costs much less at 1/2 or 1/4");
    }
    ui::SliderFloat("Cloud coverage", &S.cloudCoverage, 0, 1);
    ui::SetItemTooltip("How much of the sky is cloud. Every cloud type below is a share of this: 0 = clear sky");
    ui::SliderFloat("Cloud density", &S.cloudDensity, 0.002f, 0.2f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ui::SliderFloat("Cloud drift", &S.cloudSpeed, 0, 10);
    ui::SliderFloat("Cloud shadows", &S.cloudShadow, 0, 1);
    ui::SeparatorText("Cloud types (share of coverage)");
    auto share = [](float w) { return S.cloudCoverage * w * 100.f; };
    ui::SliderFloat("Cumulus", &S.cumulus, 0, 1);
    ui::SameLine(); ui::TextDisabled("%.0f%%", share(S.cumulus));
    ui::SliderFloat("Cumulus base (m)", &S.cloudBase, 300, 4000, "%.0f");
    ui::SliderFloat("Cumulus thickness (m)", &S.cloudThickness, 200, 4000, "%.0f");
    ui::Spacing();
    ui::SliderFloat("Stratus", &S.stratus, 0, 1);
    ui::SameLine(); ui::TextDisabled("%.0f%%", share(S.stratus));
    ui::SliderFloat("Stratus base (m)", &S.stratusBase, 100, 2500, "%.0f");
    ui::SliderFloat("Stratus thickness (m)", &S.stratusThick, 50, 1500, "%.0f");
    ui::SliderFloat("Stratus lumpiness", &S.stratusLumpy, 0, 1);
    ui::SetItemTooltip("0 = flat stratus sheet, 1 = broken stratocumulus cells");
    ui::Spacing();
    ui::SliderFloat("Cumulonimbus", &S.cumulonimbus, 0, 1);
    ui::SameLine(); ui::TextDisabled("%.0f%%", share(S.cumulonimbus));
    ui::SliderFloat("Cumulonimbus top (m)", &S.cbTop, 5000, 15000, "%.0f");
    ui::Spacing();
    ui::SliderFloat("Cirrus", &S.cirrus, 0, 1);
    ui::SameLine(); ui::TextDisabled("%.0f%%", share(S.cirrus));
    ui::SliderFloat("Cirrus altitude (m)", &S.cirrusAlt, 6000, 13000, "%.0f");
    ui::SliderFloat("Cirrus strand length", &S.cirrusLength, 0.3f, 3.f, "%.2fx");
    ui::SliderFloat("Cirrus strand fineness", &S.cirrusFine, 0, 1);
    ui::SliderFloat("Cirrus puffiness", &S.cirrusPuff, 0, 1);
    ui::SetItemTooltip("0 = fine hair-like strands, 1 = cotton tufts trailing wisps");
    ImGui::EndDisabled();
}

void panelWind() {
    if (!ui::CollapsingHeader("Wind")) return;
    ui::Checkbox("Simulated wind map", &S.windMap);
    ui::Checkbox("Refit grass BVH (rebuild every 8 frames / on camera move)", &S.grassRefit);
    ui::SetItemTooltip("Gust fronts from different directions sweep through; where they collide, pressure waves ripple the grass and water");
    ImGui::BeginDisabled(!S.windMap);
    ui::SliderFloat("Gust fronts", &S.gustStrength, 0, 1, "%.2fx");
    ui::SliderFloat("Front direction spread", &S.gustSpreadDeg, 0, 180, "%.0f deg");
    ui::SliderFloat("Wave speed", &S.windWaveSpeed, 2, 30, "%.1f m/s");
    ui::SliderFloat("Wave damping", &S.windDamping, 0, 2, "%.2f /s");
    ui::SliderFloat("Gust response", &S.windRelax, 0.1f, 5, "%.2f /s");
    ImGui::EndDisabled();
    ImGui::BeginDisabled(bench.running);
    ui::SliderFloat("Wind strength", &S.windStrength, 0, 1.5f);
    ui::SliderFloat("Wind speed", &S.windSpeed, 0, 6);
    ui::SliderFloat("Wind direction", &S.windDirDeg, 0, 360);
    ImGui::EndDisabled();
}

void panelNightSkyLightning() {
    if (!ui::CollapsingHeader("Night sky & lightning")) return;
    ui::SliderFloat("Moon phase", &S.moonPhase, 0.f, 1.f, "%.2f (0.5 full)");
    ui::SliderFloat("Latitude", &S.latitude, -80.f, 80.f, "%.0f deg");
    ui::SliderFloat("Night light boost", &S.nightBoost, 1.f, 1000.f, "%.0fx", ImGuiSliderFlags_Logarithmic);
    ui::SliderFloat("Stars", &S.stars, 0.f, 3.f); ui::SliderFloat("Milky Way", &S.milkyWay, 0.f, 3.f);
    ui::SliderFloat("Airglow", &S.airglow, 0.f, 3.f); ui::SliderFloat("Twinkle", &S.twinkle, 0.f, 2.f);
    ui::Checkbox("Purkinje shift", &S.purkinje);
    ui::Checkbox("Lightning in storms", &S.lightning);
    ui::SliderFloat("Strikes / min", &S.strikesPerMin, 0.f, 30.f);
    if (ui::Button("Strike now")) boltRequest = true;
    ui::SliderFloat("Rainbow", &S.rainbow, 0.f, 2.f);
    ui::Checkbox("Fog banks", &S.fogBanks);
    ui::SliderFloat("Bank density", &S.bankDensity, 0.f, 0.2f); ui::SliderFloat("Bank coverage", &S.bankCover, 0.f, 1.f);
}

void panelLights() {
    if (!ui::CollapsingHeader("Lights")) return;
    ImGui::BeginDisabled(bench.running);
    ui::SliderInt("Light orbs", &S.lightCount, 0, MAX_ORBS);
    ui::SliderFloat("Orb range (m)", &S.orbRange, 20.f, 400.f, "%.0f");
    if (ui::IsItemHovered()) ImGui::SetTooltip("Orbs live in a square this wide around you and fade out at its edge");
    ui::SliderFloat("Orb intensity", &S.lightIntensity, 0, 10);
    ui::SliderFloat("Model light scale", &lightScale, 1e-4f, 0.1f, "%.4f", ImGuiSliderFlags_Logarithmic);
    ImGui::EndDisabled();
}

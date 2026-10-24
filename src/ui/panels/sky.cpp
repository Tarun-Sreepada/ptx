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
    if (!ImGui::CollapsingHeader("Sun, sky & exposure")) return;
    ImGui::BeginDisabled(bench.running);
    ImGui::SliderFloat("Haze", &S.fogScale, 0, 100, "%.0f");
    ImGui::Checkbox("Day / night cycle", &S.dayCycle);
    ImGui::SetItemTooltip("The clock runs and the sun follows its real path for the latitude and the day of the year");
    if (S.dayCycle) {
        int hh = (int)S.timeOfDay, mm = (int)((S.timeOfDay - hh) * 60.f);
        char lbl[32];
        snprintf(lbl, sizeof lbl, "%02d:%02d", hh, mm);
        if (ImGui::SliderFloat("Time of day", &S.timeOfDay, 0.f, 24.f, lbl)) sunAt(S.timeOfDay, S.dayOfYear, S.latitude, S.sunElev, S.sunAzim);
        ImGui::SliderFloat("Day length", &S.dayMinutes, 1.f, 120.f, "%.0f min", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Day of year", &S.dayOfYear, 0.f, 365.f, "%.0f");
        ImGui::SetItemTooltip("Season: the sun's height and the length of the day (172 = midsummer, 355 = midwinter)");
    } else {
        ImGui::SliderFloat("Sun elevation", &S.sunElev, -10, 90);
        ImGui::SliderFloat("Sun azimuth", &S.sunAzim, 0, 360);
    }
    ImGui::SliderFloat("Sun intensity", &S.sunIntensity, 0, 20);
    ImGui::SliderFloat("Sun size (deg)", &S.sunRadiusDeg, 0.1f, 5);
    ImGui::SliderFloat("Sky intensity", &S.skyIntensity, 0, 3);
    ImGui::SliderFloat("Exposure", &S.exposure, 0.1f, 8, "%.2f", ImGuiSliderFlags_Logarithmic);
    if (ImGui::Button("Night")) { S.sunElev = -8; S.lightIntensity = 3; S.exposure = 3; }
    ImGui::SameLine();
    if (ImGui::Button("Sunset")) { S.sunElev = 6; S.sunIntensity = 6; S.exposure = 1.8f; }
    ImGui::SameLine();
    if (ImGui::Button("Day")) { S.sunElev = 45; S.sunIntensity = 6; S.exposure = 1.2f; }
    ImGui::EndDisabled();
}

void panelClouds() {
    if (!ImGui::CollapsingHeader("Clouds")) return;
    ImGui::BeginDisabled(bench.running);
    ImGui::Checkbox("Enabled##clouds", &S.clouds);
    ImGui::SliderFloat("Cloud coverage", &S.cloudCoverage, 0, 1);
    ImGui::SetItemTooltip("How much of the sky is cloud. Every cloud type below is a share of this: 0 = clear sky");
    ImGui::SliderFloat("Cloud density", &S.cloudDensity, 0.002f, 0.2f, "%.3f", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Cloud drift", &S.cloudSpeed, 0, 10);
    ImGui::SliderFloat("Cloud shadows", &S.cloudShadow, 0, 1);
    ImGui::SeparatorText("Cloud types (share of coverage)");
    auto share = [](float w) { return S.cloudCoverage * w * 100.f; };
    ImGui::SliderFloat("Cumulus", &S.cumulus, 0, 1);
    ImGui::SameLine(); ImGui::TextDisabled("%.0f%%", share(S.cumulus));
    ImGui::SliderFloat("Cumulus base (m)", &S.cloudBase, 300, 4000, "%.0f");
    ImGui::SliderFloat("Cumulus thickness (m)", &S.cloudThickness, 200, 4000, "%.0f");
    ImGui::Spacing();
    ImGui::SliderFloat("Stratus", &S.stratus, 0, 1);
    ImGui::SameLine(); ImGui::TextDisabled("%.0f%%", share(S.stratus));
    ImGui::SliderFloat("Stratus base (m)", &S.stratusBase, 100, 2500, "%.0f");
    ImGui::SliderFloat("Stratus thickness (m)", &S.stratusThick, 50, 1500, "%.0f");
    ImGui::SliderFloat("Stratus lumpiness", &S.stratusLumpy, 0, 1);
    ImGui::SetItemTooltip("0 = flat stratus sheet, 1 = broken stratocumulus cells");
    ImGui::Spacing();
    ImGui::SliderFloat("Cumulonimbus", &S.cumulonimbus, 0, 1);
    ImGui::SameLine(); ImGui::TextDisabled("%.0f%%", share(S.cumulonimbus));
    ImGui::SliderFloat("Cumulonimbus top (m)", &S.cbTop, 5000, 15000, "%.0f");
    ImGui::Spacing();
    ImGui::SliderFloat("Cirrus", &S.cirrus, 0, 1);
    ImGui::SameLine(); ImGui::TextDisabled("%.0f%%", share(S.cirrus));
    ImGui::SliderFloat("Cirrus altitude (m)", &S.cirrusAlt, 6000, 13000, "%.0f");
    ImGui::SliderFloat("Cirrus strand length", &S.cirrusLength, 0.3f, 3.f, "%.2fx");
    ImGui::SliderFloat("Cirrus strand fineness", &S.cirrusFine, 0, 1);
    ImGui::SliderFloat("Cirrus puffiness", &S.cirrusPuff, 0, 1);
    ImGui::SetItemTooltip("0 = fine hair-like strands, 1 = cotton tufts trailing wisps");
    ImGui::EndDisabled();
}

void panelWind() {
    if (!ImGui::CollapsingHeader("Wind")) return;
    ImGui::Checkbox("Simulated wind map", &S.windMap);
    ImGui::Checkbox("Refit grass BVH (rebuild every 8 frames / on camera move)", &S.grassRefit);
    ImGui::SetItemTooltip("Gust fronts from different directions sweep through; where they collide, pressure waves ripple the grass and water");
    ImGui::BeginDisabled(!S.windMap);
    ImGui::SliderFloat("Gust fronts", &S.gustStrength, 0, 1, "%.2fx");
    ImGui::SliderFloat("Front direction spread", &S.gustSpreadDeg, 0, 180, "%.0f deg");
    ImGui::SliderFloat("Wave speed", &S.windWaveSpeed, 2, 30, "%.1f m/s");
    ImGui::SliderFloat("Wave damping", &S.windDamping, 0, 2, "%.2f /s");
    ImGui::SliderFloat("Gust response", &S.windRelax, 0.1f, 5, "%.2f /s");
    ImGui::EndDisabled();
    ImGui::BeginDisabled(bench.running);
    ImGui::SliderFloat("Wind strength", &S.windStrength, 0, 1.5f);
    ImGui::SliderFloat("Wind speed", &S.windSpeed, 0, 6);
    ImGui::SliderFloat("Wind direction", &S.windDirDeg, 0, 360);
    ImGui::EndDisabled();
}

void panelNightSkyLightning() {
    if (!ImGui::CollapsingHeader("Night sky & lightning")) return;
    ImGui::SliderFloat("Moon phase", &S.moonPhase, 0.f, 1.f, "%.2f (0.5 full)");
    ImGui::SliderFloat("Latitude", &S.latitude, -80.f, 80.f, "%.0f deg");
    ImGui::SliderFloat("Night light boost", &S.nightBoost, 1.f, 1000.f, "%.0fx", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Stars", &S.stars, 0.f, 3.f); ImGui::SliderFloat("Milky Way", &S.milkyWay, 0.f, 3.f);
    ImGui::SliderFloat("Airglow", &S.airglow, 0.f, 3.f); ImGui::SliderFloat("Twinkle", &S.twinkle, 0.f, 2.f);
    ImGui::Checkbox("Purkinje shift", &S.purkinje);
    ImGui::Checkbox("Lightning in storms", &S.lightning);
    ImGui::SliderFloat("Strikes / min", &S.strikesPerMin, 0.f, 30.f);
    if (ImGui::Button("Strike now")) boltRequest = true;
    ImGui::SliderFloat("Rainbow", &S.rainbow, 0.f, 2.f);
    ImGui::Checkbox("Fog banks", &S.fogBanks);
    ImGui::SliderFloat("Bank density", &S.bankDensity, 0.f, 0.2f); ImGui::SliderFloat("Bank coverage", &S.bankCover, 0.f, 1.f);
}

void panelLights() {
    if (!ImGui::CollapsingHeader("Lights")) return;
    ImGui::BeginDisabled(bench.running);
    ImGui::SliderInt("Light orbs", &S.lightCount, 0, MAX_ORBS);
    ImGui::SliderFloat("Orb range (m)", &S.orbRange, 20.f, 400.f, "%.0f");
    if (ImGui::IsItemHovered()) ImGui::SetTooltip("Orbs live in a square this wide around you and fade out at its edge");
    ImGui::SliderFloat("Orb intensity", &S.lightIntensity, 0, 10);
    ImGui::SliderFloat("Model light scale", &lightScale, 1e-4f, 0.1f, "%.4f", ImGuiSliderFlags_Logarithmic);
    ImGui::EndDisabled();
}

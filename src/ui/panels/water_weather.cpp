// Main panel sections: River & water, fog & god rays, weather (precipitation, snow, wetness).
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
#include "sky/weather_cycle.h"
#include "tools/bench.h"
#include "ui/vegetation_panel.h"
#include "world/terrain_query.h"

void panelRiverWater() {
    if (!ImGui::CollapsingHeader("River & water")) return;
    bool shape = ImGui::SliderFloat("Half-width", &gRiver.width, 2.f, 14.f, "%.1f m");
    shape |= ImGui::SliderFloat("Depth", &gRiver.depth, 0.3f, 4.f, "%.2f m");
    shape |= ImGui::SliderFloat("Riverbed bumps", &gRiver.bumps, 0.f, 1.5f, "%.2f m");
    ImGui::SetItemTooltip("Uneven bed instead of a smooth half-pipe: humps, scours and ridges (also in the collision)");
    if (shape) { terrainDirty = placeDirty = true; stream::terrainChanged(); }  // carved terrain + collision follow
    ImGui::SliderFloat("Crater inflow", &S.craterInflow, 0.02f, 1.f, "%.2f of the river");
    ImGui::SliderFloat("Rise with rain", &S.riverRainRise, 0.f, 2.f, "%.2f m at full");
    ImGui::SetItemTooltip("Rain swells the river over a minute or so (faster current, wider water); it drains back slowly after");
    ImGui::Text("River swell now: +%.2f m", riverSwell);
    ImGui::SetItemTooltip("How much of the river's flow pours into a crater that breaks into the channel (fills it over time)");
    ImGui::SliderFloat("Flow strength", &S.flow, 0.f, 3.f, "%.2fx");
    ImGui::SliderFloat("Clarity", &S.clarity, 0.1f, 4.f, "%.2fx");
    ImGui::SliderFloat("Absorption", &S.absorb, 0.05f, 3.f, "%.2f /m");
    ImGui::ColorEdit3("Murk colour", S.murk);
    ImGui::SliderFloat("Foam", &S.foam, 0.f, 2.f);
    ImGui::SliderFloat("Foam trail length", &S.foamTrail, 0.f, 40.f, "%.1f m");
    ImGui::SetItemTooltip("How far the white wake behind a rock reaches downstream");
    ImGui::SliderFloat("Bank splashes", &S.splash, 0.f, 2.f);
    ImGui::SliderFloat("Caustics on the bed", &S.caustics, 0.f, 3.f, "%.2f");
}

void panelFogGodRays() {
    if (!ImGui::CollapsingHeader("Fog & god rays")) return;
    ImGui::Checkbox("Volumetric fog", &S.fog);
    ImGui::BeginDisabled(!S.fog);
    {
        static const char* fogResNames[] = {"full", "1/2", "1/3", "1/4"};
        int fi = std::clamp(S.fogDiv, 1, 4) - 1;
        if (ImGui::SliderInt("Fog resolution", &fi, 0, 3, fogResNames[fi])) S.fogDiv = fi + 1;
    }
    ImGui::SliderFloat("Density (at the ground)", &S.fogDensity, 0.0005f, 0.1f, "%.4f /m", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Height falloff", &S.fogFalloff, 0.005f, 1.f, "%.3f /m", ImGuiSliderFlags_Logarithmic);
    ImGui::SetItemTooltip("Higher: the fog hugs the ground (valley mist). Lower: it fills the air (haze)");
    ImGui::SliderFloat("Forward scattering", &S.fogAniso, 0.f, 0.95f, "%.2f");
    ImGui::SetItemTooltip("How strongly the fog glows looking toward the sun (light shafts)");
    ImGui::SliderFloat("Patchiness", &S.fogNoise, 0.f, 1.f, "%.2f");
    ImGui::SliderFloat("Distance", &S.fogDist, 30.f, 400.f, "%.0f m");
    ImGui::Checkbox("Lamps glow in the fog", &S.fogLights);
    ImGui::SliderFloat("Temporal smoothing", &S.fogBlend, 0.03f, 1.f, "%.2f");
    ImGui::SetItemTooltip("Share of the new frame (lower: smoother, more lag)");
    ImGui::EndDisabled();
}

void panelWeather() {
    if (!ImGui::CollapsingHeader("Weather")) return;
    ImGui::Checkbox("Changing weather", &S.weatherCycle);
    ImGui::SetItemTooltip("Fronts pass every few minutes: clear, fair, cloudy, overcast, rain, thunderstorms, fog.\n"
                          "They set the clouds, rain, wind and fog below (and in the Clouds section)");
    if (S.weatherCycle) {
        ImGui::SameLine();
        ImGui::TextDisabled("now: %s", weatherName());
        ImGui::SliderFloat("Changes", &S.weatherChange, 0.2f, 5.f, "%.1fx as often", ImGuiSliderFlags_Logarithmic);
    }
    ImGui::SliderFloat("Precipitation", &S.rain, 0.f, 1.f, "%.2f");
    ImGui::SetItemTooltip("Rain below the snow line, snow above it (falls, settles and piles up there)");
    ImGui::SeparatorText("Snow, avalanches, melt water");
    ImGui::Checkbox("Simulate (2 km around you)", &S.worldSim);
    ImGui::Checkbox("Snow line follows the cloud base", &S.snowFollowsClouds);
    ImGui::BeginDisabled(S.snowFollowsClouds);
    ImGui::SliderFloat("Snow line", &S.snowLine, -20.f, 3000.f, "%.0f m");
    ImGui::EndDisabled();
    ImGui::SliderFloat("Snowfall", &S.snowfall, 0.f, 60.f, "%.1f m/h at full", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Melt rate", &S.meltRate, 0.f, 20.f, "%.2f m/h", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Angle of repose", &S.reposeDeg, 15.f, 60.f, "%.0f deg");
    ImGui::SetItemTooltip("Snow piled steeper than this slides. Real snow: ~30-45 deg (wet / cold powder)");
    ImGui::SliderFloat("Avalanche mobility", &S.avalanche, 0.1f, 4.f, "%.2f");
    ImGui::SliderFloat("River search range", &S.riverRange, 0.f, 1000.f, "%.0f m");
    ImGui::SetItemTooltip("Melt water within this distance of a river is pulled toward it (and drains into it);\n"
                          "further away it pools and soaks in, leaving wet ground");
    ImGui::SliderFloat("Ground drying", &S.dryRate, 0.0002f, 0.05f, "%.4f /s", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Sim speed", &S.simSpeed, 1.f, 100.f, "%.0fx", ImGuiSliderFlags_Logarithmic);
    ImGui::SetItemTooltip("Fast-forward snowfall, melt, runoff and drying (avalanches move faster too)");
    ImGui::SliderInt("Terrain refresh (frames)", &S.snowMeshEvery, 1, 30);
    if (ImGui::Button("Clear snow + water")) simClear = true;
    ImGui::SeparatorText("Ground under rocks");
    ImGui::SliderFloat("Soil friction", &S.groundFriction, 0.05f, 1.2f, "%.2f");
    ImGui::SliderFloat("Soil bounce", &S.groundBounce, 0.f, 0.8f, "%.2f");
    ImGui::SetItemTooltip("Natural ground. Painted / zone textures use their kind (stone and paving bounce, sand and mud\n"
                          "absorb, ice slides); craters are loose soil; wet ground is slicker; snow is slippery and dead.");
    ImGui::Checkbox("Rain wets the ground", &S.rainWets);
    ImGui::BeginDisabled(S.rainWets);
    ImGui::SliderFloat("Wetness", &S.wetness, 0.f, 1.f, "%.2f");
    ImGui::SliderFloat("Soaked in", &S.soak, 0.f, 1.f, "%.2f");
    ImGui::SliderFloat("Dripping", &S.drip, 0.f, 1.f, "%.2f");
    ImGui::EndDisabled();
    if (S.rainWets) ImGui::Text("Surface film %.2f, soaked in %.2f, dripping %.2f", S.wetness, S.soak, S.drip);
    ImGui::SetItemTooltip("Film: glossy water on every surface, runs off within minutes. Soaked: porous surfaces (soil, rock,\n"
                          "brick, wood, concrete) darken, then dry patchily over 10-20 min (faster in sun and wind).\n"
                          "Dripping: trees and eaves keep dripping for a minute after the rain.");
    ImGui::SliderFloat("Rivulets on walls", &S.rivulets, 0.f, 1.f, "%.2f");
    ImGui::SliderFloat("Puddles", &S.puddles, 0.f, 1.f, "%.2f");
}

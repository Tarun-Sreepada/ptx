// Main panel sections: Baked interiors, world streaming, audio, controller.
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
#include "ui/world_map.h"

void panelBakedInteriors() {
    if (!ui::CollapsingHeader("Baked interiors")) return;
    ui::Checkbox("Use baked indoor GI", &bakedGI);
    ui::SameLine();
    ui::Checkbox("Bake", &bakeEnabled);
    ui::SameLine();
    if (ui::Button("Rebake all")) {
        for (auto& b : bakeVols) for (int st = 0; st < 2; st++) { b.ready[st] = b.saved[st] = false; b.pass[st] = b.next[st] = 0; }
    }
    bool allOn = true;
    for (auto& b : bakeVols) allOn &= b.on;
    if (ui::Checkbox("Indoor lights (L, or E while walking, toggles the building you're in)", &allOn))
        for (auto& b : bakeVols) b.on = allOn;
    int here = volumeAt(cam.pos.x, cam.pos.y, cam.pos.z);
    for (size_t i = 0; i < bakeVols.size(); i++) {
        auto& b = bakeVols[i];
        auto prog = [&](int st) { return b.ready[st] ? 100.f : std::min(99.f, 100.f * (b.pass[st] * b.count + b.next[st]) / (float)(BAKE_PASSES * b.count)); };
        ui::Text("%s%s: %ux%ux%u probes @ %.0f cm, lights %s   bake off %.0f%%  on %.0f%%", (int)i == here ? "> " : "  ", b.v.name.c_str(),
                    b.dims[0], b.dims[1], b.dims[2], b.spacing * 100.f, b.on ? "ON" : "off", prog(0), prog(1));
    }

}

// Biomes, oceans, lakes (biome.h gBiome). True when the terrain must be rebuilt (on slider release).
static bool panelBiomes() {
    BiomeParams& B = gBiome;
    bool ch = false;
    auto done = [&] { ch |= ImGui::IsItemDeactivatedAfterEdit(); };
    auto sl = [&](const char* n, float* v, float a, float b, const char* f, const char* tip = nullptr, ImGuiSliderFlags fl = 0) {
        ui::SliderFloat(n, v, a, b, f, fl);
        done();
        if (tip) ui::SetItemTooltip("%s", tip);
    };
    if (!ImGui::TreeNodeEx("Biomes, oceans, lakes", ImGuiTreeNodeFlags_DefaultOpen)) return false;
    if (ui::Checkbox("Biomes on", &B.on)) ch = true;
    ui::SetItemTooltip("Off: the original temperate world everywhere");
    ImGui::BeginDisabled(!B.on);
    sl("Biome size", &B.size, 400.f, 8000.f, "%.0f m", "Distance between biome cells", ImGuiSliderFlags_Logarithmic);
    sl("Blend", &B.blend, 0.02f, 0.6f, "%.2f", "How wide the transition between neighbouring biomes is");
    sl("Untouched spawn radius", &B.spawn, 0.f, 3000.f, "%.0f m", "The town and river around the origin stay temperate");
    sl("Temperature", &B.temp, -0.6f, 0.6f, "%+.2f", "Shift the whole climate colder (snow, tundra, taiga) or hotter (desert, jungle)");
    sl("Moisture", &B.moist, -0.6f, 0.6f, "%+.2f", "Drier (desert, salt, mesa) or wetter (jungle, marsh)");
    sl("Relief", &B.relief, 0.f, 2.f, "%.2fx", "Scales every biome's landforms");
    if (ImGui::TreeNode("Biome amounts")) {
        for (int b = 0; b < B_COUNT; b++) sl(BIOME_NAMES[b], &B.amount[b], 0.f, 3.f, "%.2f");
        if (ui::Button("All 1")) { for (float& a : B.amount) a = 1.f; ch = true; }
        ui::SameLine();
        if (ui::Button("Only temperate")) { for (float& a : B.amount) a = 0.f; B.amount[B_TEMPERATE] = 1.f; ch = true; }
        ImGui::TreePop();
    }
    if (ImGui::TreeNode("Ocean and coast")) {
        sl("Ocean amount", &B.ocean, 0.f, 1.f, "%.2f", "0 = no sea; more = smaller continents");
        sl("Continent size", &B.oceanScale, 2000.f, 40000.f, "%.0f m", nullptr, ImGuiSliderFlags_Logarithmic);
        sl("Sea level", &B.seaLevel, -40.f, 10.f, "%.1f m");
        sl("Ocean depth", &B.oceanDepth, 5.f, 300.f, "%.0f m");
        sl("Beach width", &B.beach, 5.f, 300.f, "%.0f m");
        ImGui::TreePop();
    }
    if (ImGui::TreeNode("Lakes and ponds")) {
        sl("Lakes", &B.lakes, 0.f, 1.f, "%.2f", "Chance of a lake per 1.2 km cell (wet biomes hold more)");
        sl("Lake size", &B.lakeSize, 0.2f, 1.6f, "%.2fx");
        sl("Ponds", &B.ponds, 0.f, 1.f, "%.2f", "Chance of a pond per 200 m cell");
        ImGui::TreePop();
    }
    if (ImGui::TreeNode("Landforms")) {
        sl("Dune height", &B.duneH, 0.f, 60.f, "%.1f m");
        sl("Dune spacing", &B.duneL, 20.f, 400.f, "%.0f m");
        sl("Wind direction", &B.wind, 0.f, 6.2832f, "%.2f rad", "Dunes lie across the wind, slip faces downwind");
        sl("Mesa height", &B.mesaH, 0.f, 300.f, "%.0f m");
        sl("Terrace step", &B.terrace, 0.f, 40.f, "%.1f m", "Rock layer height in mesas / canyon walls (0 = smooth)");
        sl("Canyon depth", &B.canyonD, 0.f, 300.f, "%.0f m");
        sl("Mountain height", &B.mountainH, 0.f, 1200.f, "%.0f m", "Alpine ridges");
        sl("Glacier height", &B.glacierH, 0.f, 800.f, "%.0f m");
        sl("Volcano height", &B.volcanoH, 0.f, 1000.f, "%.0f m");
        sl("Karst tower height", &B.karstH, 0.f, 200.f, "%.0f m");
        ImGui::TreePop();
    }
    ImGui::EndDisabled();
    ImGui::TreePop();
    return ch;
}

void panelWorldGen() {
    if (!ui::CollapsingHeader("World generation", ImGuiTreeNodeFlags_DefaultOpen)) return;
    // the seed drives the ground, hills, rivers, trees (placement + shapes), plants, rocks, grass and dirt
    bool regen = false;
    if (ui::InputInt("World seed", &S.worldSeed)) { S.worldSeed = std::max(S.worldSeed, 0); regen = true; }
    ui::SameLine();
    if (ui::Button("Random")) { S.worldSeed = (int)(((uint32_t)rand() << 8 ^ (uint32_t)rand()) & 0x7FFFFFFF); regen = true; }
    ui::SetItemTooltip("0 = the original world. Regrows the trees (a moment's pause)");
    if (ui::SliderFloat("Hilliness", &S.hilliness, 0.f, 1.f, "%.2f")) { applyWorldGen(); terrainDirty = placeDirty = true; stream::terrainChanged(); vtVersion++; }
    ui::SetItemTooltip("Rolling hills (up to 45 m) rising away from the river; 0 = gentle flats");
    if (ui::Button("World map (M)")) worldmap_ui::isOpen() = !worldmap_ui::isOpen();
    ui::SameLine();
    static float tp[2] = {0.f, 0.f};
    ImGui::SetNextItemWidth(160);
    ImGui::InputFloat2("##tp", tp, "%.0f");
    ui::SameLine();
    if (ui::Button("Teleport")) worldmap_ui::teleport(tp[0], tp[1]);
    if (panelBiomes()) { worldmap_ui::invalidate(); terrainDirty = placeDirty = true; stream::terrainChanged(); vtVersion++; }
    if (regen) {
        worldmap_ui::invalidate();
        applyWorldGen();
        terrainDirty = placeDirty = vegDirty = true;
        stream::terrainChanged();
        vtVersion++;  // ground textures follow the seed
    }
    ui::TextDisabled("River width, braids and tributaries: River & water");
}

void panelWorldStreaming() {
    if (!ui::CollapsingHeader("World streaming")) return;
    ui::Checkbox("Wildlife (foxes)", &S.wildlife);
    ui::SameLine();
    ui::Checkbox("Birds, butterflies, fireflies", &S.critters);
    auto ss = stream::stats();
    ui::Text("Cells: %d on disk, %d loaded, %d active   models %d (%d loading)", ss.cellsKnown, ss.cellsLoaded, ss.cellsActive, ss.models, ss.loading);
    ui::Text("Instances %u (raster %u visible, %zu draws)  meshes %u  textures %u", numModelInst, numModelVisible, drawRuns.size(),
                meshSlots.used(), texSlots.used() - 1);
    ui::Text("Model GPU memory %.1f MB   physics bodies %u   terrain tiles %d", gpuModelBytes / 1048576.0, physics::bodyCount(), ss.terrainTiles);
    ui::Text("Lights %u / %d  (%u streamed of %zu loaded)   zones %zu", numLights, MAX_LIGHTS, numWorldLights, stream::lights().size(),
                stream::zones().size());
    if (ui::SliderFloat("View distance", &S.viewDist, 300.f, 6000.f, "%.0f m", ImGuiSliderFlags_Logarithmic)) {
        loadRadius = std::clamp((int)ceilf(S.viewDist / 64.f), 1, 96);  // streamed cells follow
        terrainDirty = true;
    }
    ui::SetItemTooltip("How far the world is drawn: terrain, trees, rocks and streamed cells.\n"
                          "Larger costs GPU time (more trees in the BVH, coarser far terrain) and memory.");
    ui::SliderInt("Load radius (cells)", &loadRadius, 1, 96);
    ui::SliderFloat("Upload budget", &uploadBudgetMB, 8, 512, "%.0f MB/frame", ImGuiSliderFlags_Logarithmic);
}

void panelAudioRayTraced() {
    if (!ui::CollapsingHeader("Audio (ray traced)")) return;
    ui::Checkbox("Ambience (birdsong, crickets, river, leaves)", &S.ambience);
    auto as = audio::stats();
    audio::Settings& A = audio::settings();
    if (as.running) {
        ui::Text("Emitters %u   scene %u meshes / %u inst / %u tris", as.emitters, as.meshes, as.instances, as.sceneTris);
        ui::Text("Sim %.1f ms/update (%.0f Hz)   DSP %.2f ms / 21.3 ms block", as.simMs, as.simMs > 0 ? 1000.f / std::max(as.simMs, 16.f) : 0.f, as.dspMs);
        ui::SliderFloat("Volume", &A.masterGain, 0, 2);
        ui::Checkbox("Occlusion", &A.occlusion);
        ui::SameLine(); ui::Checkbox("Transmission", &A.transmission);
        ui::SameLine(); ui::Checkbox("Reflections", &A.reflections);
        ImGui::BeginDisabled(!A.reflections);
        ui::SliderInt("Reflection rays", &A.reflectionRays, 256, 4096);
        ui::SliderInt("Reflection bounces", &A.bounces, 1, 16);
        ui::SliderFloat("Reflection updates", &A.reflectionHz, 2, 60, "%.0f Hz");
        ui::SliderFloat("Reverb gain", &A.reverbGain, 0, 3);
        ImGui::EndDisabled();
    } else {
        ui::TextDisabled("No audio device");
    }

}

void panelController() {
    if (!ui::CollapsingHeader("Controller")) return;
    const input::Pad& gp = input::pad();
    if (gp.connected) {
        ui::Text("%s%s", gp.name, gp.hasGyro ? "  (gyro)" : "");
        ImGui::BeginDisabled(!gp.hasGyro);
        ui::Checkbox("Gyro aim", &gyroAim);
        ui::SameLine();
        ui::SliderFloat("##gyro", &gyroSens, 0.2f, 4.f, "sens %.1fx");
        ImGui::EndDisabled();
        ui::TextDisabled("L: move  R: look  A: jump  Y: walk/fly  LT/L3: sprint  LB/RB: down/up");
    } else {
        ui::TextDisabled("No gamepad (Steam Controller, Xbox, DualSense... plug in any time)");
    }
}

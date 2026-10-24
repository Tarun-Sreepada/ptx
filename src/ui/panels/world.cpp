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

void panelBakedInteriors() {
    if (!ImGui::CollapsingHeader("Baked interiors")) return;
    ImGui::Checkbox("Use baked indoor GI", &bakedGI);
    ImGui::SameLine();
    ImGui::Checkbox("Bake", &bakeEnabled);
    ImGui::SameLine();
    if (ImGui::Button("Rebake all")) {
        for (auto& b : bakeVols) for (int st = 0; st < 2; st++) { b.ready[st] = b.saved[st] = false; b.pass[st] = b.next[st] = 0; }
    }
    bool allOn = true;
    for (auto& b : bakeVols) allOn &= b.on;
    if (ImGui::Checkbox("Indoor lights (L, or E while walking, toggles the building you're in)", &allOn))
        for (auto& b : bakeVols) b.on = allOn;
    int here = volumeAt(cam.pos.x, cam.pos.y, cam.pos.z);
    for (size_t i = 0; i < bakeVols.size(); i++) {
        auto& b = bakeVols[i];
        auto prog = [&](int st) { return b.ready[st] ? 100.f : std::min(99.f, 100.f * (b.pass[st] * b.count + b.next[st]) / (float)(BAKE_PASSES * b.count)); };
        ImGui::Text("%s%s: %ux%ux%u probes @ %.0f cm, lights %s   bake off %.0f%%  on %.0f%%", (int)i == here ? "> " : "  ", b.v.name.c_str(),
                    b.dims[0], b.dims[1], b.dims[2], b.spacing * 100.f, b.on ? "ON" : "off", prog(0), prog(1));
    }

}

void panelWorldStreaming() {
    if (!ImGui::CollapsingHeader("World streaming")) return;
    ImGui::Checkbox("Wildlife (foxes)", &S.wildlife);
    ImGui::SameLine();
    ImGui::Checkbox("Birds, butterflies, fireflies", &S.critters);
    auto ss = stream::stats();
    ImGui::Text("Cells: %d on disk, %d loaded, %d active   models %d (%d loading)", ss.cellsKnown, ss.cellsLoaded, ss.cellsActive, ss.models, ss.loading);
    ImGui::Text("Instances %u (raster %u visible, %zu draws)  meshes %u  textures %u", numModelInst, numModelVisible, drawRuns.size(),
                meshSlots.used(), texSlots.used() - 1);
    ImGui::Text("Model GPU memory %.1f MB   physics bodies %u   terrain tiles %d", gpuModelBytes / 1048576.0, physics::bodyCount(), ss.terrainTiles);
    ImGui::Text("Lights %u / %d  (%u streamed of %zu loaded)   zones %zu", numLights, MAX_LIGHTS, numWorldLights, stream::lights().size(),
                stream::zones().size());
    if (ImGui::SliderFloat("View distance", &S.viewDist, 300.f, 6000.f, "%.0f m", ImGuiSliderFlags_Logarithmic)) {
        loadRadius = std::clamp((int)ceilf(S.viewDist / 64.f), 1, 96);  // streamed cells follow
        terrainDirty = true;
    }
    ImGui::SetItemTooltip("How far the world is drawn: terrain, trees, rocks and streamed cells.\n"
                          "Larger costs GPU time (more trees in the BVH, coarser far terrain) and memory.");
    ImGui::SliderInt("Load radius (cells)", &loadRadius, 1, 96);
    ImGui::SliderFloat("Upload budget", &uploadBudgetMB, 8, 512, "%.0f MB/frame", ImGuiSliderFlags_Logarithmic);
}

void panelAudioRayTraced() {
    if (!ImGui::CollapsingHeader("Audio (ray traced)")) return;
    ImGui::Checkbox("Ambience (birdsong, crickets, river, leaves)", &S.ambience);
    auto as = audio::stats();
    audio::Settings& A = audio::settings();
    if (as.running) {
        ImGui::Text("Emitters %u   scene %u meshes / %u inst / %u tris", as.emitters, as.meshes, as.instances, as.sceneTris);
        ImGui::Text("Sim %.1f ms/update (%.0f Hz)   DSP %.2f ms / 21.3 ms block", as.simMs, as.simMs > 0 ? 1000.f / std::max(as.simMs, 16.f) : 0.f, as.dspMs);
        ImGui::SliderFloat("Volume", &A.masterGain, 0, 2);
        ImGui::Checkbox("Occlusion", &A.occlusion);
        ImGui::SameLine(); ImGui::Checkbox("Transmission", &A.transmission);
        ImGui::SameLine(); ImGui::Checkbox("Reflections", &A.reflections);
        ImGui::BeginDisabled(!A.reflections);
        ImGui::SliderInt("Reflection rays", &A.reflectionRays, 256, 4096);
        ImGui::SliderInt("Reflection bounces", &A.bounces, 1, 16);
        ImGui::SliderFloat("Reflection updates", &A.reflectionHz, 2, 60, "%.0f Hz");
        ImGui::SliderFloat("Reverb gain", &A.reverbGain, 0, 3);
        ImGui::EndDisabled();
    } else {
        ImGui::TextDisabled("No audio device");
    }

}

void panelController() {
    if (!ImGui::CollapsingHeader("Controller")) return;
    const input::Pad& gp = input::pad();
    if (gp.connected) {
        ImGui::Text("%s%s", gp.name, gp.hasGyro ? "  (gyro)" : "");
        ImGui::BeginDisabled(!gp.hasGyro);
        ImGui::Checkbox("Gyro aim", &gyroAim);
        ImGui::SameLine();
        ImGui::SliderFloat("##gyro", &gyroSens, 0.2f, 4.f, "sens %.1fx");
        ImGui::EndDisabled();
        ImGui::TextDisabled("L: move  R: look  A: jump  Y: walk/fly  LT/L3: sprint  LB/RB: down/up");
    } else {
        ImGui::TextDisabled("No gamepad (Steam Controller, Xbox, DualSense... plug in any time)");
    }
}

// Main panel sections: Performance stats (GPU pass timeline), the benchmark, the frame rate target.
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

void panelPerformanceStats() {
    if (!ImGui::CollapsingHeader("Performance stats")) return;
    if (ImGui::TreeNode("GPU passes (render graph)")) {
        // timeline: one row per pass, bar = [start, start + ms] within the frame; async compute in orange
        const auto& tl = graph.timings();
        float total = std::max(graph.frameMs(), 0.01f), width = ImGui::GetContentRegionAvail().x - 190.f;
        ImDrawList* dl = ImGui::GetWindowDrawList();
        for (auto& t : tl) {
            ImGui::Text("%-20s %5.2f", t.name.c_str(), t.ms);
            ImGui::SameLine(190.f);
            ImVec2 p = ImGui::GetCursorScreenPos();
            float x0 = p.x + width * t.startMs / total, x1 = std::max(x0 + 1.f, p.x + width * (t.startMs + t.ms) / total);
            dl->AddRectFilled({x0, p.y + 2}, {x1, p.y + ImGui::GetTextLineHeight() - 2},
                              t.queue == rg::COMPUTE ? IM_COL32(240, 150, 40, 255) : IM_COL32(80, 160, 240, 255));
            ImGui::NewLine();
        }
        ImGui::BeginDisabled(!graph.asyncCompute());
        ImGui::Checkbox("Async compute (BVH build overlaps raster)", &asyncCompute);
        ImGui::BeginDisabled(!asyncCompute);
        ImGui::Checkbox("Async fog (overlaps ReSTIR DI + GI)", &asyncFog);
        ImGui::EndDisabled();
        ImGui::EndDisabled();
        ImGui::TreePop();
    }
    int rpp = raysPerPixel(S.raysFirst, S.bounces);
    ImGui::Text("~%d rays/pixel, %.1f M rays/frame, %.2f Grays/s", rpp, rpp * W * H / 1e6f,
                gpuEmaTotal > 0 ? rpp * W * H / (gpuEmaTotal * 1e6f) : 0.f);
    ImGui::Text("Blades: %u generated (BLAS slots %u, max %u)", lastBladeCount, bladeCap, maxBlades);
    ImGui::Text("Tris: terrain %u, grass %u, rocks %u x %u (16 shapes)", terrainTris, std::min(lastBladeCount, bladeCap) * 5, numRocks, rockTris);
    ImGui::Text("BVH alloc: terrain %.1f MB, grass %.1f MB (max), rock %.2f MB", terrainAS.size / 1048576.0, grassAS.size / 1048576.0,
                rockAS[0].size * ROCK_VARIANTS / 1048576.0);

}

void panelBenchmark() {
    if (!ImGui::CollapsingHeader("Benchmark")) return;
    if (bench.running) {
        float prog = ((bench.tier - 1) * (Bench::WARM + Bench::MEAS) + bench.frame) / float(4 * (Bench::WARM + Bench::MEAS));
        ImGui::ProgressBar(prog, {-1, 0}, "running...");
        ImGui::Text("Testing %d ray(s) per first bounce", bench.tier);
    } else if (ImGui::Button("Run benchmark (~20 s)", {-1, 0})) {
        startBenchmark();
    }
    if (bench.done) {
        if (ImGui::BeginTable("bench", 4, ImGuiTableFlags_Borders | ImGuiTableFlags_RowBg)) {
            ImGui::TableSetupColumn("Rays/bounce");
            ImGui::TableSetupColumn("avg ms");
            ImGui::TableSetupColumn("p95 ms");
            ImGui::TableSetupColumn("fps (p95)");
            ImGui::TableHeadersRow();
            for (int t = 1; t <= 4; t++) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn(); ImGui::Text("%d  (%d rpp)", t, raysPerPixel(t, 2));
                ImGui::TableNextColumn(); ImGui::Text("%.2f", bench.avg[t]);
                ImGui::TableNextColumn(); ImGui::Text("%.2f", bench.p95[t]);
                ImGui::TableNextColumn(); ImGui::Text("%.0f", bench.p95[t] > 0 ? 1000.f / bench.p95[t] : 0.f);
            }
            ImGui::EndTable();
        }
        ImGui::TextWrapped("What your system can do at %ux%u:", W, H);
        for (int fps : {60, 40, 30, 25}) {
            int t = benchBestTier(fps);
            if (t) ImGui::BulletText("%d fps -> %d ray(s) per bounce", fps, t);
            else ImGui::BulletText("%d fps -> not reachable (lower resolution)", fps);
        }
    } else if (!bench.running) {
        ImGui::TextDisabled("Run the benchmark to see what fps / quality you can get.");
    }

}

void panelFrameRateTarget() {
    if (!ImGui::CollapsingHeader("Frame rate target")) return;
    ImGui::BeginDisabled(bench.running);
    const int opts[] = {0, 60, 40, 30, 25};
    for (int i = 0; i < 5; i++) {
        if (i) ImGui::SameLine();
        char label[16];
        snprintf(label, sizeof label, i == 0 ? "Manual" : "%d", opts[i]);
        if (ImGui::RadioButton(label, targetFps == opts[i])) {
            targetFps = opts[i];
            if (targetFps && bench.done) { int t = benchBestTier(targetFps); if (t) S.raysFirst = t; }
        }
    }
    if (ImGui::Checkbox("VSync", &vsync)) swapDirty = true;
    if (targetFps) {
        ImGui::Text("Budget %.1f ms -> auto: %d ray(s) per bounce", 1000.f / targetFps, S.raysFirst);
        if (S.raysFirst == 1 && gpuEmaTotal > 1000.f / targetFps) ImGui::TextColored({1, 0.4f, 0.3f, 1}, "Can't hold target even at 1 ray: lower resolution");
    } else {
        ImGui::SliderInt("Rays per first bounce", &S.raysFirst, 1, 4);
    }
    ImGui::EndDisabled();

}

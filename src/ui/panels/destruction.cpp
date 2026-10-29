// Main panel sections: Impact marks, throwing rocks & destruction, characters.
#include "ui/panels/panels.h"
#include "sim/fire.h"

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

void panelImpactMarksDust() {
    if (!ui::CollapsingHeader("Impact marks & dust")) return;
    ui::Checkbox("Decals (cracks, soil splatter, wood scars)", &S.decals);
    ui::Text("%zu decals, %zu dust clouds", decalList.size(), dustList.size());
    if (ui::Button("Clear decals")) decalList.clear();
}

void panelThrowRocksDestruction() {
    if (!ui::CollapsingHeader("Throw rocks & destruction")) return;
    ui::TextDisabled("G / left click / right trigger");
    ui::SliderFloat("Rock size", &S.throwSize, 0.05f, 2.f, "%.2f m", ImGuiSliderFlags_Logarithmic);
    ui::SliderFloat("Throw speed", &S.throwSpeed, 0.f, 300.f, "%.0f m/s", ImGuiSliderFlags_Logarithmic);
    {
        float m = 2600.f * 2.6f * S.throwSize * S.throwSize * S.throwSize;  // rock hull ~2.6 m^3 at scale 1
        ui::Text("~%.0f kg, %.1f kJ at release", m, 0.5f * m * S.throwSpeed * S.throwSpeed / 1000.f);
    }
    ui::Text("%zu thrown", thrown.size());
    if (ui::Button("Clear thrown rocks")) while (!thrown.empty()) dropThrown(thrown.size() - 1);
    ui::SeparatorText("Fire");
    static float fireSize = 0.6f, fireBurn = 90.f;
    ui::SliderFloat("Fire size", &fireSize, 0.2f, 4.f, "%.1f m", ImGuiSliderFlags_Logarithmic);
    ui::SliderFloat("Burns for", &fireBurn, 10.f, 600.f, "%.0f s", ImGuiSliderFlags_Logarithmic);
    if (ui::Button("Light a fire where I look")) fireInView(fireSize, fireBurn);
    ui::SameLine();
    if (ui::Button("Put out fires")) putOutFires();
    ui::Text("%zu fires", fires.size());
    ui::SetItemTooltip("Lightning near trees and very hard impacts start fires too; embers can spread them in dry weather, rain puts them out");
    ui::Text("%d craters, %zu trees down, %zu debris", std::min(gCraterCount.load(), MAX_CRATERS), brokenTrees.size(), spray.size());
    if (ui::Button("Repair world")) {  // craters filled, trees stood back up
        gCraterCount.store(0);
        while (!stumps.empty()) dropStump(stumps.size() - 1);
        std::vector<std::shared_ptr<Model>> freed;
        stream::clearOverrides(freed);
        for (auto& m : freed) {
            damagedModels.erase(std::remove(damagedModels.begin(), damagedModels.end(), m), damagedModels.end());
            retireNext.push_back([m]() {
                if (m->gpu) releaseModel(*m);
                if (m->data.meshes[0].shape) physics::releaseShape(m->data.meshes[0].shape);
            });
        }
        onStreamChanged();
        brokenTrees.clear();
        damagedTrees.clear();
        terrainDirty = placeDirty = true;
        stream::terrainChanged();
    }
}

void panelCharacters() {
    if (!ui::CollapsingHeader("Characters")) return;
    static char asset[128] = "characters/CesiumMan";
    ui::InputText("Asset", asset, sizeof asset);
    ui::SetItemTooltip("A skinned glTF under world/assets (characters/Fox, characters/CesiumMan, ...)");
    if (ui::Button("Spawn in front of me")) {
        V3 f = cam.fwd();
        float l = std::max(sqrtf(f.x * f.x + f.z * f.z), 1e-4f);
        charSpawns.push_back({asset, "", {cam.pos.x + f.x / l * 4.f, 0.f, cam.pos.z + f.z / l * 4.f}, atan2f(-f.x, -f.z), 0.f, 0.f});
    }
    std::vector<character::Character*> gone;
    int n = 0;
    for (character::Character* c : character::all()) {
        ImGui::PushID(c);
        char title[160];
        snprintf(title, sizeof title, "#%d %s", n++, ((GpuChar*)c->user)->asset->model->path.substr(charWorldDir.size()).c_str());
        ui::SeparatorText(title);
        const ModelData& d = *c->data;
        int cur = c->next >= 0 ? c->next : c->clip;
        if (ui::BeginCombo("Clip", cur >= 0 ? d.clips[cur].name.c_str() : "(rest pose)")) {
            for (int i = 0; i < (int)d.clips.size(); i++)
                if (ImGui::Selectable(d.clips[i].name.c_str(), i == cur)) character::play(*c, i, 0.3f);
            ImGui::EndCombo();
        }
        ui::SliderFloat("Speed", &c->rate, 0.f, 3.f, "%.2fx");
        ui::SliderFloat("Walk", &c->walkSpeed, 0.f, 6.f, "%.1f m/s");
        ui::SetItemTooltip("Wander forward over the terrain (match it to the clip's stride)");
        ui::SliderFloat("Get up after", &c->getUpAfter, 0.f, 20.f, c->getUpAfter > 0 ? "%.1f s at rest" : "never");
        if (c->mode == character::ANIMATED) { if (ui::Button("Ragdoll")) { float j[3] = {0, 0, 0}; character::ragdoll(*c, j, c->pos); } }
        else if (ui::Button("Get up")) character::getUp(*c);
        ui::SameLine();
        if (ui::Button("Remove")) gone.push_back(c);
        ui::TextDisabled("%zu joints, %zu ragdoll bodies, %.0f kg", c->globals.size(), c->parts.size(), c->mass);
        ImGui::PopID();
    }
    for (auto* c : gone) releaseCharacter(c);
}

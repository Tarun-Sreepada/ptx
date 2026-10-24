#include "ui/vegetation_panel.h"
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <imgui.h>
#include "vegetation/trees.h"
#include "scene/vegetation.h"

// Tree / bush generator sliders (trees.h). Changes regrow the meshes when a slider is let go (regenVegetation).
void drawVegetationUI() {
    auto R = [] { if (ImGui::IsItemDeactivatedAfterEdit()) vegDirty = true; };
    auto C = [&](const char* l, float* c) { ImGui::ColorEdit3(l, c, ImGuiColorEditFlags_Float); R(); };
    if (ImGui::TreeNode("Tree generator")) {
        for (int sp = 0; sp < trees::SPECIES; sp++) {
            trees::TreeParams& P = trees::treeParams[sp];
            ImGui::PushID(sp);
            if (ImGui::TreeNode(trees::SPECIES_NAMES[sp])) {
                static int preset[trees::SPECIES] = {};
                static bool presetInit = false;
                if (!presetInit) { for (int s = 0; s < trees::SPECIES; s++) preset[s] = trees::DEFAULT_PRESET[s]; presetInit = true; }
                if (ImGui::Combo("Preset", &preset[sp], trees::TREE_PRESETS, trees::TREE_PRESET_COUNT)) {
                    uint32_t seed = P.seed;
                    trees::treePreset(preset[sp], P);
                    P.seed = seed;
                    vegDirty = true;
                }
                int seed = (int)P.seed;
                if (ImGui::InputInt("Seed", &seed)) { P.seed = (uint32_t)std::max(seed, 0); vegDirty = true; }
                ImGui::SeparatorText("Trunk");
                ImGui::SliderFloat("Height", &P.height, 2.f, 30.f, "%.1f m"); R();
                ImGui::SliderFloat("Trunk radius", &P.trunkRadius, 0.05f, 1.f, "%.2f m"); R();
                ImGui::SliderFloat("Clear trunk", &P.bole, 0.05f, 0.8f, "%.2f of height"); R();
                ImGui::SliderFloat("Lean", &P.lean, 0.f, 0.5f); R();
                ImGui::SliderFloat("Gnarl", &P.gnarl, 0.f, 1.f); R();
                ImGui::SliderFloat("Root flare", &P.flare, 0.f, 2.f); R();
                ImGui::SliderFloat("Trunk lumpiness", &P.trunkWave, 0.f, 1.5f); R();
                ImGui::SetItemTooltip("Irregular, fluted cross-section and a wandering trunk line (big limbs too)");
                ImGui::SliderFloat("Wood density", &P.density, 300.f, 1200.f, "%.0f kg/m3");
                ImGui::SliderFloat("Wood strength", &P.strength, 10.f, 100.f, "%.0f MPa");
                ImGui::SetItemTooltip("Bending strength of the green wood: what a limb takes before it snaps");
                ImGui::SeparatorText("Branches");
                ImGui::Combo("Growth", &P.growth, trees::GROWTH_NAMES, trees::GROWTH_COUNT); R();
                ImGui::SetItemTooltip("Branching rules: regular, tidy (conifers). Space colonization: branches compete for\n"
                                      "the space inside the crown shape, the irregular look of a grown oak or maple");
                ImGui::Combo("Crown shape", &P.crown, trees::CROWN_NAMES, trees::CROWN_COUNT); R();
                ImGui::SliderInt("Levels", &P.levels, 1, 4); R();
                ImGui::SliderInt("Main branches", &P.branches, 2, 50); R();
                ImGui::SliderInt("Sub-branches", &P.children, 1, 12); R();
                ImGui::SliderFloat("Branch angle", &P.branchAngle, 5.f, 110.f, "%.0f deg"); R();
                ImGui::SliderFloat("Angle variation", &P.angleVar, 0.f, 40.f, "%.0f deg"); R();
                ImGui::SliderFloat("Branch length", &P.lengthRatio, 0.1f, 1.2f); R();
                ImGui::SliderFloat("Sub-branch length", &P.childLength, 0.1f, 1.2f); R();
                ImGui::SliderFloat("Branch thickness", &P.radiusRatio, 0.2f, 1.f); R();
                ImGui::SliderFloat("Droop (gravity)", &P.gravity, 0.f, 2.f); R();
                ImGui::SliderFloat("Reach for light", &P.upturn, 0.f, 1.f); R();
                ImGui::SeparatorText("Leaves");
                ImGui::SliderFloat("Leaf size", &P.leafSize, 0.02f, 0.5f, "%.2f m"); R();
                ImGui::SliderFloat("Leaf width", &P.leafWidth, 0.05f, 1.2f); R();
                ImGui::SetItemTooltip("Low values give needles");
                ImGui::SliderFloat("Leaf density", &P.leafDensity, 0.f, 4.f); R();
                ImGui::SliderFloat("Cluster spread", &P.leafSpread, 0.05f, 1.f, "%.2f m"); R();
                ImGui::SliderFloat("Leaf curl", &P.leafDroop, 0.f, 1.f); R();
                C("Leaf colour", P.leafColor);
                ImGui::SliderFloat("Colour variation", &P.hueVar, 0.f, 1.f); R();
                C("Bark colour", P.barkColor);
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        ImGui::TreePop();
    }
    if (ImGui::TreeNode("Bush generator")) {
        const char* names[trees::BUSH_KINDS] = {"Bush", "Berry bush", "Rose", "Flowering bush"};
        const char* fruit[trees::BUSH_KINDS] = {"Berries", "Berries", "Flowers", "Blossoms"};
        for (int k = 0; k < trees::BUSH_KINDS; k++) {
            trees::BushParams& B = trees::bushParams[k];
            ImGui::PushID(100 + k);
            if (ImGui::TreeNode(names[k])) {
                int seed = (int)B.seed;
                if (ImGui::InputInt("Seed", &seed)) { B.seed = (uint32_t)std::max(seed, 0); vegDirty = true; }
                ImGui::SliderFloat("Size", &B.size, 0.2f, 3.f, "%.2f m"); R();
                ImGui::SliderInt("Stems", &B.stems, 1, 16); R();
                ImGui::SliderFloat("Spread", &B.spread, 0.f, 1.5f); R();
                ImGui::SliderInt("Branch levels", &B.levels, 1, 3); R();
                ImGui::SliderFloat("Branch angle", &B.branchAngle, 10.f, 90.f, "%.0f deg"); R();
                ImGui::SliderFloat("Gnarl", &B.gnarl, 0.f, 1.f); R();
                ImGui::SliderFloat("Leaf size", &B.leafSize, 0.01f, 0.2f, "%.3f m"); R();
                ImGui::SliderFloat("Leaf density", &B.leafDensity, 0.2f, 4.f); R();
                C("Leaf colour", B.leafColor);
                char l[64];
                snprintf(l, sizeof l, "%s amount", fruit[k]);
                ImGui::SliderFloat(l, &B.fruitDensity, 0.f, 2.5f); R();
                snprintf(l, sizeof l, "%s size", fruit[k]);
                ImGui::SliderFloat(l, &B.fruitSize, 0.003f, 0.06f, "%.3f m"); R();
                snprintf(l, sizeof l, "%s colour", fruit[k]);
                C(l, B.fruitColor);
                ImGui::TreePop();
            }
            ImGui::PopID();
        }
        ImGui::TreePop();
    }
    if (ImGui::Button("Regenerate vegetation")) vegDirty = true;
}

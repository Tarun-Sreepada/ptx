#include "ui/vegetation_panel.h"
#include <algorithm>
#include <cstdio>
#include <cstdint>
#include <imgui.h>
#include "ui/search.h"
#include "vegetation/trees.h"
#include "scene/vegetation.h"

// Tree / bush generator sliders (trees.h). Changes regrow the meshes when a slider is let go (regenVegetation).
void drawVegetationUI() {
    auto R = [] { if (ui::IsItemDeactivatedAfterEdit()) vegDirty = true; };
    auto C = [&](const char* l, float* c) { ui::ColorEdit3(l, c, ImGuiColorEditFlags_Float); R(); };
    if (ui::TreeNode("Tree generator")) {
        for (int sp = 0; sp < trees::SPECIES; sp++) {
            trees::TreeParams& P = trees::treeParams[sp];
            ImGui::PushID(sp);
            if (ui::TreeNode(trees::SPECIES_NAMES[sp])) {
                static int preset[trees::SPECIES] = {};
                static bool presetInit = false;
                if (!presetInit) { for (int s = 0; s < trees::SPECIES; s++) preset[s] = trees::DEFAULT_PRESET[s]; presetInit = true; }
                if (ui::Combo("Preset", &preset[sp], trees::TREE_PRESETS, trees::TREE_PRESET_COUNT)) {
                    uint32_t seed = P.seed;
                    trees::treePreset(preset[sp], P);
                    P.seed = seed;
                    vegDirty = true;
                }
                int seed = (int)P.seed;
                if (ui::InputInt("Seed", &seed)) { P.seed = (uint32_t)std::max(seed, 0); vegDirty = true; }
                ui::SeparatorText("Trunk");
                ui::SliderFloat("Height", &P.height, 2.f, 30.f, "%.1f m"); R();
                ui::SliderFloat("Trunk radius", &P.trunkRadius, 0.05f, 1.f, "%.2f m"); R();
                ui::SliderFloat("Clear trunk", &P.bole, 0.05f, 0.8f, "%.2f of height"); R();
                ui::SliderFloat("Lean", &P.lean, 0.f, 0.5f); R();
                ui::SliderFloat("Gnarl", &P.gnarl, 0.f, 1.f); R();
                ui::SliderFloat("Root flare", &P.flare, 0.f, 2.f); R();
                ui::SliderFloat("Trunk lumpiness", &P.trunkWave, 0.f, 1.5f); R();
                ui::SetItemTooltip("Irregular, fluted cross-section and a wandering trunk line (big limbs too)");
                ui::SliderFloat("Wood density", &P.density, 300.f, 1200.f, "%.0f kg/m3");
                ui::SliderFloat("Wood strength", &P.strength, 10.f, 100.f, "%.0f MPa");
                ui::SetItemTooltip("Bending strength of the green wood: what a limb takes before it snaps");
                ui::SeparatorText("Branches");
                ui::Combo("Growth", &P.growth, trees::GROWTH_NAMES, trees::GROWTH_COUNT); R();
                if (P.growth == trees::GROWTH_COLONIZE) ui::TextDisabled("(space colonization ignores the branch layout sliders below)");
                ui::SetItemTooltip("Branching rules: regular, tidy (conifers). Space colonization: branches compete for\n"
                                      "the space inside the crown shape, the irregular look of a grown oak or maple");
                ui::Combo("Crown shape", &P.crown, trees::CROWN_NAMES, trees::CROWN_COUNT); R();
                ui::SliderInt("Levels", &P.levels, 1, 4); R();
                ui::SliderInt("Main branches", &P.branches, 2, 50); R();
                ui::SliderInt("Sub-branches", &P.children, 1, 12); R();
                ui::SliderFloat("Branch angle", &P.branchAngle, 5.f, 110.f, "%.0f deg"); R();
                ui::SliderFloat("Angle variation", &P.angleVar, 0.f, 40.f, "%.0f deg"); R();
                ui::SliderFloat("Branch length", &P.lengthRatio, 0.1f, 1.2f); R();
                ui::SliderFloat("Sub-branch length", &P.childLength, 0.1f, 1.2f); R();
                ui::SliderFloat("Branch thickness", &P.radiusRatio, 0.2f, 1.f); R();
                ui::SliderFloat("Droop (gravity)", &P.gravity, 0.f, 2.f); R();
                ui::SliderFloat("Reach for light", &P.upturn, 0.f, 1.f); R();
                ui::SliderFloat("Branches up to", &P.branchTop, 0.3f, 1.f, "%.2f of leader"); R();
                ui::SetItemTooltip("How far up the leader main branches grow (start height = Clear trunk)");
                ui::SliderFloat("Sub-branch start", &P.childStart, 0.f, 0.9f, "%.2f along parent"); R();
                ui::SliderFloat("Length variation", &P.lengthVar, 0.f, 0.6f); R();
                ui::SliderFloat("Sub-branch taper", &P.childTaper, 0.f, 1.f); R();
                ui::SetItemTooltip("Sub-branches get shorter toward the parent's tip");
                ui::SliderFloat("Limb tip thickness", &P.tipTaper, 0.05f, 0.9f); R();
                ui::SliderFloat("Tier tilt", &P.tierTilt, 0.f, 1.f); R();
                ui::SetItemTooltip("Lower main branches flatter, upper ones steeper");
                ui::SeparatorText("Leaves");
                ui::SliderFloat("Leaf size", &P.leafSize, 0.02f, 0.5f, "%.2f m"); R();
                ui::SliderFloat("Leaf width", &P.leafWidth, 0.05f, 1.2f); R();
                ui::SetItemTooltip("Low values give needles");
                ui::SliderFloat("Leaf density", &P.leafDensity, 0.f, 4.f); R();
                ui::SliderInt("Leaves per cluster", &P.leavesPerCluster, 2, 120); R();
                ui::SliderInt("Clusters per twig", &P.maxClusters, 1, 24); R();
                ui::SliderFloat("Leaves start", &P.leafStart, 0.f, 0.95f, "%.2f along twig"); R();
                ui::SliderFloat("Inner leaves", &P.innerLeaves, 0.f, 3.f); R();
                ui::SetItemTooltip("Leaves on the last sub-branches between their twigs (fills the crown's inside)");
                ui::TextDisabled("A tree is capped at 220k leaf triangles: past it every cluster thins out");
                ui::SliderFloat("Cluster spread", &P.leafSpread, 0.05f, 1.f, "%.2f m"); R();
                ui::SliderFloat("Leaf curl", &P.leafDroop, 0.f, 1.f); R();
                C("Leaf colour", P.leafColor);
                ui::SliderFloat("Colour variation", &P.hueVar, 0.f, 1.f); R();
                C("Bark colour", P.barkColor);
                ui::TreePop();
            }
            ImGui::PopID();
        }
        ui::TreePop();
    }
    if (ui::TreeNode("Bush generator")) {
        const char* names[trees::BUSH_KINDS] = {"Bush", "Berry bush", "Rose", "Flowering bush"};
        const char* fruit[trees::BUSH_KINDS] = {"Berries", "Berries", "Flowers", "Blossoms"};
        for (int k = 0; k < trees::BUSH_KINDS; k++) {
            trees::BushParams& B = trees::bushParams[k];
            ImGui::PushID(100 + k);
            if (ui::TreeNode(names[k])) {
                int seed = (int)B.seed;
                if (ui::InputInt("Seed", &seed)) { B.seed = (uint32_t)std::max(seed, 0); vegDirty = true; }
                ui::SliderFloat("Size", &B.size, 0.2f, 3.f, "%.2f m"); R();
                ui::SliderInt("Stems", &B.stems, 1, 16); R();
                ui::SliderFloat("Spread", &B.spread, 0.f, 1.5f); R();
                ui::SliderInt("Branch levels", &B.levels, 1, 3); R();
                ui::SliderFloat("Branch angle", &B.branchAngle, 10.f, 90.f, "%.0f deg"); R();
                ui::SliderFloat("Gnarl", &B.gnarl, 0.f, 1.f); R();
                ui::SliderFloat("Leaf size", &B.leafSize, 0.01f, 0.2f, "%.3f m"); R();
                ui::SliderFloat("Leaf density", &B.leafDensity, 0.2f, 4.f); R();
                C("Leaf colour", B.leafColor);
                char l[64];
                snprintf(l, sizeof l, "%s amount", fruit[k]);
                ui::SliderFloat(l, &B.fruitDensity, 0.f, 2.5f); R();
                snprintf(l, sizeof l, "%s size", fruit[k]);
                ui::SliderFloat(l, &B.fruitSize, 0.003f, 0.06f, "%.3f m"); R();
                snprintf(l, sizeof l, "%s colour", fruit[k]);
                C(l, B.fruitColor);
                ui::TreePop();
            }
            ImGui::PopID();
        }
        ui::TreePop();
    }
    if (ui::Button("Regenerate vegetation")) vegDirty = true;
}

// Main panel sections: Grass, ground textures, trees & plants.
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

void panelGrass() {
    if (!ImGui::CollapsingHeader("Grass")) return;
    ImGui::BeginDisabled(bench.running);
    ImGui::SliderFloat("Density", &S.grassDensity, 10, 400, "%.0f blades/m2", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Length", &S.grassLength, 0.2f, 3, "x%.2f");
    ImGui::SliderFloat("Full density until", &S.lodStart, 2, 30, "%.0f m");
    ImGui::SliderFloat("Draw distance", &S.drawDist, 15, 150, "%.0f m");

    ImGui::SeparatorText("Blades");
    ImGui::SliderFloat("Clumping", &S.grassClump, 0, 1);
    ImGui::SetItemTooltip("0 = evenly scattered blades, 1 = tufts that splay out from their centre");
    ImGui::SliderFloat("Clump size", &S.grassClumpSize, 0.08f, 1.5f, "%.2f m");
    ImGui::SliderFloat("Splay", &S.grassTilt, 0, 1);
    ImGui::SliderFloat("Blade width", &S.grassWidth, 0.3f, 3.f, "x%.2f");
    ImGui::SliderFloat("Curvature", &S.grassCurve, 0, 3, "x%.2f");
    ImGui::SliderFloat("Twist", &S.grassTwist, 0, 1);
    ImGui::SliderFloat("Height variation", &S.grassHeightVar, 0, 1);
    ImGui::SliderFloat("Dead blades", &S.grassDead, 0, 1, "%.2f");
    ImGui::SliderFloat("Colour variation", &S.grassColorVar, 0, 2);

    ImGui::SeparatorText("Culling");
    ImGui::SliderFloat("Cull margin", &S.fovMargin, 0, 40, "+%.0f deg");
    ImGui::SliderFloat("Keep radius", &S.keepRadius, 0, 15, "%.1f m");
    ImGui::SetItemTooltip("Grass within this distance is kept at full density even behind the camera");
    ImGui::SliderFloat("Off-screen grass", &S.offscreenDensity, 0, 1, "%.2f");
    ImGui::SetItemTooltip("Density of grass outside the view (thinner, wider blades) so it still casts shadows and reflects");

    ImGui::EndDisabled();
}

void panelGroundTextures() {
    if (!ImGui::CollapsingHeader("Ground textures")) return;
    ImGui::Combo("Ground detail", &S.groundMode, "Tiled textures\0Fractal (no repeat)\0Hybrid\0Layered (virtual texture)\0");
    if (S.groundMode == 3) {
        ImGui::SliderInt("VT tiles / frame", &S.vtBudget, 8, 1024);
        ImGui::Text("VT: %u tiles generated this frame", vtQueued);
    }
    ImGui::SetItemTooltip("Fractal: generated per pixel from the world position (eroded fBm, pebbles, grit), detail down to\n"
                          "the pixel at any distance and nothing ever repeats. Hybrid: the textures, broken up by it.");
    ImGui::BeginDisabled(S.groundMode == 0);
    ImGui::SliderFloat("Fractal relief", &S.fractalRelief, 0.f, 3.f, "%.2fx");
    ImGui::SliderFloat("Pebbles", &S.fractalPebbles, 0.f, 1.f, "%.2f");
    ImGui::SliderFloat("Finest detail", &S.fractalFinest, 0.0005f, 0.05f, "%.4f m", ImGuiSliderFlags_Logarithmic);
    ImGui::SliderFloat("Large-scale warp", &S.groundWarp, 0.f, 4.f, "%.2fx");
    ImGui::SliderFloat("Grit", &S.groundGrit, 0.f, 3.f, "%.2fx");
    ImGui::SliderFloat("Mottling", &S.groundMottle, 0.f, 3.f, "%.2fx");
    ImGui::SliderFloat("Mottling scale", &S.groundMottleScale, 0.2f, 5.f, "%.2fx");
    ImGui::EndDisabled();
    ImGui::SeparatorText("Variation");
    ImGui::SliderFloat("Dark spots", &S.darkSpots, 0.f, 1.f, "%.2f");
    ImGui::SetItemTooltip("Damp, humus-rich patches");
    ImGui::SliderFloat("Dark spot size", &S.darkSpotSize, 0.1f, 5.f, "%.2fx");
    ImGui::SliderFloat("Hollow shading", &S.groundCavity, 0.f, 2.f, "%.2fx");
    ImGui::SliderFloat("Colour variation", &S.groundSat, 0.f, 2.f, "%.2f");
    ImGui::SetItemTooltip("Patches of drier (greyer) and lusher (more saturated) ground");
    ImGui::SliderFloat("Tile breakup", &S.tileBreakup, 0.f, 1.f, "%.2f");
    ImGui::BeginDisabled(!groundReady);
    ImGui::Checkbox("Generated ground textures", &S.groundTextures);
    ImGui::Text("%ux%u per 4 m tile (%.1f mm/texel), cached in texcache/ (--ground-tex N to change)", groundTexSize, groundTexSize,
                4000.f / groundTexSize);
    ImGui::SliderFloat("Relief strength", &S.groundNormal, 0, 3, "%.2fx");
    ImGui::Checkbox("Parallax occlusion", &S.parallax);
    ImGui::SliderFloat("Parallax distance", &S.parallaxDist, 2, 40, "%.0f m");
    ImGui::EndDisabled();
}

void panelTreesPlants() {
    if (!ImGui::CollapsingHeader("Trees & plants")) return;
    if (ImGui::SliderFloat("Ground clutter", &S.clutter, 0.f, 3.f, "%.2fx")) placeDirty = true;
    ImGui::SetItemTooltip("Ferns, fallen twigs, leaf litter under the trees, pebbles on bare ground and river banks");
    if (ImGui::SliderFloat("Clutter distance", &S.clutterRange, 10.f, 120.f, "%.0f m")) placeDirty = true;
    ImGui::Text("Trees %u in BVH, %u meshes + %u impostors drawn, %zu trunk colliders", numTrees, numTreeNear, numImpostors, treeBodies.size());
    ImGui::Text("Plants %u in BVH, %u drawn", numPlants, numPlantsDrawn);
    ImGui::SliderFloat("Tree density", &S.treeDensity, 0.f, 2.f, "%.2fx");
    ImGui::SliderFloat("Impostors from", &S.impostorDist, 30.f, 400.f, "%.0f m");
    if (ImGui::SliderFloat("Bushes", &S.bushAmount, 0.f, 4.f, "%.2fx")) placeDirty = true;
    if (ImGui::SliderFloat("Berry bush share", &S.berryShare, 0.f, 1.f, "%.2f")) placeDirty = true;
    drawVegetationUI();
}

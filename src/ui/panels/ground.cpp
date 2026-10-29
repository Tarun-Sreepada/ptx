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
#include "render/occlusion.h"
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
    if (!ui::CollapsingHeader("Grass")) return;
    ImGui::BeginDisabled(bench.running);
    ui::SliderFloat("Density", &S.grassDensity, 10, 400, "%.0f blades/m2", ImGuiSliderFlags_Logarithmic);
    ui::SliderFloat("Length", &S.grassLength, 0.2f, 3, "x%.2f");
    ui::SliderFloat("Full density until", &S.lodStart, 2, 30, "%.0f m");
    ui::SliderFloat("Draw distance", &S.drawDist, 15, 150, "%.0f m");

    ui::SeparatorText("Blades");
    ui::SliderFloat("Clumping", &S.grassClump, 0, 1);
    ui::SetItemTooltip("0 = evenly scattered blades, 1 = tufts that splay out from their centre");
    ui::SliderFloat("Clump size", &S.grassClumpSize, 0.08f, 1.5f, "%.2f m");
    ui::SliderFloat("Splay", &S.grassTilt, 0, 1);
    ui::SliderFloat("Blade width", &S.grassWidth, 0.3f, 3.f, "x%.2f");
    ui::SliderFloat("Curvature", &S.grassCurve, 0, 3, "x%.2f");
    ui::SliderFloat("Twist", &S.grassTwist, 0, 1);
    ui::SliderFloat("Height variation", &S.grassHeightVar, 0, 1);
    ui::SliderFloat("Dead blades", &S.grassDead, 0, 1, "%.2f");
    ui::SliderFloat("Colour variation", &S.grassColorVar, 0, 2);

    ui::SeparatorText("Culling");
    ui::SliderFloat("Cull margin", &S.fovMargin, 0, 40, "+%.0f deg");
    ui::SliderFloat("Keep radius", &S.keepRadius, 0, 15, "%.1f m");
    ui::SetItemTooltip("Grass within this distance is kept at full density even behind the camera");
    ui::SliderFloat("Off-screen grass", &S.offscreenDensity, 0, 1, "%.2f");
    ui::SetItemTooltip("Density of grass outside the view (thinner, wider blades) so it still casts shadows and reflects");

    ImGui::EndDisabled();
}

void panelGroundTextures() {
    if (!ui::CollapsingHeader("Ground textures")) return;
    ui::Combo("Ground detail", &S.groundMode, "Tiled textures\0Fractal (no repeat)\0Hybrid\0Layered (virtual texture)\0");
    if (S.groundMode == 3) {
        ui::SliderInt("VT tiles / frame", &S.vtBudget, 8, 1024);
        ui::Text("VT: %u tiles generated this frame", vtQueued);
    }
    ui::SetItemTooltip("Fractal: generated per pixel from the world position (eroded fBm, pebbles, grit), detail down to\n"
                          "the pixel at any distance and nothing ever repeats. Hybrid: the textures, broken up by it.");
    ImGui::BeginDisabled(S.groundMode == 0);
    ui::SliderFloat("Fractal relief", &S.fractalRelief, 0.f, 3.f, "%.2fx");
    ui::SliderFloat("Pebbles", &S.fractalPebbles, 0.f, 1.f, "%.2f");
    ui::SliderFloat("Finest detail", &S.fractalFinest, 0.0005f, 0.05f, "%.4f m", ImGuiSliderFlags_Logarithmic);
    ui::SliderFloat("Large-scale warp", &S.groundWarp, 0.f, 4.f, "%.2fx");
    ui::SliderFloat("Grit", &S.groundGrit, 0.f, 3.f, "%.2fx");
    ui::SliderFloat("Mottling", &S.groundMottle, 0.f, 3.f, "%.2fx");
    ui::SliderFloat("Mottling scale", &S.groundMottleScale, 0.2f, 5.f, "%.2fx");
    ImGui::EndDisabled();
    ui::SeparatorText("Variation");
    ui::SliderFloat("Dark spots", &S.darkSpots, 0.f, 1.f, "%.2f");
    ui::SetItemTooltip("Damp, humus-rich patches");
    ui::SliderFloat("Dark spot size", &S.darkSpotSize, 0.1f, 5.f, "%.2fx");
    ui::SliderFloat("Hollow shading", &S.groundCavity, 0.f, 2.f, "%.2fx");
    ui::SliderFloat("Colour variation", &S.groundSat, 0.f, 2.f, "%.2f");
    ui::SetItemTooltip("Patches of drier (greyer) and lusher (more saturated) ground");
    ui::SliderFloat("Tile breakup", &S.tileBreakup, 0.f, 1.f, "%.2f");
    ImGui::BeginDisabled(!groundReady);
    ui::Checkbox("Generated ground textures", &S.groundTextures);
    ui::Text("%ux%u per 4 m tile (%.1f mm/texel), cached in texcache/ (--ground-tex N to change)", groundTexSize, groundTexSize,
                4000.f / groundTexSize);
    ui::SliderFloat("Relief strength", &S.groundNormal, 0, 3, "%.2fx");
    ui::Checkbox("Parallax occlusion", &S.parallax);
    ui::SliderFloat("Parallax distance", &S.parallaxDist, 2, 40, "%.0f m");
    ImGui::EndDisabled();
}

void panelTreesPlants() {
    if (!ui::CollapsingHeader("Trees & plants")) return;
    if (ui::SliderFloat("Ground clutter", &S.clutter, 0.f, 3.f, "%.2fx")) placeDirty = true;
    ui::SetItemTooltip("Ferns, fallen twigs, leaf litter under the trees, pebbles on bare ground and river banks");
    if (ui::SliderFloat("Clutter distance", &S.clutterRange, 10.f, 120.f, "%.0f m")) placeDirty = true;
    ui::Text("Trees %u in BVH, %u meshes + %u impostors drawn, %zu trunk colliders", numTrees, numTreeNear, numImpostors, treeBodies.size());
    ui::Text("Plants %u in BVH, %u drawn", numPlants, numPlantsDrawn);
    ui::Text("Occlusion culled %u instances", occCulled);
    ui::SliderFloat("Tree density", &S.treeDensity, 0.f, 2.f, "%.2fx");
    ui::SliderFloat("Impostors from", &S.impostorDist, 30.f, 400.f, "%.0f m");
    if (ui::SliderFloat("Bushes", &S.bushAmount, 0.f, 4.f, "%.2fx")) placeDirty = true;
    if (ui::SliderFloat("Berry bush share", &S.berryShare, 0.f, 1.f, "%.2f")) placeDirty = true;
    drawVegetationUI();
}

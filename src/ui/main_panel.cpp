// The main settings panel ("Path Tracer" window): a header with the frame stats, then every section in ui/panels/.
#include "ui/main_panel.h"

#include <algorithm>
#include <cfloat>

#include <imgui.h>

#include "gpu/context.h"
#include "gpu/swapchain.h"
#include "render/targets.h"
#include "app/frame_state.h"
#include "ui/panels/panels.h"
#include "ui/search.h"

void drawUI() {
    ImGui::SetNextWindowPos({10, 10}, ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize({420, 860}, ImGuiCond_FirstUseEver);
    // never taller than the screen (a saved size can be), so the panel scrolls instead of running off the bottom
    float maxH = std::max(200.f, ImGui::GetIO().DisplaySize.y - 20.f);
    ImGui::SetNextWindowSizeConstraints({250, 150}, {FLT_MAX, maxH});
    ImGui::Begin("Path Tracer", nullptr, ImGuiWindowFlags_AlwaysVerticalScrollbar);
    if (ImGui::GetWindowPos().y + ImGui::GetWindowHeight() > ImGui::GetIO().DisplaySize.y)
        ImGui::SetWindowPos({ImGui::GetWindowPos().x, std::max(0.f, ImGui::GetIO().DisplaySize.y - ImGui::GetWindowHeight())});
    ImGui::Text("%s", gpuName);
    ImGui::Text("Render %ux%u -> output %ux%u   window %ux%u", W, H, OW, OH, swapExtent.width, swapExtent.height);
    ImGui::Separator();
    ImGui::Text("FPS %.1f  (CPU frame %.2f ms)", 1000.f / cpuFrameMs, cpuFrameMs);
    ImGui::Text("GPU %.2f ms  -> GPU-bound max %.0f fps", gpuEmaTotal, gpuEmaTotal > 0 ? 1000.f / gpuEmaTotal : 0.f);
    ui::beginSearch();
    panelPerformanceStats();
    panelBenchmark();
    panelFrameRateTarget();
    panelRendering();
    panelSurfaceShading();
    panelDisplayWindow();
    panelExposureTonemapBloom();
    panelLensCameraEffects();
    panelDenoiserDebugView();
    panelWorldGen();
    panelCameraMovement();
    panelSunSkyExposure();
    panelClouds();
    panelWind();
    panelLights();
    panelGrass();
    panelGroundTextures();
    panelTreesPlants();
    panelRiverWater();
    panelFogGodRays();
    panelWeather();
    panelNightSkyLightning();
    panelImpactMarksDust();
    panelCharacters();
    panelThrowRocksDestruction();
    panelBakedInteriors();
    panelWorldStreaming();
    panelAudioRayTraced();
    panelController();
    ui::endSearch();
    ImGui::End();
}

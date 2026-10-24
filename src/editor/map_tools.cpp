// World map editor (M): terrain brushes, painting grass / trees / textures, rivers, snow.
#include "editor/editor_internal.h"

namespace editor {

bool terrainUnderMouse(float out[3]) {
    float d[3];
    mouseRay(d);
    float t = rayTerrain(view.pos, d);
    if (t < 0) return false;
    for (int k = 0; k < 3; k++) out[k] = view.pos[k] + d[k] * t;
    return true;
}

void groundCircle(const float c[3], float r, ImU32 col) {
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    ImVec2 prev;
    bool have = false;
    for (int i = 0; i <= 64; i++) {
        float a = i / 64.f * 6.2831853f, q[3] = {c[0] + cosf(a) * r, 0, c[2] + sinf(a) * r};
        q[1] = host.height(q[0], q[2]) + 0.15f;
        ImVec2 sc;
        bool ok = project(q, sc);
        if (ok && have) dl->AddLine(prev, sc, col, 2.f);
        prev = sc; have = ok;
    }
}

void commitRiver() {
    if (riverPts.size() < 2) { status = "a river needs at least two points"; return; }
    float total = 0;
    for (size_t i = 1; i < riverPts.size(); i++) total += hypotf(riverPts[i].x - riverPts[i - 1].x, riverPts[i].z - riverPts[i - 1].z);
    worldmap::beginStroke();
    worldmap::addRiver(riverPts);
    status = fmt("river carved: %zu points, %.0f m (water runs from the first point downhill)", riverPts.size(), total);
    riverPts.clear();
}

void mapTools() {
    ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureKeyboard && ImGui::IsKeyPressed(ImGuiKey_M, false)) { mapMode = !mapMode; mapOpen = true; }
    if (!mapMode) { if (host.snowBrush) host.snowBrush(0, 0, 0, 0); return; }
    // ---- panel
    ImGui::SetNextWindowPos(ImVec2(20, 60), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(330, 560), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(250, 150), ImVec2(FLT_MAX, std::max(200.f, io.DisplaySize.y - 20.f)));
    if (ImGui::Begin("World map (M)", &mapOpen)) {
        static const struct { int t; const char* n; const char* tip; } tools[] = {
            {worldmap::RAISE, "Raise", "Hold to build up ground (strength = m/s at the centre)"},
            {worldmap::LOWER, "Lower", "Hold to dig down"},
            {worldmap::SMOOTH, "Smooth", "Average out bumps"},
            {worldmap::FLATTEN, "Flatten", "Pull toward the flatten height (Ctrl+click picks it)"},
            {worldmap::NOISE, "Roughen", "Add natural unevenness"},
            {worldmap::PAINT_GRASS, "Grass +", "Paint grass (up to 2x, also on bare ground)"},
            {worldmap::ERASE_GRASS, "Grass -", "Remove grass"},
            {worldmap::PAINT_TREES, "Trees +", "Paint forest density (up to 2x)"},
            {worldmap::ERASE_TREES, "Trees -", "Clear trees"},
            {worldmap::PAINT_TEXTURE, "Texture", "Paint the selected ground texture"},
            {worldmap::ERASE_TEXTURE, "Texture -", "Back to the natural ground"},
            {MT_RIVER, "River", "Click points (or drag) along the course, Enter to carve it. Water runs from the first point."},
            {worldmap::ERASE_WATER, "Water -", "Remove drawn river water (the carved bed stays)"},
            {MT_SNOW_ADD, "Snow +", "Pile snow (it slides when steeper than the angle of repose)"},
            {MT_SNOW_REMOVE, "Snow -", "Shovel snow away"},
        };
        int col = 0;
        for (auto& t : tools) {
            if (ImGui::RadioButton(t.n, mapTool == t.t)) { mapTool = t.t; if (mapTool != MT_RIVER) riverPts.clear(); }
            ImGui::SetItemTooltip("%s", t.tip);
            if (++col % 3) ImGui::SameLine(); else col = 0;
        }
        ImGui::NewLine();
        ImGui::SeparatorText("Brush");
        ImGui::SliderFloat("Radius", &mapRadius, 1.f, 500.f, "%.1f m", ImGuiSliderFlags_Logarithmic);
        ImGui::SliderFloat("Strength", &mapStrength, 0.05f, 100.f, "%.2f", ImGuiSliderFlags_Logarithmic);
        ImGui::SetItemTooltip("Raise / lower: metres per second at the centre. Paint / smooth: how fast it takes.");
        if (mapTool == worldmap::FLATTEN) ImGui::DragFloat("Flatten height", &mapFlatten, 0.1f, -500.f, 5000.f, "%.1f m");
        if (mapTool == worldmap::PAINT_TEXTURE || mapTool == worldmap::ERASE_TEXTURE) {
            ImGui::SeparatorText("Ground textures");
            for (int i = 1; i < worldmap::PALETTE; i++) {
                ImGui::PushID(i);
                if (ImGui::RadioButton("##use", mapTexSlot == i)) mapTexSlot = i;
                ImGui::SameLine();
                std::string nm = worldmap::paletteName(i);
                ImGui::SetNextItemWidth(200);
                if (matCombo(fmt("slot %d", i).c_str(), nm, true)) { worldmap::setPalette(i, nm); mapTexSlot = i; }
                ImGui::PopID();
            }
        }
        if (mapTool == MT_RIVER) {
            ImGui::SeparatorText("River pen");
            ImGui::DragFloat("Half-width (wheel)", &riverW, 0.05f, 0.5f, 60.f, "%.1f m");
            ImGui::DragFloat("Depth (Shift+wheel)", &riverD, 0.05f, 0.2f, 20.f, "%.1f m");
            ImGui::TextWrapped("Click or hold and drag to draw; scroll while drawing to widen / narrow from that point on.");
            ImGui::Text("%zu points", riverPts.size());
            if (ImGui::Button("Carve river (Enter)")) commitRiver();
            ImGui::SameLine();
            if (ImGui::Button("Undo point (Bksp)") && !riverPts.empty()) riverPts.pop_back();
            ImGui::SameLine();
            if (ImGui::Button("Cancel")) riverPts.clear();
        }
        if (mapTool == MT_SNOW_ADD || mapTool == MT_SNOW_REMOVE) {
            ImGui::SliderFloat("Snow rate", &snowRate, 0.1f, 50.f, "%.1f m/s", ImGuiSliderFlags_Logarithmic);
            ImGui::TextWrapped("Snow lives in the simulation (Path Tracer > Weather): it slides, melts below the snow line and the melt water runs toward rivers.");
        }
        ImGui::Separator();
        if (ImGui::Button("Undo (Ctrl+Z)")) worldmap::undo();
        ImGui::SameLine();
        if (ImGui::Button(worldmap::dirty() ? "Save (Ctrl+S) *" : "Save (Ctrl+S)")) { std::string err; if (!saveAll(err)) status = "save failed: " + err; }
        ImGui::TextWrapped("4 km x 4 km around the origin, 2 m cells. Saved to world/terrain/map.bin. M: back to object editing.");
        if (!status.empty()) ImGui::TextWrapped("%s", status.c_str());
    }
    ImGui::End();
    if (!mapOpen) { mapMode = false; mapOpen = true; }

    // ---- keys
    if (!io.WantCaptureKeyboard) {
        if (io.KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_Z, false)) worldmap::undo();
        if (mapTool == MT_RIVER) {
            if (ImGui::IsKeyPressed(ImGuiKey_Enter, false) || ImGui::IsKeyPressed(ImGuiKey_KeypadEnter, false)) commitRiver();
            if (ImGui::IsKeyPressed(ImGuiKey_Backspace, false) && !riverPts.empty()) riverPts.pop_back();
            if (ImGui::IsKeyPressed(ImGuiKey_Escape, false)) riverPts.clear();
        }
    }

    // ---- brush
    float at[3];
    bool hit = !io.WantCaptureMouse && terrainUnderMouse(at);
    bool snowT = mapTool == MT_SNOW_ADD || mapTool == MT_SNOW_REMOVE;
    if (mapTool == MT_RIVER) {
        // the pen: wheel = width, Shift+wheel = depth (over the world, not the panels)
        if (!io.WantCaptureMouse && io.MouseWheel != 0.f) {
            float k = powf(1.12f, io.MouseWheel);
            if (io.KeyShift) riverD = std::clamp(riverD * k, 0.2f, 20.f);
            else riverW = std::clamp(riverW * k, 0.5f, 60.f);
        }
        // the course so far, and the next segment to the mouse
        ImDrawList* dl = ImGui::GetBackgroundDrawList();
        for (auto& rp : riverPts) {  // each point's own width
            float c[3] = {rp.x, 0, rp.z};
            groundCircle(c, rp.width, IM_COL32(80, 170, 255, 70));
        }
        for (size_t i = 0; i < riverPts.size(); i++) {
            float q[3] = {riverPts[i].x, host.height(riverPts[i].x, riverPts[i].z) + 0.3f, riverPts[i].z};
            ImVec2 a;
            if (!project(q, a)) continue;
            dl->AddCircleFilled(a, 4, IM_COL32(80, 170, 255, 230));
            if (i + 1 < riverPts.size() || hit) {
                float q2[3];
                if (i + 1 < riverPts.size()) { q2[0] = riverPts[i + 1].x; q2[2] = riverPts[i + 1].z; }
                else { q2[0] = at[0]; q2[2] = at[2]; }
                q2[1] = host.height(q2[0], q2[2]) + 0.3f;
                ImVec2 b;
                if (project(q2, b)) dl->AddLine(a, b, IM_COL32(80, 170, 255, i + 1 < riverPts.size() ? 230 : 120), 3.f);
            }
        }
        if (hit) {
            groundCircle(at, riverW, IM_COL32(80, 170, 255, 200));
            ImVec2 sp;
            float q[3] = {at[0], at[1] + 0.5f, at[2]};
            if (project(q, sp))
                dl->AddText(ImVec2(sp.x + 14, sp.y - 8), IM_COL32(200, 230, 255, 255), fmt("width %.1f m  depth %.1f m", 2.f * riverW, riverD).c_str());
        }
        if (hit && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) riverPts.push_back({at[0], at[2], riverW, riverD});
        else if (hit && ImGui::IsMouseDown(ImGuiMouseButton_Left) && !riverPts.empty() &&
                 hypotf(at[0] - riverPts.back().x, at[2] - riverPts.back().z) > std::clamp(riverW * 0.75f, 2.f, 8.f))
            riverPts.push_back({at[0], at[2], riverW, riverD});  // dragging draws the course with the pen's current size
        if (host.snowBrush) host.snowBrush(0, 0, 0, 0);
        return;
    }
    if (hit) groundCircle(at, mapRadius, snowT ? IM_COL32(230, 240, 255, 200) : IM_COL32(255, 200, 60, 200));
    if (hit && io.KeyCtrl && ImGui::IsMouseClicked(ImGuiMouseButton_Left) && mapTool == worldmap::FLATTEN) {
        mapFlatten = at[1];
        status = fmt("flatten height %.1f m", mapFlatten);
        return;
    }
    bool down = hit && ImGui::IsMouseDown(ImGuiMouseButton_Left);
    if (snowT) {
        if (host.snowBrush) host.snowBrush(at[0], at[2], down ? mapRadius : 0.f, mapTool == MT_SNOW_ADD ? snowRate : -snowRate);
        return;
    }
    if (host.snowBrush) host.snowBrush(0, 0, 0, 0);
    if (down && !stroking) { worldmap::beginStroke(); stroking = true; }
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) stroking = false;
    if (down) worldmap::brush((worldmap::Tool)mapTool, at[0], at[2], mapRadius, mapStrength, std::min(io.DeltaTime, 0.05f), mapFlatten, mapTexSlot);
}

}  // namespace editor

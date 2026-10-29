#include "ui/overlays.h"

#include <algorithm>
#include <cfloat>
#include <cmath>

#include <vulkan/vulkan.h>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>

#include "core/math.h"
#include "core/prof.h"
#include "editor/editor.h"
#include "gpu/context.h"
#include "gpu/swapchain.h"
#include "render/bake.h"
#include "scene/camera.h"
#include "tools/screenshot.h"
#include "ui/world_map.h"

namespace {
constexpr float HINT_MARGIN = 10.f;
constexpr double SHOT_MESSAGE_SECONDS = 4.0;
constexpr float DEG = 3.14159265f / 180.f;

bool keyEdge(int key, bool& prev) {
    bool k = glfwGetKey(window, key) == GLFW_PRESS;
    bool hit = k && !prev;
    prev = k;
    return hit;
}

void helpWindow(bool* open) {
    ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f), ImGuiCond_FirstUseEver, ImVec2(0.5f, 0.5f));
    ImGui::SetNextWindowSize(ImVec2(620, 640), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(300, 200), ImVec2(FLT_MAX, std::max(200.f, io.DisplaySize.y - 20.f)));
    if (ImGui::Begin("Controls (F1)", open)) {
        auto section = [](const char* t) { ImGui::SeparatorText(t); };
        auto row = [](const char* k, const char* what) {
            ImGui::TableNextRow();
            ImGui::TableNextColumn(); ImGui::TextColored(ImVec4(1.f, 0.85f, 0.4f, 1), "%s", k);
            ImGui::TableNextColumn(); ImGui::TextWrapped("%s", what);
        };
        auto table = [](const char* id) {
            bool ok = ImGui::BeginTable(id, 2, ImGuiTableFlags_RowBg | ImGuiTableFlags_SizingStretchProp);
            if (ok) { ImGui::TableSetupColumn("key", ImGuiTableColumnFlags_WidthFixed, 170); ImGui::TableSetupColumn("action"); }
            return ok;
        };
        section("Moving");
        if (table("move")) {
            row("Right mouse + move", "Look around");
            row("W A S D", "Move");
            row("E / Q", "Fly up / down (E switches lights while walking)");
            row("Shift / Ctrl", "Faster / turbo (fly)");
            row("F", "Toggle walking / flying");
            row("Space", "Jump (walking)");
            row("Gamepad", "Sticks move / look, RT throw, A jump, Y walk / fly, LB / RB down / up, LT or L3 sprint");
            ImGui::EndTable();
        }
        section("World");
        if (table("world")) {
            row("G / Left mouse", "Throw a rock (size and speed: Interaction panel)");
            row("L", "Lights of the building you're in on / off");
            row("P", "Primary rays: raster G-buffer / path traced");
            row("I", "Global illumination on / off");
            ImGui::EndTable();
        }
        section("Editor (Tab)");
        if (table("edit")) {
            row("Tab", "Object editor on / off");
            row("Click / drag", "Select / move on the ground (Alt+drag: up / down)");
            row("R / Shift+R", "Rotate by the step");
            row("Arrows, PgUp / PgDn", "Nudge by the grid");
            row("N", "Grid snap on / off (hold Shift to invert)");
            row("Ctrl+D / Del", "Duplicate / delete");
            row("Ctrl+Z / Ctrl+S", "Undo / save (cells and the world map)");
            row("Drag from Assets", "Place an asset where you release it; drop files on the window to import");
            ImGui::EndTable();
        }
        section("World map (M, inside the editor)");
        if (table("map")) {
            row("M", "World map mode on / off: left mouse paints with the tool");
            row("Left mouse (hold)", "Raise / lower / smooth / flatten / roughen, paint grass, trees, textures, snow");
            row("Ctrl+click", "Flatten: pick the height under the cursor");
            row("River: click / drag", "Draw the course (water runs from the first point)");
            row("River: wheel", "Wider / narrower from here on");
            row("River: Shift+wheel", "Deeper / shallower");
            row("River: Enter / Bksp / Esc", "Carve it / remove last point / cancel");
            row("Ctrl+Z", "Undo the last stroke");
            ImGui::EndTable();
        }
        section("Capture and debug");
        if (table("dbg")) {
            row("F12", "Screenshot (scene) -> screenshots/*.png + .txt scene, path copied");
            row("Shift+F12", "Screenshot with the UI");
            row("F9", "Profiler: capture frames (flame chart)");
            row("F8", "Profiler window");
            row("F1", "This list");
            row("M", "World map (click: marker, right-click: teleport)");
            ImGui::EndTable();
        }
        ImGui::TextDisabled("Reopen a screenshot's scene: ./build/ptx --from-shot <file.png>");
    }
    ImGui::End();
}

void profilerKeys() {
    static bool f9 = false, profOpen = false;
    if (keyEdge(GLFW_KEY_F9, f9)) { prof::toggleCapture(); profOpen = true; }
    if (ImGui::IsKeyPressed(ImGuiKey_F8, false)) profOpen = !profOpen;
    if (profOpen) prof::ui(&profOpen);
}

void helpKeyAndHint() {
    static bool helpOpen = false;
    if (ImGui::IsKeyPressed(ImGuiKey_F1, false)) helpOpen = !helpOpen;
    ImGuiIO& io = ImGui::GetIO();
    const char* hint = "F1: controls";
    ImVec2 ts = ImGui::CalcTextSize(hint);
    ImGui::GetForegroundDrawList()->AddText(ImVec2(io.DisplaySize.x - ts.x - HINT_MARGIN, 6), IM_COL32(255, 255, 255, 140), hint);
    if (helpOpen) helpWindow(&helpOpen);
}

// F12 the scene, Shift+F12 with the UI (falls back to the scene where the swapchain can't be read)
void screenshotKeys() {
    static bool f12 = false;
    if (keyEdge(GLFW_KEY_F12, f12)) {
        if (ImGui::GetIO().KeyShift && swapCanCopy && !hdrActive) shotUIReq = true;
        else shotSceneReq = true;
    }
    if (glfwGetTime() - shotMsgT < SHOT_MESSAGE_SECONDS) {
        ImGuiIO& io = ImGui::GetIO();
        ImDrawList* dl = ImGui::GetForegroundDrawList();
        dl->AddRectFilled(ImVec2(10, io.DisplaySize.y - 34), ImVec2(io.DisplaySize.x * 0.75f, io.DisplaySize.y - 10), IM_COL32(0, 0, 0, 170), 4.f);
        dl->AddText(ImVec2(16, io.DisplaySize.y - 30), IM_COL32(255, 255, 255, 255), shotMsg.c_str());
    }
}

void editorToggle() {
    static bool tabPrev = false;
    bool tab = !ImGui::GetIO().WantCaptureKeyboard && glfwGetKey(window, GLFW_KEY_TAB) == GLFW_PRESS;
    if (tab && !tabPrev) editor::setActive(!editor::active());
    tabPrev = tab;
    V3 f = cam.fwd(), r = cam.right(), u = cam.up();
    editor::View ev{{cam.pos.x, cam.pos.y, cam.pos.z}, {f.x, f.y, f.z}, {r.x, r.y, r.z}, {u.x, u.y, u.z}, tanf(cam.fovDeg * 0.5f * DEG)};
    editor::ui(ev);
}

// light switch for the building you're in: E while walking (E flies up otherwise), L any time
void lightSwitch() {
    static bool prevKey = false;
    bool k = !ImGui::GetIO().WantCaptureKeyboard &&
             ((walkMode && glfwGetKey(window, GLFW_KEY_E) == GLFW_PRESS) || glfwGetKey(window, GLFW_KEY_L) == GLFW_PRESS);
    if (k && !prevKey)
        if (int v = volumeAt(cam.pos.x, cam.pos.y, cam.pos.z); v >= 0) bakeVols[v].on = !bakeVols[v].on;
    prevKey = k;
}
}  // namespace

void drawOverlays() {
    profilerKeys();
    helpKeyAndHint();
    screenshotKeys();
    editorToggle();
    lightSwitch();
    worldmap_ui::window();
}

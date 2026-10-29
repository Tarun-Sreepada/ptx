// World map window (M): the generated terrain seen from above - biomes, sea, lakes, rivers, hill shading - with the
// player's position. Wheel zooms, left-drag pans, left-click places a marker, right-click: teleport / marker menu.
// The image is sampled on a worker thread (terrain.h / biome.h, no streamed zones) whenever the view changes.
#include "ui/world_map.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <string>
#include <thread>
#include <vector>
#include <imgui.h>
#include "physics/physics.h"
#include "scene/camera.h"
#include "world/terrain.h"
#include "world/terrain_query.h"

namespace {
constexpr int MAP_N = 192;                 // samples per side
constexpr float EYE = 1.65f;               // walking eye height (camera.cpp EYE_HEIGHT)
// map colour per biome
constexpr ImU32 BIOME_MAP_COL[B_COUNT] = {
    IM_COL32(86, 128, 52, 255),  IM_COL32(222, 186, 120, 255), IM_COL32(186, 98, 56, 255),  IM_COL32(176, 160, 82, 255),
    IM_COL32(128, 124, 96, 255), IM_COL32(232, 240, 250, 255), IM_COL32(128, 124, 120, 255), IM_COL32(44, 84, 52, 255),
    IM_COL32(28, 110, 38, 255),  IM_COL32(70, 96, 70, 255),   IM_COL32(48, 42, 44, 255),   IM_COL32(240, 238, 228, 255),
    IM_COL32(150, 156, 140, 255)};

struct Marker { float x, z; std::string name; };
std::vector<Marker> markers;
bool open_ = false;
float viewX = 0.f, viewZ = 0.f, span = 6000.f;  // centre and width of the view (m)
bool follow = true;

// worker: one image at a time; the newest request wins
struct Img { float x0, z0, span; std::vector<ImU32> px; std::vector<float> h; std::vector<uint8_t> biome; };
Img shown;
std::atomic<bool> busy{false}, ready{false};
Img work;
float reqX = 1e30f, reqZ = 1e30f, reqSpan = 0.f;
int reqVersion = -1, mapVersion = 0;

ImU32 shade(ImU32 c, float k) {
    auto ch = [&](int s) { return (ImU32)std::clamp((int)(((c >> s) & 255) * k), 0, 255) << s; };
    return ch(0) | ch(8) | ch(16) | (c & 0xFF000000u);
}
ImU32 lerpCol(ImU32 a, ImU32 b, float t) {
    auto ch = [&](int s) { return (ImU32)std::clamp((int)(((a >> s) & 255) * (1 - t) + ((b >> s) & 255) * t), 0, 255) << s; };
    return ch(0) | ch(8) | ch(16) | 0xFF000000u;
}

void build(Img& im) {
    int N = MAP_N;
    float cs = im.span / N;
    im.px.assign(N * N, 0);
    im.h.assign(N * N, 0.f);
    im.biome.assign(N * N, 0);
    std::vector<float> water(N * N, -1e9f);
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++) {
            float x = im.x0 + (i + 0.5f) * cs, z = im.z0 + (j + 0.5f) * cs;
            float h = terrainBase(x, z, nullptr, 0);
            im.h[j * N + i] = h;
            float w[B_COUNT];
            biomeWeights(x, z, w);
            int b = int(std::max_element(w, w + B_COUNT) - w);
            im.biome[j * N + i] = (uint8_t)b;
            float sw = standingWaterAt(x, z);
            RiverInfo r = riverInfo(x, z);
            if (fabsf(r.d) < r.w + cs * 0.3f && r.level > h) sw = std::max(sw, r.level);
            water[j * N + i] = sw > h ? sw : -1e9f;
        }
    for (int j = 0; j < N; j++)
        for (int i = 0; i < N; i++) {
            int k = j * N + i;
            float h = im.h[k];
            if (water[k] > -1e8f) {  // deeper = darker
                float d = std::clamp((water[k] - h) / 40.f, 0.f, 1.f);
                im.px[k] = lerpCol(IM_COL32(70, 140, 190, 255), IM_COL32(16, 40, 92, 255), sqrtf(d));
                continue;
            }
            float hx = im.h[j * N + std::min(i + 1, N - 1)] - im.h[j * N + std::max(i - 1, 0)];
            float hz = im.h[std::min(j + 1, N - 1) * N + i] - im.h[std::max(j - 1, 0) * N + i];
            float sl = (-hx - hz) / (2.f * cs);  // light from the north-west
            float k2 = std::clamp(1.f + sl * 1.6f, 0.45f, 1.5f) * (0.85f + std::clamp(h / 600.f, -0.1f, 0.3f));
            im.px[k] = shade(BIOME_MAP_COL[im.biome[k]], k2);
        }
}

void request(float x0, float z0, float sp) {
    if (busy.load()) return;
    if (x0 == reqX && z0 == reqZ && sp == reqSpan && reqVersion == mapVersion) return;
    reqX = x0; reqZ = z0; reqSpan = sp; reqVersion = mapVersion;
    work.x0 = x0; work.z0 = z0; work.span = sp;
    busy = true;
    std::thread([] { build(work); ready = true; busy = false; }).detach();
}

const char* biomeAt(float x, float z) {
    if (standingWaterAt(x, z) > terrainBase(x, z, nullptr, 0)) return coastDist(x, z) < gBiome.beach ? "Ocean" : "Lake";
    float w[B_COUNT];
    biomeWeights(x, z, w);
    return BIOME_NAMES[std::max_element(w, w + B_COUNT) - w];
}
}  // namespace

namespace worldmap_ui {
void invalidate() { mapVersion++; }
bool& isOpen() { return open_; }

void teleport(float x, float z) {
    float h = heightFn(x, z);
    float sw = standingWaterAt(x, z);
    float floor_ = std::max(h, sw > -1e8f ? sw : h);
    if (walkMode) {
        V3 feet{x, floor_ + 0.3f, z};
        physics::spawnPlayer(&feet.x);
        playerFeet = feet;
        walkReset = true;
        cam.pos = feet + V3{0, EYE, 0};
    } else {
        cam.pos = {x, floor_ + 25.f, z};
    }
    prevCam = cam;
}

void window() {
    ImGuiIO& io = ImGui::GetIO();
    if (!io.WantCaptureKeyboard && ImGui::IsKeyPressed(ImGuiKey_M, false)) open_ = !open_;
    if (!open_) return;
    ImGui::SetNextWindowSize(ImVec2(760, 820), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("World map", &open_)) { ImGui::End(); return; }
    if (ready.exchange(false)) std::swap(shown, work);

    ImGui::Checkbox("Follow player", &follow);
    ImGui::SameLine();
    if (ImGui::Button("Centre")) { viewX = cam.pos.x; viewZ = cam.pos.z; }
    ImGui::SameLine();
    ImGui::SetNextItemWidth(160);
    ImGui::SliderFloat("Width", &span, 200.f, 60000.f, "%.0f m", ImGuiSliderFlags_Logarithmic);
    ImGui::SameLine();
    if (ImGui::Button("Redraw")) invalidate();
    if (follow) { viewX = cam.pos.x; viewZ = cam.pos.z; }

    float side = std::max(std::min(ImGui::GetContentRegionAvail().x, ImGui::GetContentRegionAvail().y - 150.f), 200.f);
    ImVec2 p0 = ImGui::GetCursorScreenPos();
    ImGui::InvisibleButton("map", ImVec2(side, side), ImGuiButtonFlags_MouseButtonLeft | ImGuiButtonFlags_MouseButtonRight);
    bool hovered = ImGui::IsItemHovered();
    // screen <-> world: +x to the right, +z up the screen (north)
    auto toWorld = [&](ImVec2 s, float& x, float& z) {
        x = viewX + ((s.x - p0.x) / side - 0.5f) * span;
        z = viewZ - ((s.y - p0.y) / side - 0.5f) * span;
    };
    auto toScreen = [&](float x, float z) {
        return ImVec2(p0.x + ((x - viewX) / span + 0.5f) * side, p0.y + (0.5f - (z - viewZ) / span) * side);
    };
    if (hovered && io.MouseWheel != 0.f) {  // zoom about the cursor
        float mx, mz;
        toWorld(io.MousePos, mx, mz);
        float ns = std::clamp(span * powf(0.8f, io.MouseWheel), 200.f, 60000.f);
        viewX = mx + (viewX - mx) * ns / span;
        viewZ = mz + (viewZ - mz) * ns / span;
        span = ns;
        follow = false;
    }
    if (ImGui::IsItemActive() && ImGui::IsMouseDragging(ImGuiMouseButton_Left, 3.f)) {
        viewX -= io.MouseDelta.x / side * span;
        viewZ += io.MouseDelta.y / side * span;
        follow = false;
    }
    static float popX = 0.f, popZ = 0.f;
    if (hovered && ImGui::IsMouseReleased(ImGuiMouseButton_Left) && ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 3.f).x == 0.f &&
        ImGui::GetMouseDragDelta(ImGuiMouseButton_Left, 3.f).y == 0.f) {
        float x, z;
        toWorld(io.MousePos, x, z);
        char nm[32];
        snprintf(nm, sizeof nm, "Marker %zu", markers.size() + 1);
        markers.push_back({x, z, nm});
    }
    if (hovered && ImGui::IsMouseClicked(ImGuiMouseButton_Right)) {
        toWorld(io.MousePos, popX, popZ);
        ImGui::OpenPopup("mapctx");
    }

    // the image: re-sampled when the view moves more than a few cells (shown shifted meanwhile)
    float cs = span / MAP_N;
    float x0 = floorf((viewX - span * 0.5f) / cs) * cs, z0 = floorf((viewZ - span * 0.5f) / cs) * cs;
    request(x0, z0, span);
    ImDrawList* dl = ImGui::GetWindowDrawList();
    dl->PushClipRect(p0, ImVec2(p0.x + side, p0.y + side), true);
    dl->AddRectFilled(p0, ImVec2(p0.x + side, p0.y + side), IM_COL32(20, 24, 30, 255));
    if (!shown.px.empty()) {
        float c = shown.span / MAP_N;
        for (int j = 0; j < MAP_N; j++)
            for (int i = 0; i < MAP_N; i++) {
                ImVec2 a = toScreen(shown.x0 + i * c, shown.z0 + (j + 1) * c), b = toScreen(shown.x0 + (i + 1) * c, shown.z0 + j * c);
                if (b.x < p0.x || a.x > p0.x + side || b.y < p0.y || a.y > p0.y + side) continue;
                dl->AddRectFilled(a, ImVec2(b.x + 0.5f, b.y + 0.5f), shown.px[j * MAP_N + i]);
            }
    }
    // markers and the player
    for (size_t i = 0; i < markers.size(); i++) {
        ImVec2 s = toScreen(markers[i].x, markers[i].z);
        dl->AddTriangleFilled(ImVec2(s.x, s.y), ImVec2(s.x - 6, s.y - 14), ImVec2(s.x + 6, s.y - 14), IM_COL32(230, 50, 50, 255));
        dl->AddCircleFilled(ImVec2(s.x, s.y - 14), 6, IM_COL32(230, 50, 50, 255));
        dl->AddText(ImVec2(s.x + 8, s.y - 22), IM_COL32(255, 255, 255, 230), markers[i].name.c_str());
    }
    {
        ImVec2 s = toScreen(cam.pos.x, cam.pos.z);
        float fx = sinf(cam.yaw), fz = cosf(cam.yaw);  // forward on the map (screen y is -z)
        ImVec2 tip(s.x + fx * 12, s.y - fz * 12), l(s.x - fz * 6 - fx * 6, s.y - fx * 6 + fz * 6), r(s.x + fz * 6 - fx * 6, s.y + fx * 6 + fz * 6);
        dl->AddTriangleFilled(tip, l, r, IM_COL32(255, 255, 255, 255));
        dl->AddTriangle(tip, l, r, IM_COL32(0, 0, 0, 255), 1.5f);
    }
    // scale bar
    {
        float m = powf(10.f, floorf(log10f(span * 0.25f)));
        if (span * 0.25f / m > 5.f) m *= 5.f; else if (span * 0.25f / m > 2.f) m *= 2.f;
        float px = m / span * side;
        ImVec2 a(p0.x + 12, p0.y + side - 14);
        dl->AddLine(a, ImVec2(a.x + px, a.y), IM_COL32(255, 255, 255, 230), 3.f);
        char t[32];
        snprintf(t, sizeof t, m >= 1000.f ? "%.0f km" : "%.0f m", m >= 1000.f ? m / 1000.f : m);
        dl->AddText(ImVec2(a.x, a.y - 16), IM_COL32(255, 255, 255, 230), t);
    }
    if (busy.load()) dl->AddText(ImVec2(p0.x + side - 70, p0.y + 6), IM_COL32(255, 255, 255, 200), "drawing...");
    dl->PopClipRect();

    if (hovered) {
        float x, z;
        toWorld(io.MousePos, x, z);
        ImGui::SetTooltip("%s\n%.0f, %.0f   %.0f m", biomeAt(x, z), x, z, terrainBase(x, z, nullptr, 0));
    }
    if (ImGui::BeginPopup("mapctx")) {
        ImGui::TextDisabled("%.0f, %.0f  %s", popX, popZ, biomeAt(popX, popZ));
        if (ImGui::MenuItem("Teleport here")) teleport(popX, popZ);
        if (ImGui::MenuItem("Place marker here")) {
            char nm[32];
            snprintf(nm, sizeof nm, "Marker %zu", markers.size() + 1);
            markers.push_back({popX, popZ, nm});
        }
        int near = -1;
        float best = span * 0.02f;
        for (size_t i = 0; i < markers.size(); i++) {
            float d = hypotf(markers[i].x - popX, markers[i].z - popZ);
            if (d < best) { best = d; near = (int)i; }
        }
        if (near >= 0) {
            if (ImGui::MenuItem("Teleport to marker")) teleport(markers[near].x, markers[near].z);
            if (ImGui::MenuItem("Remove marker")) markers.erase(markers.begin() + near);
        }
        if (ImGui::MenuItem("Centre map here")) { viewX = popX; viewZ = popZ; follow = false; }
        ImGui::EndPopup();
    }

    ImGui::Text("You: %.0f, %.0f  (%s)", cam.pos.x, cam.pos.z, biomeAt(cam.pos.x, cam.pos.z));
    ImGui::TextDisabled("Wheel: zoom   drag: pan   click: marker   right-click: teleport / options");
    if (!markers.empty() && ImGui::BeginTable("markers", 3, ImGuiTableFlags_SizingStretchProp | ImGuiTableFlags_ScrollY, ImVec2(0, 90))) {
        for (size_t i = 0; i < markers.size(); i++) {
            ImGui::PushID((int)i);
            ImGui::TableNextColumn();
            char buf[64];
            snprintf(buf, sizeof buf, "%s", markers[i].name.c_str());
            ImGui::SetNextItemWidth(-1);
            if (ImGui::InputText("##n", buf, sizeof buf)) markers[i].name = buf;
            ImGui::TableNextColumn();
            ImGui::Text("%.0f, %.0f", markers[i].x, markers[i].z);
            ImGui::TableNextColumn();
            if (ImGui::SmallButton("Teleport")) teleport(markers[i].x, markers[i].z);
            ImGui::SameLine();
            bool del = ImGui::SmallButton("X");
            ImGui::PopID();
            if (del) { markers.erase(markers.begin() + i); break; }
        }
        ImGui::EndTable();
    }
    // legend
    for (int b = 0; b < B_COUNT; b++) {
        if (b % 5) ImGui::SameLine();
        ImGui::ColorButton(BIOME_NAMES[b], ImGui::ColorConvertU32ToFloat4(BIOME_MAP_COL[b]), ImGuiColorEditFlags_NoTooltip, ImVec2(12, 12));
        ImGui::SameLine();
        ImGui::TextUnformatted(BIOME_NAMES[b]);
    }
    ImGui::End();
}
}  // namespace worldmap_ui

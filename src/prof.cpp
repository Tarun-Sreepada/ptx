#include "prof.h"

#include <imgui.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <ctime>
#include <filesystem>
#include <map>
#include <mutex>
#include <thread>

namespace prof {
std::atomic<bool> capturing{false};

namespace {
struct Event { const char* name; uint64_t t0, t1; uint16_t depth; };
struct Lane {  // one per thread that ever opened a zone
    std::mutex mu;
    std::vector<Event> ev;
    std::string name;
    uint32_t id = 0;
    int depth = 0;
};
std::mutex lanesMu;
std::vector<Lane*> lanes;  // never freed: threads live for the program
thread_local Lane* myLane = nullptr;

Lane& lane() {
    if (!myLane) {
        myLane = new Lane;
        std::lock_guard lk(lanesMu);
        myLane->id = (uint32_t)lanes.size();
        myLane->name = "thread " + std::to_string(myLane->id);
        lanes.push_back(myLane);
    }
    return *myLane;
}

// capture state (main thread)
int wantFrames = 20, framesLeft = 0;
bool pendingStart = false;
uint64_t capStart = 0;
std::vector<uint64_t> frameStarts;
struct GpuEv { std::string name; bool async; uint64_t t0, t1; };
std::vector<GpuEv> gpuEvs;
float slowMo = 0.f;
bool slowDuring = true;

// the finished capture, for the viewer
struct Snap {
    struct L { std::string name; std::vector<Event> ev; int maxDepth = 0; };
    std::vector<L> lanes;
    std::vector<GpuEv> gpu;
    std::vector<uint64_t> frames;
    uint64_t t0 = 0, t1 = 1;
    struct Agg { std::string name; int calls = 0; double total = 0, self = 0, maxMs = 0; };
    std::vector<Agg> agg;
} snap;
double v0 = 0, v1 = 1;  // view window (ns from capture start)
std::string lastExport;

void finishCapture() {
    capturing = false;
    snap = {};
    snap.frames = frameStarts;
    snap.gpu = gpuEvs;
    snap.t0 = capStart;
    snap.t1 = Scope("").t0;  // now
    std::map<std::string, Snap::Agg> agg;
    std::lock_guard lk(lanesMu);
    for (Lane* l : lanes) {
        Snap::L s;
        {
            std::lock_guard lk2(l->mu);
            s.ev.swap(l->ev);
        }
        if (s.ev.empty()) continue;
        s.name = l->name;
        std::sort(s.ev.begin(), s.ev.end(), [](const Event& a, const Event& b) { return a.t0 < b.t0 || (a.t0 == b.t0 && a.depth < b.depth); });
        // self time: a zone's duration minus its direct children's
        std::vector<size_t> stack;
        std::vector<double> child(s.ev.size(), 0.0);
        for (size_t i = 0; i < s.ev.size(); i++) {
            const Event& e = s.ev[i];
            s.maxDepth = std::max(s.maxDepth, (int)e.depth);
            while (!stack.empty() && s.ev[stack.back()].t1 <= e.t0) stack.pop_back();
            if (!stack.empty()) child[stack.back()] += (double)(e.t1 - e.t0);
            stack.push_back(i);
        }
        for (size_t i = 0; i < s.ev.size(); i++) {
            const Event& e = s.ev[i];
            auto& a = agg[e.name];
            a.name = e.name;
            a.calls++;
            double ms = (e.t1 - e.t0) / 1e6;
            a.total += ms;
            a.self += ms - child[i] / 1e6;
            a.maxMs = std::max(a.maxMs, ms);
        }
        snap.lanes.push_back(std::move(s));
    }
    for (auto& g : snap.gpu) {
        auto& a = agg["[GPU] " + g.name];
        a.name = "[GPU] " + g.name;
        a.calls++;
        double ms = (g.t1 - g.t0) / 1e6;
        a.total += ms; a.self += ms; a.maxMs = std::max(a.maxMs, ms);
    }
    for (auto& [k, a] : agg) snap.agg.push_back(a);
    std::sort(snap.agg.begin(), snap.agg.end(), [](auto& a, auto& b) { return a.self > b.self; });
    v0 = 0;
    v1 = (double)(snap.t1 - snap.t0);
}

std::string exportJson() {
    std::error_code ec;
    std::filesystem::create_directories("traces", ec);
    char name[96];
    time_t t = time(nullptr);
    strftime(name, sizeof name, "traces/ptx_%Y%m%d_%H%M%S.json", localtime(&t));
    FILE* f = fopen(name, "w");
    if (!f) return "";
    auto esc = [](const std::string& s) { std::string o; for (char c : s) { if (c == '"' || c == '\\') o += '\\'; o += c; } return o; };
    fprintf(f, "{\"displayTimeUnit\":\"ms\",\"traceEvents\":[\n");
    bool first = true;
    auto sep = [&] { if (!first) fprintf(f, ",\n"); first = false; };
    sep(); fprintf(f, "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":1,\"args\":{\"name\":\"ptx CPU\"}}");
    sep(); fprintf(f, "{\"name\":\"process_name\",\"ph\":\"M\",\"pid\":2,\"args\":{\"name\":\"ptx GPU (aligned to frame starts)\"}}");
    for (size_t li = 0; li < snap.lanes.size(); li++) {
        auto& l = snap.lanes[li];
        sep(); fprintf(f, "{\"name\":\"thread_name\",\"ph\":\"M\",\"pid\":1,\"tid\":%zu,\"args\":{\"name\":\"%s\"}}", li, esc(l.name).c_str());
        for (auto& e : l.ev) {
            sep();
            fprintf(f, "{\"name\":\"%s\",\"ph\":\"X\",\"pid\":1,\"tid\":%zu,\"ts\":%.3f,\"dur\":%.3f}", esc(e.name).c_str(), li,
                    (e.t0 - snap.t0) / 1e3, (e.t1 - e.t0) / 1e3);
        }
    }
    for (auto& g : snap.gpu) {
        sep();
        fprintf(f, "{\"name\":\"%s\",\"ph\":\"X\",\"pid\":2,\"tid\":%d,\"ts\":%.3f,\"dur\":%.3f}", esc(g.name).c_str(), g.async ? 1 : 0,
                (g.t0 - snap.t0) / 1e3, (g.t1 - g.t0) / 1e3);
    }
    for (size_t i = 0; i < snap.frames.size(); i++) {
        sep();
        fprintf(f, "{\"name\":\"frame %zu\",\"ph\":\"i\",\"s\":\"g\",\"pid\":1,\"tid\":0,\"ts\":%.3f}", i, (snap.frames[i] - snap.t0) / 1e3);
    }
    fprintf(f, "\n]}\n");
    fclose(f);
    return std::filesystem::absolute(name).string();
}

ImU32 colorOf(const char* name, bool gpu) {
    uint32_t h = 2166136261u;
    for (const char* p = name; *p; p++) h = (h ^ (uint8_t)*p) * 16777619u;
    float hue = (h % 1000) / 1000.f;
    ImVec4 c;
    ImGui::ColorConvertHSVtoRGB(gpu ? 0.55f + 0.2f * hue : 0.02f + 0.14f * hue, 0.55f, gpu ? 0.75f : 0.9f, c.x, c.y, c.z);  // GPU blues, CPU warm
    return ImGui::GetColorU32(ImVec4(c.x, c.y, c.z, 1));
}
}  // namespace

uint64_t nowNs() { return (uint64_t)std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

void setThreadName(const char* name) {
    lane().name = name;
    tracy::SetThreadName(name);
#if PTX_NVTX
    nvtxNameOsThreadA((uint32_t)pthread_self(), name);
#endif
}

Scope::Scope(const char* n) : name(n), t0(nowNs()), rec(capturing.load(std::memory_order_relaxed)) {
#if PTX_NVTX
    if (*n) nvtxRangePushA(n);
#endif
    if (rec) lane().depth++;
}
Scope::~Scope() {
#if PTX_NVTX
    if (*name) nvtxRangePop();
#endif
    if (!rec) return;
    Lane& l = lane();
    l.depth--;
    if (!capturing.load(std::memory_order_relaxed)) return;
    uint64_t t1 = nowNs();
    std::lock_guard lk(l.mu);
    l.ev.push_back({name, t0, t1, (uint16_t)std::max(l.depth, 0)});
}

void setCaptureFrames(int n) { wantFrames = std::max(1, n); }
bool hasCapture() { return !snap.lanes.empty(); }
std::string exportTrace() { return exportJson(); }

void toggleCapture() {
    if (capturing) { framesLeft = 0; finishCapture(); }
    else pendingStart = true;
}

void frameBegin() {
    if (capturing && --framesLeft <= 0) finishCapture();
    if (pendingStart) {
        pendingStart = false;
        std::lock_guard lk(lanesMu);
        for (Lane* l : lanes) { std::lock_guard lk2(l->mu); l->ev.clear(); }
        frameStarts.clear();
        gpuEvs.clear();
        capStart = nowNs();
        framesLeft = wantFrames;
        capturing = true;
    }
    if (capturing) frameStarts.push_back(nowNs());
}

void gpuFrame(const std::vector<GpuPass>& passes) {
    // last frame's passes, laid from that frame's CPU start (the GPU clock isn't the CPU's: alignment is approximate)
    if (!capturing || frameStarts.size() < 2) return;
    uint64_t base = frameStarts[frameStarts.size() - 2];
    for (auto& p : passes) gpuEvs.push_back({p.name, p.async, base + (uint64_t)(p.startMs * 1e6), base + (uint64_t)((p.startMs + p.ms) * 1e6)});
}

float slowFps() { return slowMo > 0.f && (!slowDuring || capturing) ? slowMo : 0.f; }

void ui(bool* open) {
    ImGui::SetNextWindowSize(ImVec2(1100, 520), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Profiler (F9 capture)", open)) { ImGui::End(); return; }
    ImGui::SetNextItemWidth(120);
    ImGui::SliderInt("##frames", &wantFrames, 1, 240, "%d frames");
    ImGui::SameLine();
    if (ImGui::Button(capturing ? "Stop" : "Capture")) toggleCapture();
    ImGui::SameLine();
    ImGui::SetNextItemWidth(140);
    ImGui::SliderFloat("##slow", &slowMo, 0.f, 60.f, slowMo > 0 ? "slow motion %.0f fps" : "slow motion off");
    ImGui::SetItemTooltip("Caps the frame rate (the main loop sleeps): everything still runs, so captures and Nsight / RenderDoc\n"
                          "hooks have time to keep up. Physics uses the real frame time, so it slows too.");
    ImGui::SameLine();
    ImGui::Checkbox("only while capturing", &slowDuring);
    ImGui::SameLine();
    ImGui::BeginDisabled(snap.lanes.empty());
    if (ImGui::Button("Export Chrome trace")) lastExport = exportJson();
    ImGui::EndDisabled();
    ImGui::SetItemTooltip("JSON for chrome://tracing or ui.perfetto.dev. For Nsight Systems run:\n"
                          "  nsys profile --trace=nvtx,vulkan,osrt ./build/ptx\n"
                          "(CPU zones are NVTX ranges, GPU passes are Vulkan debug labels)");
    if (!lastExport.empty()) { ImGui::SameLine(); ImGui::TextDisabled("%s", lastExport.c_str()); }
    if (capturing) ImGui::TextColored(ImVec4(1, 0.4f, 0.3f, 1), "capturing... %d frames left", framesLeft);
    if (snap.lanes.empty()) {
        ImGui::TextWrapped("Capture records every CPU zone on every thread and the GPU passes for the next N frames. "
                           "Wheel zooms the timeline, drag pans, double-click resets.");
        ImGui::End();
        return;
    }
    double spanMs = (snap.t1 - snap.t0) / 1e6;
    ImGui::Text("%zu frames, %.1f ms (%.2f ms/frame)", snap.frames.size(), spanMs, spanMs / std::max<size_t>(1, snap.frames.size()));

    if (ImGui::BeginTabBar("pt")) {
        if (ImGui::BeginTabItem("Flame chart")) {
            ImGui::BeginChild("tl", ImVec2(0, 0), ImGuiChildFlags_Borders, ImGuiWindowFlags_HorizontalScrollbar);
            ImDrawList* dl = ImGui::GetWindowDrawList();
            const float labelW = 110, rowH = 17;
            ImVec2 o = ImGui::GetCursorScreenPos();
            float W = std::max(100.f, ImGui::GetContentRegionAvail().x - labelW);
            double scale = W / std::max(v1 - v0, 1.0);
            auto X = [&](uint64_t t) { return o.x + labelW + (float)(((double)(t - snap.t0) - v0) * scale); };
            ImGuiIO& io = ImGui::GetIO();
            const Event* hovE = nullptr;
            const GpuEv* hovG = nullptr;
            float y = o.y + 16;
            // time ruler + frame lines
            double stepMs = pow(10.0, floor(log10(std::max((v1 - v0) / 1e6 / 8.0, 1e-3))));
            for (double ms = ceil(v0 / 1e6 / stepMs) * stepMs; ms * 1e6 < v1; ms += stepMs) {
                float x = o.x + labelW + (float)((ms * 1e6 - v0) * scale);
                dl->AddLine(ImVec2(x, o.y), ImVec2(x, o.y + 6), IM_COL32(160, 160, 160, 255));
                char b[32]; snprintf(b, sizeof b, "%.*f ms", stepMs < 1 ? 2 : 0, ms);
                dl->AddText(ImVec2(x + 2, o.y), IM_COL32(160, 160, 160, 255), b);
            }
            auto laneBar = [&](const char* label, int rows) {
                dl->AddText(ImVec2(o.x, y), IM_COL32(220, 220, 220, 255), label);
                dl->AddLine(ImVec2(o.x, y - 1), ImVec2(o.x + labelW + W, y - 1), IM_COL32(70, 70, 70, 255));
                float top = y;
                y += rows * rowH + 4;
                return top;
            };
            auto box = [&](float x0, float x1, float top, const char* name, ImU32 col) {
                x0 = std::max(x0, o.x + labelW); x1 = std::min(x1, o.x + labelW + W);
                if (x1 < x0) return false;
                if (x1 - x0 < 1) x1 = x0 + 1;
                dl->AddRectFilled(ImVec2(x0, top), ImVec2(x1, top + rowH - 1), col);
                if (x1 - x0 > 30) {
                    dl->PushClipRect(ImVec2(x0, top), ImVec2(x1, top + rowH), true);
                    dl->AddText(ImVec2(x0 + 2, top + 1), IM_COL32(20, 20, 20, 255), name);
                    dl->PopClipRect();
                }
                return io.MousePos.x >= x0 && io.MousePos.x < x1 && io.MousePos.y >= top && io.MousePos.y < top + rowH;
            };
            for (int q = 0; q < 2; q++) {
                float top = laneBar(q ? "GPU async" : "GPU", 1);
                for (auto& g : snap.gpu)
                    if (g.async == (q == 1) && box(X(g.t0), X(g.t1), top, g.name.c_str(), colorOf(g.name.c_str(), true))) hovG = &g;
            }
            for (auto& l : snap.lanes) {
                float top = laneBar(l.name.c_str(), l.maxDepth + 1);
                for (auto& e : l.ev)
                    if (box(X(e.t0), X(e.t1), top + e.depth * rowH, e.name, colorOf(e.name, false))) hovE = &e;
            }
            for (size_t i = 0; i < snap.frames.size(); i++) {
                float x = X(snap.frames[i]);
                if (x >= o.x + labelW && x <= o.x + labelW + W) dl->AddLine(ImVec2(x, o.y + 14), ImVec2(x, y), IM_COL32(255, 255, 255, 50));
            }
            ImGui::Dummy(ImVec2(labelW + W, y - o.y));
            if (ImGui::IsWindowHovered()) {
                double mt = v0 + (io.MousePos.x - o.x - labelW) / scale;
                if (io.MouseWheel != 0) {
                    double f = io.MouseWheel > 0 ? 0.8 : 1.25;
                    v0 = mt - (mt - v0) * f;
                    v1 = mt + (v1 - mt) * f;
                }
                if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 0)) {
                    double d = io.MouseDelta.x / scale;
                    v0 -= d; v1 -= d;
                }
                if (ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) { v0 = 0; v1 = (double)(snap.t1 - snap.t0); }
                if (hovE) ImGui::SetTooltip("%s\n%.3f ms", hovE->name, (hovE->t1 - hovE->t0) / 1e6);
                else if (hovG) ImGui::SetTooltip("[GPU%s] %s\n%.3f ms", hovG->async ? " async" : "", hovG->name.c_str(), (hovG->t1 - hovG->t0) / 1e6);
            }
            ImGui::EndChild();
            ImGui::EndTabItem();
        }
        if (ImGui::BeginTabItem("Totals")) {
            if (ImGui::BeginTable("agg", 5, ImGuiTableFlags_RowBg | ImGuiTableFlags_ScrollY | ImGuiTableFlags_Borders | ImGuiTableFlags_Resizable)) {
                ImGui::TableSetupColumn("zone");
                ImGui::TableSetupColumn("self ms / frame");
                ImGui::TableSetupColumn("total ms / frame");
                ImGui::TableSetupColumn("calls / frame");
                ImGui::TableSetupColumn("max ms");
                ImGui::TableHeadersRow();
                float nf = (float)std::max<size_t>(1, snap.frames.size());
                for (auto& a : snap.agg) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn(); ImGui::TextUnformatted(a.name.c_str());
                    ImGui::TableNextColumn(); ImGui::Text("%.3f", a.self / nf);
                    ImGui::TableNextColumn(); ImGui::Text("%.3f", a.total / nf);
                    ImGui::TableNextColumn(); ImGui::Text("%.1f", a.calls / nf);
                    ImGui::TableNextColumn(); ImGui::Text("%.3f", a.maxMs);
                }
                ImGui::EndTable();
            }
            ImGui::EndTabItem();
        }
        ImGui::EndTabBar();
    }
    ImGui::End();
}
}  // namespace prof

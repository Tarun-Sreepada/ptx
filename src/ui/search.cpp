#include "ui/search.h"
#include <cctype>
#include <cstdarg>
#include <cstdio>
#include <cstring>
#include <unordered_map>
#include <vector>

namespace ui {
namespace {
char query[128] = "";
bool active = false;   // between beginSearch / endSearch (the main panel only)
bool last = true;      // the previous widget was drawn
bool pendingSame = false;
float sameOffset = 0.f, sameSpacing = -1.f;
// a section (frames[0]) and the tree nodes open inside it. all: its title matched, everything in it shows;
// dry: nothing in it matched last frame, so it runs without drawing to find out whether something does now
struct Frame { ImGuiID id; bool all, dry, any, drawn; };
std::vector<Frame> frames;
std::unordered_map<ImGuiID, bool> had;  // section / node -> something in it matched (last time it ran)
// open / closed as the user left them before a search forced them open: put back once the search is cleared
std::unordered_map<ImGuiID, bool> userOpen;

void forceOpen(ImGuiID id) {
    if (!userOpen.count(id)) userOpen[id] = ImGui::GetStateStorage()->GetInt(id, 0) != 0;
    ImGui::SetNextItemOpen(true);
}
// no search: a section the search opened goes back to how it was
void restoreOpen(ImGuiID id) {
    auto it = userOpen.find(id);
    if (it == userOpen.end()) return;
    ImGui::SetNextItemOpen(it->second);
    userOpen.erase(it);
}

bool matches(const char* label) {
    if (!label) return false;
    size_t n = strlen(query);
    for (const char* s = label; *s && !(s[0] == '#' && s[1] == '#'); s++) {
        size_t i = 0;
        while (i < n && s[i] && tolower((unsigned char)s[i]) == tolower((unsigned char)query[i])) i++;
        if (i == n) return true;
    }
    return false;
}

void endHeader() {
    if (!frames.empty()) had[frames[0].id] = frames[0].any;
    frames.clear();
}

Frame* top() { return frames.empty() ? nullptr : &frames.back(); }
}  // namespace

bool searching() { return active && query[0]; }
bool lastShown() { return last; }

void beginSearch() {
    if (ImGui::GetIO().KeyCtrl && ImGui::IsKeyPressed(ImGuiKey_F)) ImGui::SetKeyboardFocusHere();
    ImGui::SetNextItemWidth(-60.f);
    ImGui::InputTextWithHint("##search", "Search settings (Ctrl+F)", query, sizeof query);
    ImGui::SameLine();
    if (ImGui::Button("Clear")) query[0] = 0;
    active = true;
    last = true;
    pendingSame = false;
    frames.clear();
}

void endSearch() {
    endHeader();
    active = false;
    last = true;
}

bool show(const char* label) {
    bool same = pendingSame;
    pendingSame = false;
    if (!searching()) return last = true;
    Frame* f = top();
    bool m = matches(label) || (f && f->all);
    if (m) for (Frame& fr : frames) fr.any = true;
    if (!m || (f && f->dry)) return last = false;
    if (same) ImGui::SameLine(sameOffset, sameSpacing);
    return last = true;
}

bool showText(const char* fmt, va_list args) {
    if (!searching()) { pendingSame = false; return last = true; }
    char buf[512];
    vsnprintf(buf, sizeof buf, fmt, args);
    return show(buf);
}

bool CollapsingHeader(const char* label, ImGuiTreeNodeFlags flags) {
    if (!searching()) { last = true; restoreOpen(ImGui::GetID(label)); return ImGui::CollapsingHeader(label, flags); }
    endHeader();
    ImGuiID id = ImGui::GetID(label);
    bool all = matches(label), drawn = all || had[id];
    frames.push_back({id, all, !drawn, all, drawn});
    if (drawn) {
        forceOpen(id);
        ImGui::CollapsingHeader(label, flags);
    }
    last = drawn;
    return true;  // the contents always run: dry ones only to look for matches
}

bool TreeNode(const char* label) {
    if (!searching() || frames.empty()) { last = true; restoreOpen(ImGui::GetID(label)); return ImGui::TreeNode(label); }
    Frame& p = frames.back();
    ImGuiID id = ImGui::GetID(label);
    bool all = matches(label) || p.all, drawn = !p.dry && (all || had[id]);
    if (all) for (Frame& fr : frames) fr.any = true;
    frames.push_back({id, all, !drawn, all, drawn});
    if (drawn) {
        forceOpen(id);
        ImGui::TreeNode(label);
    } else ImGui::PushID(label);  // the same ID scope either way
    last = drawn;
    return true;
}

void TreePop() {
    if (!searching() || frames.size() < 2) { ImGui::TreePop(); return; }
    Frame f = frames.back();
    frames.pop_back();
    had[f.id] = f.any;
    if (f.drawn) ImGui::TreePop();
    else ImGui::PopID();
}

void SameLine(float offset, float spacing) {
    if (!searching()) { ImGui::SameLine(offset, spacing); return; }
    // only if both this widget and the next one are drawn: wait for the next one
    pendingSame = last;
    sameOffset = offset;
    sameSpacing = spacing;
}

void Separator() { if (!searching()) ImGui::Separator(); }
void SeparatorText(const char* label) { if (!searching()) ImGui::SeparatorText(label); }
void Spacing() { if (!searching()) ImGui::Spacing(); }
void NewLine() { if (!searching()) ImGui::NewLine(); }
bool IsItemDeactivatedAfterEdit() { return (!searching() || last) && ImGui::IsItemDeactivatedAfterEdit(); }
bool IsItemHovered(ImGuiHoveredFlags flags) { return (!searching() || last) && ImGui::IsItemHovered(flags); }

void SetItemTooltip(const char* fmt, ...) {
    if (searching() && !last) return;
    va_list a;
    va_start(a, fmt);
    ImGui::SetItemTooltipV(fmt, a);
    va_end(a);
}

#define TEXT_WRAPPER(name, call)                         \
    void name(const char* fmt, ...) {                    \
        va_list a, b;                                    \
        va_start(a, fmt);                                \
        va_copy(b, a);                                   \
        if (showText(fmt, a)) call;                      \
        va_end(b);                                       \
        va_end(a);                                       \
    }
TEXT_WRAPPER(Text, ImGui::TextV(fmt, b))
TEXT_WRAPPER(TextDisabled, ImGui::TextDisabledV(fmt, b))
TEXT_WRAPPER(TextWrapped, ImGui::TextWrappedV(fmt, b))
TEXT_WRAPPER(BulletText, ImGui::BulletTextV(fmt, b))

void TextColored(const ImVec4& col, const char* fmt, ...) {
    va_list a, b;
    va_start(a, fmt);
    va_copy(b, a);
    if (showText(fmt, a)) ImGui::TextColoredV(col, fmt, b);
    va_end(b);
    va_end(a);
}

void ProgressBar(float fraction, const ImVec2& size, const char* overlay) {
    if (show(overlay ? overlay : "")) ImGui::ProgressBar(fraction, size, overlay);
}
}  // namespace ui

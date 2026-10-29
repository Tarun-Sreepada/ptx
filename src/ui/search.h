// Search box for the main panel: the panels call these instead of ImGui:: directly. With no query they are plain
// ImGui; with one, only widgets whose label contains it (case-insensitive) are drawn, inside their sections, which
// open by themselves. A section or tree node whose own title matches shows everything in it. A section learns
// whether anything in it matches by running its contents without drawing (one frame later it shows), so sections
// with no match stay out of the way.
#pragma once
#include <imgui.h>

namespace ui {
void beginSearch();  // top of the panel: the search box
void endSearch();    // end of the panel
bool show(const char* label);  // draw this widget? (also records a match for its section)
bool lastShown();              // the previous widget was drawn (for tooltips, SameLine)
bool searching();
bool showText(const char* fmt, va_list args);

bool CollapsingHeader(const char* label, ImGuiTreeNodeFlags flags = 0);
bool TreeNode(const char* label);
void TreePop();
void SameLine(float offset = 0.f, float spacing = -1.f);
void Separator();
void SeparatorText(const char* label);
void Spacing();
void NewLine();
bool IsItemDeactivatedAfterEdit();
bool IsItemHovered(ImGuiHoveredFlags flags = 0);
void SetItemTooltip(const char* fmt, ...) IM_FMTARGS(1);
void Text(const char* fmt, ...) IM_FMTARGS(1);
void TextDisabled(const char* fmt, ...) IM_FMTARGS(1);
void TextWrapped(const char* fmt, ...) IM_FMTARGS(1);
void BulletText(const char* fmt, ...) IM_FMTARGS(1);
void TextColored(const ImVec4& col, const char* fmt, ...) IM_FMTARGS(2);
void ProgressBar(float fraction, const ImVec2& size = ImVec2(-FLT_MIN, 0), const char* overlay = nullptr);

// widgets: drawn when their label matches
template <class... A> bool SliderFloat(const char* l, A... a) { return show(l) && ImGui::SliderFloat(l, a...); }
template <class... A> bool SliderInt(const char* l, A... a) { return show(l) && ImGui::SliderInt(l, a...); }
template <class... A> bool Checkbox(const char* l, A... a) { return show(l) && ImGui::Checkbox(l, a...); }
template <class... A> bool Combo(const char* l, A... a) { return show(l) && ImGui::Combo(l, a...); }
template <class... A> bool BeginCombo(const char* l, A... a) { return show(l) && ImGui::BeginCombo(l, a...); }
template <class... A> bool ColorEdit3(const char* l, A... a) { return show(l) && ImGui::ColorEdit3(l, a...); }
template <class... A> bool InputInt(const char* l, A... a) { return show(l) && ImGui::InputInt(l, a...); }
template <class... A> bool InputText(const char* l, A... a) { return show(l) && ImGui::InputText(l, a...); }
inline bool Button(const char* l, const ImVec2& size = ImVec2(0, 0)) { return show(l) && ImGui::Button(l, size); }
template <class... A> bool RadioButton(const char* l, A... a) { return show(l) && ImGui::RadioButton(l, a...); }
template <class... A> bool BeginTable(const char* l, A... a) { return show(l) && ImGui::BeginTable(l, a...); }
}  // namespace ui

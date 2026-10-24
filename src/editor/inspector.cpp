// Selection operations (delete, duplicate), outlines, and the inspector panel.
#include "editor/editor_internal.h"

namespace editor {

void deleteSelected() {
    ECell* c = selCell();
    if (!c || !selected()) return;
    pushUndo(*c);
    c->items.erase(c->items.begin() + sel.idx);
    sel.idx = -1;
    changed(*c, true);
}

void duplicateSelected() {
    ECell* c = selCell();
    Item* it = selected();
    if (!c || !it) return;
    Item cp = *it;
    float yaw = cp.rot[1] * DEG, step = std::max(grid, 0.25f);
    float w = cp.kind == K_PRIM ? cp.size[0] : 1.f;  // next to it along its own x
    cp.pos[0] += cosf(yaw) * std::max(w, step);
    cp.pos[2] -= sinf(yaw) * std::max(w, step);
    cp.slot = -1;
    pushUndo(*c);
    c->items.push_back(cp);
    sel.idx = (int)c->items.size() - 1;
    changed(*c, true);
    rehome(*c, sel.idx);
}

void drawBox(ImDrawList* dl, const ECell& c, const Item& it, ImU32 col, float thick) {
    float mn[3], mx[3];
    localBounds(c, it, mn, mx);
    Xf x = boundsXf(it);
    ImVec2 s[8];
    bool ok[8];
    for (int i = 0; i < 8; i++) {
        float l[3] = {i & 1 ? mx[0] : mn[0], i & 2 ? mx[1] : mn[1], i & 4 ? mx[2] : mn[2]}, w[3];
        xfPoint(x, l, w);
        ok[i] = project(w, s[i]);
    }
    const int e[12][2] = {{0, 1}, {2, 3}, {4, 5}, {6, 7}, {0, 2}, {1, 3}, {4, 6}, {5, 7}, {0, 4}, {1, 5}, {2, 6}, {3, 7}};
    for (auto& k : e)
        if (ok[k[0]] && ok[k[1]]) dl->AddLine(s[k[0]], s[k[1]], col, thick);
}

bool matCombo(const char* label, std::string& mat, bool allowNone) {
    bool ch = false;
    const matlib::Entry* cur = matlib::find(mat);
    if (cur) {
        ImGui::ColorButton("##sw", ImVec4(cur->swatch[0], cur->swatch[1], cur->swatch[2], 1), ImGuiColorEditFlags_NoTooltip, ImVec2(18, 18));
        ImGui::SameLine();
    }
    if (ImGui::BeginCombo(label, mat.empty() ? "(none)" : mat.c_str())) {
        if (allowNone && ImGui::Selectable("(none)", mat.empty())) { mat.clear(); ch = true; }
        for (auto& e : matlib::list()) {
            ImGui::ColorButton(("##" + e.name).c_str(), ImVec4(e.swatch[0], e.swatch[1], e.swatch[2], 1), ImGuiColorEditFlags_NoTooltip, ImVec2(16, 16));
            ImGui::SameLine();
            if (ImGui::Selectable(e.name.c_str(), e.name == mat)) { mat = e.name; ch = true; }
        }
        ImGui::EndCombo();
    }
    return ch;
}

void inspector() {
    ECell* c = selCell();
    Item* it = selected();
    if (!c || !it) { ImGui::TextDisabled("nothing selected (click something)"); return; }
    ImGui::Text("%s in cell %d, %d", KIND_NAME[it->kind], c->cx, c->cz);
    char nm[128];
    snprintf(nm, sizeof nm, "%s", it->name.c_str());
    if (ImGui::InputText("Name", nm, sizeof nm)) { it->name = nm; changed(*c, true); }
    bool begin = false, move = false, geom = false;
    auto track = [&](bool edited, bool isGeom) {
        begin |= ImGui::IsItemActivated();
        if (edited) (isGeom ? geom : move) = true;
    };
    track(ImGui::DragFloat3("Position", it->pos, 0.05f), false);
    if (it->kind != K_LIGHT && it->kind != K_VOLUME && it->kind != K_ZONE) {
        if (it->kind == K_TREE || it->kind == K_ROCK) track(ImGui::DragFloat("Yaw", &it->rot[1], 1.f), false);
        else track(ImGui::DragFloat3("Rotation (x y z)", it->rot, 1.f), false);
    }
    if (it->kind == K_MESH || it->kind == K_ASSET || it->kind == K_VOLUME || it->kind == K_ZONE)
        track(ImGui::DragFloat3(it->kind == K_VOLUME || it->kind == K_ZONE ? "Half size" : "Scale", it->scale, 0.01f, 0.01f, 1000.f), false);
    if (it->kind == K_TREE || it->kind == K_ROCK) track(ImGui::DragFloat("Scale", &it->scale[0], 0.01f, 0.1f, 10.f), false);
    if (it->kind == K_PRIM) {
        int sh = it->shape;
        if (ImGui::Combo("Shape", &sh, "Box\0Cylinder\0")) { begin = true; it->shape = sh; geom = true; }
        track(ImGui::DragFloat3(it->shape == BOX ? "Size (w h d)" : "Size (dx h dz)", it->size, 0.01f, 0.02f, 200.f), true);
        if (it->shape == CYLINDER) track(ImGui::SliderInt("Sides", &it->sides, 3, 64), true);
        if (matCombo("Material", it->mat)) {
            begin = true; geom = true;
            if (auto* e = matlib::find(it->mat)) it->strength = e->strength;
        }
        track(ImGui::DragFloat("Texture size (m)", &it->uvScale, 0.01f, 0.05f, 20.f), true);
        track(ImGui::ColorEdit3("Tint", it->tint), true);
    }
    if (it->kind == K_PRIM || it->kind == K_MESH) {
        track(ImGui::DragFloat("Strength", &it->strength, 0.02f, 0.02f, 100.f, "%.2f x brick"), true);
        ImGui::SameLine();
        ImGui::TextDisabled("(?)");
        if (ImGui::IsItemHovered()) ImGui::SetTooltip("How hard it is to breach: energy needed scales with this (brick 1, concrete 2, metal 5, planks 0.4)");
    }
    if (it->kind == K_PRIM || it->kind == K_MESH || it->kind == K_ASSET) {
        bool col = it->collide;
        if (ImGui::Checkbox("Collide / breakable", &col)) { begin = true; it->collide = col; geom = true; }
    }
    if (it->kind == K_ASSET) ImGui::Text("Asset: %s", it->asset.c_str());
    if (it->kind == K_LIGHT) {
        track(ImGui::ColorEdit3("Colour", it->color), true);
        track(ImGui::DragFloat("Intensity (cd)", &it->intensity, 2.f, 0.f, 100000.f, "%.0f", ImGuiSliderFlags_Logarithmic), true);
        if (ImGui::IsItemHovered())
            ImGui::SetTooltip("Physical units: a 60 W bulb is ~70 cd, a bright LED lamp ~150, a street light ~2000.\n"
                              "Next to daylight (~100000 lux) any lamp is invisible, as in reality: test at night or indoors.");
    }
    if (it->kind == K_ZONE) {
        track(ImGui::DragFloat("Falloff (m)", &it->falloff, 0.1f, 0.f, 100.f), true);
        if (matCombo("Ground texture", it->zoneMat, true)) { begin = true; geom = true; }
        if (!it->zoneMat.empty()) track(ImGui::DragFloat("Texture size (m)##zt", &it->zoneTile, 0.02f, 0.1f, 50.f), true);
    }
    if (it->kind == K_TREE) {
        int v = it->variant;
        if (ImGui::Combo("Variant", &v, TREE_NAME, trees::VARIANTS)) { begin = true; it->variant = v; geom = true; }
    }
    if (begin) {  // snapshot taken before this edit started
        std::vector<Item> snap = c->items;
        c->undo.push_back(snap);
    }
    if (geom) changed(*c, true);
    else if (move) changed(*c, false);
    if (ImGui::Button("Duplicate (Ctrl+D)")) duplicateSelected();
    ImGui::SameLine();
    if (ImGui::Button("Drop to ground")) {
        pushUndo(*c);
        snapToGround(*c, *it);
        changed(*c, false);
    }
    ImGui::SameLine();
    if (ImGui::Button("Delete (Del)")) deleteSelected();
}

}  // namespace editor

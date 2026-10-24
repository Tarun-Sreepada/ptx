#include "editor/editor_internal.h"

namespace editor {

void init(const Host& h) {
    host = h;
    matlib::init(h.worldDir);
    scanAssets();
}
bool active() { return on; }
void setActive(bool v) { on = v; dragging = false; }
bool takeMoveOnly() { bool r = moveOnly; moveOnly = false; return r; }
bool wantsMouse() { return on; }

void demoHouse() {
    float p[3];
    placePoint(p);
    float yaw = camYawSnapped() * DEG;
    p[0] = snapv(p[0], 0.5f); p[2] = snapv(p[2], 0.5f);
    // a house in its own frame: x across (8 m), z deep (6 m), walls 0.25 thick, 3 m storey; front at -z
    const float W = 8, D = 6, H = 3, T = 0.25f;
    std::vector<Item> parts;
    auto put = [&](Item it, float x, float y, float z) {
        it.pos[0] = p[0] + cosf(yaw) * x + sinf(yaw) * z;
        it.pos[2] = p[2] - sinf(yaw) * x + cosf(yaw) * z;
        it.pos[1] = p[1] + y;
        it.rot[1] += yaw / DEG;
        parts.push_back(it);
    };
    put(prim(BOX, W + 0.6f, 0.3f, D + 0.6f, "concrete", "slab"), 0, -0.1f, 0);
    put(prim(BOX, W, H, T, "brick", "back wall"), 0, 0.2f, D * 0.5f - T * 0.5f);
    // side walls with a window each: below, above, either side of a 1.2 x 1.2 m opening
    for (int s = -1; s <= 1; s += 2) {
        float x = s * (W * 0.5f - T * 0.5f), d = D - 2 * T;
        Item lo = prim(BOX, d, 1.0f, T, "brick", "side wall"); lo.rot[1] = 90;
        put(lo, x, 0.2f, 0);
        Item hi = prim(BOX, d, H - 2.2f, T, "brick", "side wall"); hi.rot[1] = 90;
        put(hi, x, 2.4f, 0);
        Item a = prim(BOX, (d - 1.2f) * 0.5f, 1.2f, T, "brick", "side wall"); a.rot[1] = 90;
        put(a, x, 1.2f, -(1.2f + (d - 1.2f) * 0.5f) * 0.5f);
        put(a, x, 1.2f, (1.2f + (d - 1.2f) * 0.5f) * 0.5f);
    }
    // front: door 1.1 x 2.2 in the middle, lintel over it
    float side = (W - 1.1f) * 0.5f;
    put(prim(BOX, side, H, T, "brick", "front wall"), -(1.1f + side) * 0.5f, 0.2f, -D * 0.5f + T * 0.5f);
    put(prim(BOX, side, H, T, "brick", "front wall"), (1.1f + side) * 0.5f, 0.2f, -D * 0.5f + T * 0.5f);
    put(prim(BOX, 1.1f, H - 2.2f, T, "concrete", "lintel"), 0, 2.4f, -D * 0.5f + T * 0.5f);
    for (int sx = -1; sx <= 1; sx += 2)  // porch posts
        put(prim(CYLINDER, 0.3f, H, 0.3f, "planks", "post"), sx * 1.2f, 0.2f, -D * 0.5f - 1.4f);
    put(prim(BOX, W + 0.8f, 0.25f, D + 2.4f, "planks", "roof"), 0, 0.2f + H, -0.8f);
    Item L;
    L.kind = K_LIGHT;
    L.intensity = 400;
    put(L, 0, 2.5f, 0.5f);
    Item V;
    V.kind = K_VOLUME;
    V.name = fmt("house_%d", (int)now() % 100000);
    V.scale[0] = W * 0.5f; V.scale[1] = H * 0.5f + 0.2f; V.scale[2] = D * 0.5f;
    put(V, 0, H * 0.5f + 0.2f, 0);
    parts.back().scale[0] = fabsf(cosf(yaw)) * W * 0.5f + fabsf(sinf(yaw)) * D * 0.5f;  // axis-aligned box around it
    parts.back().scale[2] = fabsf(sinf(yaw)) * W * 0.5f + fabsf(cosf(yaw)) * D * 0.5f;
    parts.back().rot[1] = 0;
    ECell* c = open(cellOf(p[0]), cellOf(p[2]));
    if (!c) { status = "that cell is still loading"; return; }
    pushUndo(*c);
    for (auto& it : parts) c->items.push_back(it);
    changed(*c, true);
    sel = {c->cx, c->cz, (int)c->items.size() - (int)parts.size() + 1};
    status = "example house added";
}

bool saveAll(std::string& err) {
    int n = 0;
    for (auto& [k, c] : cells) {
        if (!c.dirty) continue;
        std::string path = cellPath(c.cx, c.cz);
        std::error_code ec;
        std::filesystem::create_directories(host.worldDir + "cells", ec);
        if (std::filesystem::exists(path) && !std::filesystem::exists(path + ".bak")) std::filesystem::copy_file(path, path + ".bak", ec);
        ModelData d;
        build(c, d, false);
        if (!writeGlb(path, d, err)) return false;
        c.dirty = false;
        n++;
    }
    status = fmt("saved %d cell%s", n, n == 1 ? "" : "s");
    if (worldmap::dirty()) {
        if (!worldmap::save(err)) return false;
        status += ", world map";
    }
    return true;
}

void apply() {
    for (auto [cx, cz] : reverts) {  // the saved file (or nothing) back into the stream, edits dropped
        auto m = std::make_shared<Model>();
        std::string err, path = cellPath(cx, cz);
        if (std::filesystem::exists(path) && !loadGltf(path, m->data, err)) { status = "revert failed: " + err; continue; }
        std::vector<bool> coll(m->data.meshes.size(), false);
        for (auto& in : m->data.instances) if (in.mesh >= 0 && in.collide) coll[in.mesh] = true;
        for (size_t i = 0; i < m->data.meshes.size(); i++) {
            auto& md = m->data.meshes[i];
            if (coll[i] && !md.idx.empty()) md.shape = physics::meshShape(md.verts.data(), 8, (uint32_t)(md.verts.size() / 8), md.idx.data(), (uint32_t)(md.idx.size() / 3));
        }
        m->state = 1;
        if (!host.upload(*m)) { status = "revert: upload failed"; continue; }
        std::vector<std::shared_ptr<Model>> freed;
        stream::setCellModel(cx, cz, m, freed);
        for (auto& f : freed) freeModel(f);
        cells.erase({cx, cz});
        if (sel.cx == cx && sel.cz == cz) sel.idx = -1;
        status = "cell reverted to the saved file";
    }
    reverts.clear();
    if (cells.empty()) return;
    bool editing = ImGui::GetCurrentContext() && ImGui::IsAnyItemActive();
    for (auto& [k, c] : cells) {
        auto live = stream::cellModel(c.cx, c.cz);
        // the stream reloaded the cell from disk under unsaved edits: put them back
        if (c.modified && live && live != c.live) c.rebuild = true;
        if (c.rebuild) {
            if (editing && now() - lastRebuild < 0.1) continue;  // slider drags: ~10 rebuilds a second
            c.rebuild = c.refresh = false;
            pushLive(c);
        } else if (c.refresh) {
            c.refresh = false;
            Item* d = dragging ? selected() : nullptr;
            moveOnly = d && (d->kind == K_PRIM || d->kind == K_MESH || d->kind == K_ASSET || d->kind == K_LIGHT);
            bool ok = true;
            for (auto& it : c.items) ok &= it.slot >= 0 ? moveInPlace(c, it) : true;
            if (ok) stream::refreshCell(c.cx, c.cz);
            else pushLive(c);
        }
    }
}

void ui(const View& v) {
    view = v;
    if (!on) return;
    pollJobs();
    ImGuiIO& io = ImGui::GetIO();
    // keep the cells around the camera open for picking
    int ccx = cellOf(v.pos[0]), ccz = cellOf(v.pos[2]);
    for (int dz = -1; dz <= 1; dz++)
        for (int dx = -1; dx <= 1; dx++) open(ccx + dx, ccz + dz);

    mapTools();
    // ---- mouse: pick, drag
    float o[3] = {v.pos[0], v.pos[1], v.pos[2]}, d[3];
    mouseRay(d);
    if (!mapMode && !io.WantCaptureMouse && ImGui::IsMouseClicked(ImGuiMouseButton_Left)) {
        Hit h = pick(o, d);
        sel = {h.cx, h.cz, h.idx};
        if (Item* it = selected()) {
            dragging = true;
            dragMoved = false;
            dragVertical = io.KeyAlt;
            float hp[3] = {o[0] + d[0] * h.t, o[1] + d[1] * h.t, o[2] + d[2] * h.t};
            memcpy(dragP0, hp, 12);
            for (int k = 0; k < 3; k++) dragOff[k] = it->pos[k] - hp[k];
            dragPlaneY = hp[1];
            float n[3] = {-v.fwd[0], 0, -v.fwd[2]}, l = sqrtf(n[0] * n[0] + n[2] * n[2]);
            dragN[0] = n[0] / std::max(l, 1e-6f); dragN[1] = 0; dragN[2] = n[2] / std::max(l, 1e-6f);
        }
    }
    snapInvert = io.KeyShift;
    if (!io.WantCaptureKeyboard && ImGui::IsKeyPressed(ImGuiKey_N, false)) snap = !snap;
    if (dragging) {
        Item* it = selected();
        ECell* c = selCell();
        if (!it || !c || !ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
            dragging = false;
            if (it && c && dragMoved) { changed(*c, false); rehome(*c, sel.idx); }  // full refresh once the drag ends
        } else if (ImGui::IsMouseDragging(ImGuiMouseButton_Left, 1.f)) {
            float t;
            float p[3];
            if (dragVertical) {  // plane through the grab point facing the camera, vertical motion only
                float den = d[0] * dragN[0] + d[2] * dragN[2];
                t = fabsf(den) > 1e-6f ? ((dragP0[0] - o[0]) * dragN[0] + (dragP0[2] - o[2]) * dragN[2]) / den : -1;
                if (t > 0) {
                    float y = o[1] + d[1] * t + dragOff[1];
                    if (!dragMoved) pushUndo(*c);
                    it->pos[1] = snapv(y, grid);
                    dragMoved = true;
                    changed(*c, false);
                }
            } else if (fabsf(d[1]) > 1e-4f && (t = (dragPlaneY - o[1]) / d[1]) > 0 && t < 2000.f) {
                for (int k = 0; k < 3; k++) p[k] = o[k] + d[k] * t;
                if (!dragMoved) pushUndo(*c);
                it->pos[0] = snapv(p[0] + dragOff[0], grid);
                it->pos[2] = snapv(p[2] + dragOff[2], grid);
                if (groundSnap && groundable(*it)) snapToGround(*c, *it);
                dragMoved = true;
                changed(*c, false);
            }
        }
    }

    // ---- keys
    if (!io.WantCaptureKeyboard) {
        bool ctrl = io.KeyCtrl;
        if (ImGui::IsKeyPressed(ImGuiKey_Delete)) deleteSelected();
        if (ctrl && ImGui::IsKeyPressed(ImGuiKey_D, false)) duplicateSelected();
        if (ctrl && ImGui::IsKeyPressed(ImGuiKey_S, false)) { std::string err; if (!saveAll(err)) status = "save failed: " + err; }
        if (!mapMode && ctrl && ImGui::IsKeyPressed(ImGuiKey_Z, false))
            if (ECell* c = selCell(); c && !c->undo.empty()) {
                c->items = c->undo.back();
                c->undo.pop_back();
                if (sel.idx >= (int)c->items.size()) sel.idx = -1;
                changed(*c, true);
            }
        Item* it = selected();
        ECell* c = selCell();
        if (it && c) {
            float mv[3] = {0, 0, 0};
            float g = std::max(grid, 0.05f);
            if (ImGui::IsKeyPressed(ImGuiKey_LeftArrow)) mv[0] -= g;
            if (ImGui::IsKeyPressed(ImGuiKey_RightArrow)) mv[0] += g;
            if (ImGui::IsKeyPressed(ImGuiKey_UpArrow)) mv[2] += g;
            if (ImGui::IsKeyPressed(ImGuiKey_DownArrow)) mv[2] -= g;
            if (ImGui::IsKeyPressed(ImGuiKey_PageUp)) mv[1] += g;
            if (ImGui::IsKeyPressed(ImGuiKey_PageDown)) mv[1] -= g;
            bool rot = !ctrl && ImGui::IsKeyPressed(ImGuiKey_R, false);
            if (mv[0] || mv[1] || mv[2] || rot) {
                pushUndo(*c);
                for (int k = 0; k < 3; k++) it->pos[k] += mv[k];
                if (rot) it->rot[1] = fmodf(it->rot[1] + (io.KeyShift ? -rotStep : rotStep) + 360.f, 360.f);
                changed(*c, false);
                if (mv[0] || mv[2]) rehome(*c, sel.idx);
            }
        }
    }

    // ---- overlay
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    for (auto& [k, c] : cells)
        for (size_t i = 0; i < c.items.size(); i++) {
            const Item& it = c.items[i];
            bool isSel = k.first == sel.cx && k.second == sel.cz && (int)i == sel.idx;
            if (isSel) continue;
            if (it.kind == K_LIGHT) {
                ImVec2 s;
                if (project(it.pos, s)) dl->AddCircle(s, 6.f, IM_COL32(255, 220, 120, 200), 0, 2.f);
            } else if ((it.kind == K_VOLUME && showVolumes) ) drawBox(dl, c, it, IM_COL32(80, 200, 255, 110), 1.f);
            else if (it.kind == K_ZONE && showVolumes) drawBox(dl, c, it, IM_COL32(120, 255, 120, 110), 1.f);
            else if (showAll) drawBox(dl, c, it, IM_COL32(255, 255, 255, 50), 1.f);
        }
    if (Item* it = selected()) drawBox(dl, *selCell(), *it, IM_COL32(255, 210, 40, 255), 2.f);
    {   // placement point
        ImVec2 c(io.DisplaySize.x * 0.5f, io.DisplaySize.y * 0.5f);
        dl->AddLine(ImVec2(c.x - 6, c.y), ImVec2(c.x + 6, c.y), IM_COL32(255, 255, 255, 160));
        dl->AddLine(ImVec2(c.x, c.y - 6), ImVec2(c.x, c.y + 6), IM_COL32(255, 255, 255, 160));
    }

    // ---- panel
    ImGui::SetNextWindowPos(ImVec2(io.DisplaySize.x - 400, 20), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(380, 720), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSizeConstraints(ImVec2(250, 150), ImVec2(FLT_MAX, std::max(200.f, io.DisplaySize.y - 20.f)));  // scrolls, never off-screen
    ImGui::Begin("World editor (Tab)");
    int dirty = 0;
    for (auto& [k, c] : cells) dirty += c.dirty;
    if (ImGui::Button(dirty ? "Save (Ctrl+S) *" : "Save (Ctrl+S)")) { std::string err; if (!saveAll(err)) status = "save failed: " + err; }
    ImGui::SameLine();
    ImGui::TextDisabled("%d unsaved cell%s", dirty, dirty == 1 ? "" : "s");
    if (!status.empty()) ImGui::TextWrapped("%s", status.c_str());
    ImGui::Checkbox("Snap (N)", &snap);
    ImGui::SetItemTooltip("Hold Shift while dragging to invert: free movement with snap on, grid steps with it off");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(80);
    ImGui::DragFloat("Grid (m)", &grid, 0.01f, 0.01f, 10.f, "%.2f");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(60);
    ImGui::DragFloat("R step", &rotStep, 1.f, 1.f, 90.f, "%.0f");
    ImGui::Checkbox("Show volumes / zones", &showVolumes);
    ImGui::SameLine();
    ImGui::Checkbox("Show all bounds", &showAll);

    if (ImGui::CollapsingHeader("Add (at the crosshair)", ImGuiTreeNodeFlags_DefaultOpen)) {
        matCombo("New material", newMat);
        if (ImGui::Button("Wall")) addItem(prim(BOX, 4, 3, 0.25f, newMat, "wall"));
        ImGui::SameLine();
        if (ImGui::Button("Floor")) addItem(prim(BOX, 4, 0.2f, 4, newMat, "floor"));
        ImGui::SameLine();
        if (ImGui::Button("Box")) addItem(prim(BOX, 1, 1, 1, newMat, "box"));
        ImGui::SameLine();
        if (ImGui::Button("Beam")) addItem(prim(BOX, 4, 0.3f, 0.3f, newMat, "beam"));
        if (ImGui::Button("Pillar")) addItem(prim(CYLINDER, 0.4f, 3, 0.4f, newMat, "pillar"));
        ImGui::SameLine();
        if (ImGui::Button("Cylinder")) addItem(prim(CYLINDER, 1, 1, 1, newMat, "cylinder"));
        ImGui::SameLine();
        if (ImGui::Button("Example house")) demoHouse();
        buildingPanel();
        if (ImGui::Button("Light")) { Item it; it.kind = K_LIGHT; it.name = "light"; addItem(it); }
        ImGui::SameLine();
        if (ImGui::Button("Bake volume")) {
            Item it; it.kind = K_VOLUME; it.scale[0] = 4; it.scale[1] = 1.6f; it.scale[2] = 3;
            it.name = fmt("vol_%d", (int)now() % 100000);
            addItem(it);
        }
        ImGui::SameLine();
        if (ImGui::Button("Flatten zone")) { Item it; it.kind = K_ZONE; it.scale[0] = it.scale[2] = 10; it.scale[1] = 0.5f; addItem(it); }
        static int treeVar = 0;
        ImGui::SetNextItemWidth(130);
        ImGui::Combo("##tv", &treeVar, TREE_NAME, trees::VARIANTS);
        ImGui::SameLine();
        if (ImGui::Button("Tree")) { Item it; it.kind = K_TREE; it.variant = treeVar; it.rot[1] = (float)(rand() % 360); addItem(it); }
        ImGui::SameLine();
        if (ImGui::Button("Rock")) { Item it; it.kind = K_ROCK; it.rot[1] = (float)(rand() % 360); addItem(it); }
        ImGui::Checkbox("Asset browser", &showBrowser);
        ImGui::SameLine();
        ImGui::TextDisabled("(drop files / folders on the window to import)");
        if (ImGui::Button("Rescan assets + textures")) { scanAssets(); matlib::rescan(); }
    }
    if (ImGui::CollapsingHeader("Selection", ImGuiTreeNodeFlags_DefaultOpen)) inspector();
    if (ImGui::CollapsingHeader("Cell contents")) {
        ECell* c = open(ccx, ccz);
        if (!c) ImGui::TextDisabled("loading...");
        else {
            ImGui::Text("cell %d, %d: %zu items", c->cx, c->cz, c->items.size());
            ImGui::BeginChild("list", ImVec2(0, 200), true);
            for (size_t i = 0; i < c->items.size(); i++) {
                const Item& it = c->items[i];
                std::string label = fmt("%s  %s##%zu", KIND_NAME[it.kind], it.name.empty() ? (it.kind == K_ASSET ? it.asset.c_str() : "") : it.name.c_str(), i);
                if (ImGui::Selectable(label.c_str(), sel.cx == c->cx && sel.cz == c->cz && sel.idx == (int)i)) sel = {c->cx, c->cz, (int)i};
            }
            ImGui::EndChild();
            if (ImGui::Button("Revert cell to saved")) reverts.push_back({c->cx, c->cz});
        }
    }
    if (ImGui::CollapsingHeader("Help")) {
        ImGui::TextWrapped(
            "Click: select.  Drag: move on the ground plane (Alt+drag: up / down).\n"
            "R / Shift+R: rotate by the step.  Arrows, PgUp/PgDn: nudge by the grid.\n"
            "Ctrl+D duplicate, Del delete, Ctrl+Z undo, Ctrl+S save.\n"
            "Right mouse + WASD / Q E: fly.  G: throw a rock to test.\n"
            "Primitives are breakable: strength scales the energy a breach takes. Textures: built-in set, plus any "
            "PNG / JPG in world/textures/ (or dropped on the window). Drag assets from the Assets window into the world. Saved into world/cells/<x>_<z>.glb (old file kept as .bak).");
    }
    ImGui::End();
    if (showBrowser) assetBrowser();
    if (showScatter) scatterPanel();
    assetDrop();
}

void dropFiles(const std::vector<std::string>& paths) {
    if (host.worldDir.empty()) return;
    setActive(true);
    showBrowser = true;
    importPaths(paths);
}

}  // namespace editor

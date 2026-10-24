// Asset browser: folders and thumbnails, drag and drop into the world, rename / move / delete.
#include "editor/editor_internal.h"

namespace editor {

std::string dirOf(const std::string& n) { size_t p = n.rfind('/'); return p == std::string::npos ? "" : n.substr(0, p); }

std::string baseOf(const std::string& n) {
    std::string b = n.substr(n.rfind('/') == std::string::npos ? 0 : n.rfind('/') + 1);
    std::string l = lower(b);
    for (const char* e : {".glb", ".gltf"}) if (l.size() > strlen(e) && l.compare(l.size() - strlen(e), strlen(e), e) == 0) return b.substr(0, b.size() - strlen(e));
    return b;
}

bool isLoose(const std::string& n) { std::string l = lower(n); return l.size() > 5 && l.compare(l.size() - 5, 5, ".gltf") == 0; }

// the name scanAssets gives a .glb at assets/<dir>/<base>.glb
std::string glbName(const std::string& dir, const std::string& base) { return dir.empty() ? base : dir + "/" + base + ".glb"; }

void renameThumb(const std::string& from, const std::string& to) {
    std::error_code ec;
    std::filesystem::create_directories(std::filesystem::path(thumbPath(to)).parent_path(), ec);
    std::filesystem::rename(thumbPath(from), thumbPath(to), ec);
    std::filesystem::rename(thumbPath(from) + ".txt", thumbPath(to) + ".txt", ec);
    if (auto it = thumbs.find(from); it != thumbs.end()) { thumbs[to] = it->second; thumbs.erase(it); }
    for (auto& m : scatterMix) if (m.name == from) m.name = to;
}

bool moveAsset(const std::string& name, const std::string& dir, const std::string& base) {
    if (isLoose(name)) { status = "pack " + name + " into a .glb first (right-click > Pack): its parts would be left behind"; return false; }
    std::string to = glbName(dir, base);
    if (to == name || base.empty()) return false;
    std::error_code ec;
    std::string src = assetFile(host.worldDir, name), dst = host.worldDir + "assets/" + (dir.empty() ? "" : dir + "/") + base + ".glb";
    if (std::filesystem::exists(dst, ec)) { status = to + " already exists"; return false; }
    std::filesystem::create_directories(std::filesystem::path(dst).parent_path(), ec);
    std::filesystem::rename(src, dst, ec);
    if (ec) { status = "move failed: " + ec.message(); return false; }
    addAssetAlias(host.worldDir, name, to);
    renameThumb(name, to);
    status = name + " -> " + to;
    scanAssets();
    return true;
}

void deleteAsset(const std::string& name) {
    std::error_code ec;
    std::filesystem::remove(assetFile(host.worldDir, name), ec);
    std::filesystem::remove(thumbPath(name), ec);
    std::filesystem::remove(thumbPath(name) + ".txt", ec);
    thumbs.erase(name);
    scatterMix.erase(std::remove_if(scatterMix.begin(), scatterMix.end(), [&](auto& m) { return m.name == name; }), scatterMix.end());
    status = ec ? "delete failed: " + ec.message() : "deleted " + name;
    scanAssets();
}

void renameFolder(const std::string& dir, const std::string& newBase) {
    std::string to = dirOf(dir).empty() ? newBase : dirOf(dir) + "/" + newBase;
    if (to == dir || newBase.empty()) return;
    std::error_code ec;
    if (std::filesystem::exists(host.worldDir + "assets/" + to, ec)) { status = to + " already exists"; return; }
    std::vector<std::string> inside;
    for (auto& n : assetNames) if (n.rfind(dir + "/", 0) == 0) inside.push_back(n);
    std::filesystem::rename(host.worldDir + "assets/" + dir, host.worldDir + "assets/" + to, ec);
    if (ec) { status = "rename failed: " + ec.message(); return; }
    std::filesystem::rename(host.worldDir + "assets/.thumbs/" + dir, host.worldDir + "assets/.thumbs/" + to, ec);
    for (auto& n : inside) {
        std::string nn = to + n.substr(dir.size());
        addAssetAlias(host.worldDir, n, nn);
        if (auto it = thumbs.find(n); it != thumbs.end()) { thumbs[nn] = it->second; thumbs.erase(it); }
        for (auto& m : scatterMix) if (m.name == n) m.name = nn;
    }
    if (curDir == dir || curDir.rfind(dir + "/", 0) == 0) curDir = to + curDir.substr(dir.size());
    status = "folder " + dir + " -> " + to;
    scanAssets();
}

void deleteFolder(const std::string& dir) {
    std::error_code ec;
    if (!std::filesystem::is_empty(host.worldDir + "assets/" + dir, ec)) { status = dir + " isn't empty: move or delete its assets first"; return; }
    std::filesystem::remove(host.worldDir + "assets/" + dir, ec);
    std::filesystem::remove_all(host.worldDir + "assets/.thumbs/" + dir, ec);
    if (curDir == dir) curDir = dirOf(dir);
    status = ec ? "delete failed: " + ec.message() : "folder " + dir + " deleted";
    scanAssets();
}

void assetBrowser() {
    ImGuiIO& io = ImGui::GetIO();
    ImGui::SetNextWindowPos(ImVec2(20, io.DisplaySize.y - 320), ImGuiCond_FirstUseEver);
    ImGui::SetNextWindowSize(ImVec2(std::min(900.f, io.DisplaySize.x - 440), 300), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Assets", &showBrowser)) { ImGui::End(); return; }
    static char filter[64] = "";
    static float tile = 96;
    ImGui::SetNextItemWidth(160);
    ImGui::InputTextWithHint("##filter", "search", filter, sizeof filter);
    ImGui::SameLine();
    ImGui::SetNextItemWidth(110);
    ImGui::SliderFloat("##size", &tile, 48, 192, "size %.0f");
    ImGui::SameLine();
    if (ImGui::Button("Rescan")) { scanAssets(); matlib::rescan(); }
    ImGui::SetItemTooltip("Re-read world/assets/ (nested folders too): .glb and .gltf files placed there show up without importing");
    ImGui::SameLine();
    ImGui::BeginDisabled(pickerOpen);
    if (ImGui::Button("Import files...")) openPicker(false);
    ImGui::SameLine();
    if (ImGui::Button("Import folder...")) openPicker(true);
    ImGui::EndDisabled();
    ImGui::SameLine();
    if (ImGui::Button("Open assets folder")) {
        std::error_code ec;
        std::filesystem::create_directories(host.worldDir + "assets", ec);
        openFolder(std::filesystem::absolute(host.worldDir + "assets").string());
    }
    ImGui::Checkbox("Scatter tool", &showScatter);
    ImGui::SameLine();
    {
        float pct = importKeep * 100.f;
        ImGui::SetNextItemWidth(150);
        if (ImGui::SliderFloat("##detail", &pct, 1, 100, pct >= 99.95f ? "import: full detail" : "import: %.0f%% triangles")) importKeep = pct / 100.f;
        ImGui::SetItemTooltip("Simplify meshes on import (meshoptimizer). Trees with millions of triangles break / cut down\n"
                              "much faster and load quicker at 10-25%%. Existing assets: right-click > Make simplified copy.");
        ImGui::SameLine();
    }
    ImGui::Checkbox("Snap to ground", &groundSnap);
    ImGui::SetItemTooltip("Placed and dragged assets, trees and rocks sit on the terrain (their lowest point)");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(110);
    ImGui::DragFloat("Sink into ground", &sinkDepth, 0.01f, -2.f, 5.f, "%.2f m");
    ImGui::SetItemTooltip("How far the base goes below the surface (roots, foundations, bedded rocks). Negative floats it.");
    ImGui::SameLine();
    ImGui::SetNextItemWidth(std::max(80.f, ImGui::GetContentRegionAvail().x - 70));
    bool go = ImGui::InputTextWithHint("##imp", "/path/to/model.gltf, .glb or folder (or drop onto the window)", importPath, sizeof importPath,
                                       ImGuiInputTextFlags_EnterReturnsTrue);
    ImGui::SameLine();
    if ((ImGui::Button("Import") || go) && importPath[0]) importPaths({importPath});
    if (int n = jobsLeft.load()) { ImGui::SameLine(); ImGui::TextDisabled("working (%d)...", n); }

    // breadcrumb: assets / a / b
    if (ImGui::SmallButton("assets")) curDir.clear();
    {
        std::string acc;
        size_t p0 = 0;
        while (p0 < curDir.size()) {
            size_t p1 = curDir.find('/', p0);
            if (p1 == std::string::npos) p1 = curDir.size();
            acc = curDir.substr(0, p1);
            ImGui::SameLine(0, 2); ImGui::TextDisabled("/"); ImGui::SameLine(0, 2);
            if (ImGui::SmallButton((curDir.substr(p0, p1 - p0) + "##bc" + acc).c_str())) { curDir = acc; break; }
            p0 = p1 + 1;
        }
    }
    ImGui::SameLine();
    if (ImGui::SmallButton("+ New folder")) { popup = POP_NEWFOLDER; popupName[0] = 0; }

    ImGui::BeginChild("grid", ImVec2(0, 0), ImGuiChildFlags_Borders);
    const float pad = ImGui::GetStyle().FramePadding.x, sp = ImGui::GetStyle().ItemSpacing.x, cell = tile + 2 * pad;
    int cols = std::max(1, (int)((ImGui::GetContentRegionAvail().x + sp) / (cell + sp))), col = 0;
    std::string f = lower(filter);
    auto label = [&](const std::string& s) {  // clipped to the tile
        std::string l = s;
        while (l.size() > 3 && ImGui::CalcTextSize(l.c_str()).x > cell) l = l.substr(0, l.size() - 4) + "..";
        ImGui::TextUnformatted(l.c_str());
    };
    auto next = [&] { if (++col < cols) ImGui::SameLine(); else col = 0; };
    // a folder tile: double-click opens, assets dropped on it move there
    auto folderTile = [&](const std::string& dir, const std::string& shown, bool up) {
        ImGui::PushID(("dir:" + dir + (up ? ".." : "")).c_str());
        ImGui::BeginGroup();
        ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.32f, 0.27f, 0.12f, 1));
        ImGui::Button(up ? "<-" : "[ ]", ImVec2(cell, cell));
        ImGui::PopStyleColor();
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) curDir = dir;
        if (ImGui::BeginDragDropTarget()) {
            if (const ImGuiPayload* p = ImGui::AcceptDragDropPayload("PTX_ASSET")) {
                std::string n = (const char*)p->Data;
                moveAsset(n, dir, baseOf(n));
                dragAsset.clear();
            }
            ImGui::EndDragDropTarget();
        }
        ImGui::SetItemTooltip(up ? "up (drop assets here to move them up)" : "%s\ndouble-click to open, drop assets on it to move them in", dir.c_str());
        if (!up && ImGui::BeginPopupContextItem("fctx")) {
            if (ImGui::MenuItem("Open")) curDir = dir;
            if (ImGui::MenuItem("Rename...")) { popup = POP_RENAMEDIR; popupTarget = dir; snprintf(popupName, sizeof popupName, "%s", shown.c_str()); }
            if (ImGui::MenuItem("Delete (when empty)")) deleteFolder(dir);
            if (ImGui::MenuItem("Show in folder")) openFolder(std::filesystem::absolute(host.worldDir + "assets/" + dir).string());
            ImGui::EndPopup();
        }
        label(shown);
        ImGui::EndGroup();
        ImGui::PopID();
        next();
    };
    if (f.empty()) {
        if (!curDir.empty()) folderTile(dirOf(curDir), "..", true);
        for (auto& d : assetDirs) if (dirOf(d) == curDir) folderTile(d, baseOf(d), false);
    }
    int shownCount = 0;
    for (auto& name : assetNames) {
        if (!f.empty() ? lower(name).find(f) == std::string::npos : dirOf(name) != curDir) continue;
        shownCount++;
        ImGui::PushID(name.c_str());
        ImGui::BeginGroup();
        uint64_t tex = thumbs.count(name) ? thumbs[name].tex : 0;
        if (tex) ImGui::ImageButton("##t", (ImTextureID)tex, ImVec2(tile, tile), ImVec2(0, 0), ImVec2(1, 1), ImVec4(0.12f, 0.13f, 0.15f, 1));
        else ImGui::Button("...", ImVec2(cell, cell));
        if (ImGui::IsItemHovered() && ImGui::IsMouseDoubleClicked(ImGuiMouseButton_Left)) addAsset(name, nullptr);
        if (ImGui::BeginDragDropSource()) {
            ImGui::SetDragDropPayload("PTX_ASSET", name.c_str(), name.size() + 1);
            if (tex) ImGui::Image((ImTextureID)tex, ImVec2(64, 64));
            ImGui::TextUnformatted(name.c_str());
            ImGui::EndDragDropSource();
        }
        ImGui::SetItemTooltip("%s\ndrag into the world or onto a folder; double-click adds it at the crosshair (right-click: more)", name.c_str());
        bool loose = isLoose(name);
        if (ImGui::BeginPopupContextItem("ctx")) {
            if (ImGui::MenuItem("Add at crosshair")) addAsset(name, nullptr);
            if (ImGui::MenuItem("Add to scatter mix")) {
                if (!std::any_of(scatterMix.begin(), scatterMix.end(), [&](auto& m) { return m.name == name; })) scatterMix.push_back({name});
                showScatter = true;
            }
            ImGui::Separator();
            if (ImGui::MenuItem("Rename...", nullptr, false, !loose)) { popup = POP_RENAME; popupTarget = name; snprintf(popupName, sizeof popupName, "%s", baseOf(name).c_str()); }
            if (ImGui::BeginMenu("Move to", !loose)) {
                if (ImGui::MenuItem("assets (top)", nullptr, false, dirOf(name) != "")) moveAsset(name, "", baseOf(name));
                for (auto& d : assetDirs)
                    if (ImGui::MenuItem(d.c_str(), nullptr, false, d != dirOf(name))) moveAsset(name, d, baseOf(name));
                ImGui::EndMenu();
            }
            if (loose) ImGui::SetItemTooltip("Pack it into one .glb first: a loose .gltf's .bin and textures would be left behind");
            if (ImGui::MenuItem("Delete...")) { popup = POP_DELETE; popupTarget = name; }
            ImGui::Separator();
            if (ImGui::MenuItem("Pack into one .glb", nullptr, false, loose)) queueJob({true, assetFile(host.worldDir, name)});
            ImGui::SetItemTooltip("A .gltf keeps its mesh (.bin) and textures as separate files: packing bakes them into one\n"
                                  "self-contained .glb so they can't get separated (the .gltf is left as it was)");
            if (ImGui::MenuItem("Extract materials to texture library")) queueJob({true, assetFile(host.worldDir, name), 1.f, assetName(name), true});
            static float keepPct = 25.f;
            ImGui::SetNextItemWidth(120);
            ImGui::SliderFloat("##keep", &keepPct, 1, 90, "%.0f%% triangles");
            ImGui::SameLine();
            if (ImGui::MenuItem("Make simplified copy")) queueJob({true, assetFile(host.worldDir, name), keepPct / 100.f, assetName(name) + fmt("_%d", (int)keepPct)});
            if (ImGui::MenuItem("Show in folder")) openFolder(std::filesystem::absolute(std::filesystem::path(assetFile(host.worldDir, name)).parent_path()).string());
            ImGui::EndPopup();
        }
        label(baseOf(name));
        ImGui::EndGroup();
        ImGui::PopID();
        next();
    }
    if (assetNames.empty())
        ImGui::TextWrapped("No assets yet. Drag .gltf / .glb files or whole folders from your file manager onto this window to import them.");
    else if (!shownCount && f.empty() && curDir.size())
        ImGui::TextDisabled("(empty folder: drag assets onto its tile, or drop files here to import)");
    ImGui::EndChild();

    // ---- dialogs
    const char* titles[] = {"", "New folder", "Rename asset", "Rename folder", "Delete asset"};
    if (popup) { ImGui::OpenPopup(titles[popup]); popupOpen = popup; popup = POP_NONE; }
    for (int k = 1; k <= 4; k++) {
        if (!ImGui::BeginPopupModal(titles[k], nullptr, ImGuiWindowFlags_AlwaysAutoResize)) continue;
        bool ok = false, close = false;
        if (k == POP_DELETE) {
            int refs = 0;
            for (auto& [kk, c] : cells) for (auto& it : c.items) refs += it.kind == K_ASSET && resolveAsset(host.worldDir, it.asset) == resolveAsset(host.worldDir, popupTarget);
            ImGui::Text("Delete %s from disk?", popupTarget.c_str());
            if (refs) ImGui::TextColored(ImVec4(1, 0.6f, 0.3f, 1), "%d placed cop%s in open cells will vanish (and any in other cells).", refs, refs == 1 ? "y" : "ies");
            if (isLoose(popupTarget)) ImGui::TextDisabled("Only the .gltf is removed; its .bin / textures stay in the folder.");
            ok = ImGui::Button("Delete");
        } else {
            if (ImGui::IsWindowAppearing()) ImGui::SetKeyboardFocusHere();
            ok = ImGui::InputText("Name", popupName, sizeof popupName, ImGuiInputTextFlags_EnterReturnsTrue);
            ok |= ImGui::Button("OK");
        }
        ImGui::SameLine();
        close = ImGui::Button("Cancel") || ImGui::IsKeyPressed(ImGuiKey_Escape);
        if (ok) {
            std::string nm = assetName(std::filesystem::path(std::string(popupName) + ".x"));  // sanitised, no extension
            if (k == POP_NEWFOLDER) {
                std::error_code ec;
                std::string d = curDir.empty() ? nm : curDir + "/" + nm;
                std::filesystem::create_directories(host.worldDir + "assets/" + d, ec);
                status = ec ? "couldn't create folder: " + ec.message() : "folder " + d + " created";
                scanAssets();
            } else if (k == POP_RENAME) moveAsset(popupTarget, dirOf(popupTarget), nm);
            else if (k == POP_RENAMEDIR) renameFolder(popupTarget, nm);
            else if (k == POP_DELETE) deleteAsset(popupTarget);
            close = true;
        }
        if (close) { ImGui::CloseCurrentPopup(); popupOpen = POP_NONE; }
        ImGui::EndPopup();
    }
    ImGui::End();
}

// an asset dragged from the browser: ghost marker over the world, placed on release
void assetDrop() {
    if (const ImGuiPayload* p = ImGui::GetDragDropPayload(); p && p->IsDataType("PTX_ASSET")) dragAsset = (const char*)p->Data;
    if (dragAsset.empty()) return;
    bool overUi = ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow);
    float at[3];
    bool hit = !overUi && mousePoint(at);
    if (hit) {
        ImVec2 s;
        if (project(at, s)) {
            ImDrawList* dl = ImGui::GetForegroundDrawList();
            dl->AddCircle(s, 10.f, IM_COL32(255, 210, 40, 230), 0, 2.f);
            dl->AddText(ImVec2(s.x + 14, s.y - 7), IM_COL32(255, 255, 255, 230), dragAsset.c_str());
        }
    }
    if (!ImGui::IsMouseDown(ImGuiMouseButton_Left)) {
        if (hit) addAsset(dragAsset, at);
        dragAsset.clear();
    }
}

}  // namespace editor

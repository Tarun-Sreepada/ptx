// Scatter brush: a weighted mix of assets dropped over a disc (dart throwing with footprint spacing).
#include "editor/editor_internal.h"

namespace editor {

float footOf(const std::string& name) {
    auto it = thumbs.find(name);
    return it != thumbs.end() && it->second.foot > 0 ? it->second.foot : 2.f;  // unknown yet: assume a small tree
}

// Fill the disc around `at` up to `density` instances per 100 m2. Each candidate's footprint (bounding radius x
// scale x spacing) must clear the ones already there (assets in nearby cells included), so nothing interpenetrates.
void scatter(const float at[3]) {
    float wsum = 0;
    for (auto& e : scatterMix) wsum += std::max(e.weight, 0.f);
    if (wsum <= 0) { status = "add assets to the scatter mix first (right-click a thumbnail)"; return; }
    struct Disc { float x, z, r; };
    std::vector<Disc> taken;
    float R = scatterRadius;
    for (auto& [k, c] : cells)
        for (auto& it : c.items)
            if (it.kind == K_ASSET || it.kind == K_TREE) {
                float dx = it.pos[0] - at[0], dz = it.pos[2] - at[2];
                if (dx * dx + dz * dz < (R + 30) * (R + 30))
                    taken.push_back({it.pos[0], it.pos[2], (it.kind == K_ASSET ? footOf(it.asset) : 1.f) * it.scale[0] * scatterSpacing});
            }
    int want = (int)(scatterDensity * 3.14159f * R * R / 100.f), placed = 0, skippedCells = 0;
    std::map<std::pair<int, int>, bool> undone;
    for (int tries = 0; placed < want && tries < want * 30; tries++) {
        float a = (float)rand() / RAND_MAX * 6.2831853f, d = R * sqrtf((float)rand() / RAND_MAX);
        float x = at[0] + cosf(a) * d, z = at[2] + sinf(a) * d;
        float pick = (float)rand() / RAND_MAX * wsum;
        const MixEntry* e = &scatterMix.back();
        for (auto& m : scatterMix) { if ((pick -= std::max(m.weight, 0.f)) <= 0) { e = &m; break; } }
        float sc = scatterScale[0] + (scatterScale[1] - scatterScale[0]) * (float)rand() / RAND_MAX;
        float r = footOf(e->name) * sc * scatterSpacing;
        bool clear = true;
        for (auto& t : taken) if ((t.x - x) * (t.x - x) + (t.z - z) * (t.z - z) < (t.r + r) * (t.r + r)) { clear = false; break; }
        if (!clear) continue;
        ECell* c = open(cellOf(x), cellOf(z));
        if (!c) { skippedCells++; continue; }
        if (!undone[{c->cx, c->cz}]) { pushUndo(*c); undone[{c->cx, c->cz}] = true; }  // one undo step per cell
        Item it;
        it.kind = K_ASSET;
        it.asset = it.name = e->name;
        it.pos[0] = x; it.pos[2] = z;
        it.rot[1] = (float)rand() / RAND_MAX * 360.f;
        it.scale[0] = it.scale[1] = it.scale[2] = sc;
        snapToGround(*c, it);
        c->items.push_back(it);
        changed(*c, true);
        taken.push_back({x, z, r});
        placed++;
    }
    status = fmt("scattered %d of %d (%s)", placed, want, placed < want ? "the rest had no room at this spacing" : "full");
    if (skippedCells) status += fmt(", %d in cells still loading", skippedCells);
}

// remove scattered mix assets inside the disc
void unscatter(const float at[3]) {
    int n = 0;
    for (auto& [k, c] : cells) {
        bool any = false;
        for (size_t i = c.items.size(); i-- > 0;) {
            const Item& it = c.items[i];
            float dx = it.pos[0] - at[0], dz = it.pos[2] - at[2];
            if (it.kind != K_ASSET || dx * dx + dz * dz > scatterRadius * scatterRadius) continue;
            if (!std::any_of(scatterMix.begin(), scatterMix.end(), [&](auto& m) { return m.name == it.asset; })) continue;
            if (!any) { pushUndo(c); any = true; }
            c.items.erase(c.items.begin() + i);
            n++;
        }
        if (any) { changed(c, true); if (sel.cx == c.cx && sel.cz == c.cz) sel.idx = -1; }
    }
    status = fmt("removed %d", n);
}

void scatterPanel() {
    ImGui::SetNextWindowSize(ImVec2(340, 360), ImGuiCond_FirstUseEver);
    if (!ImGui::Begin("Scatter (forests, rocks)", &showScatter)) { ImGui::End(); return; }
    ImGui::TextWrapped("Mix: right-click thumbnails > Add to scatter mix, or drag them here.");
    ImGui::BeginChild("mix", ImVec2(0, 130), ImGuiChildFlags_Borders);
    for (size_t i = 0; i < scatterMix.size(); i++) {
        ImGui::PushID((int)i);
        uint64_t tex = thumbs.count(scatterMix[i].name) ? thumbs[scatterMix[i].name].tex : 0;
        if (tex) { ImGui::Image((ImTextureID)tex, ImVec2(28, 28)); ImGui::SameLine(); }
        ImGui::SetNextItemWidth(90);
        ImGui::SliderFloat("##w", &scatterMix[i].weight, 0, 10, "weight %.1f");
        ImGui::SameLine();
        ImGui::TextUnformatted(scatterMix[i].name.c_str());
        ImGui::SameLine();
        bool del = ImGui::SmallButton("x");
        ImGui::PopID();
        if (del) { scatterMix.erase(scatterMix.begin() + i); break; }
    }
    ImGui::EndChild();
    if (ImGui::BeginDragDropTarget()) {
        if (const ImGuiPayload* p = ImGui::AcceptDragDropPayload("PTX_ASSET")) {
            std::string n = (const char*)p->Data;
            if (!std::any_of(scatterMix.begin(), scatterMix.end(), [&](auto& m) { return m.name == n; })) scatterMix.push_back({n});
            dragAsset.clear();
        }
        ImGui::EndDragDropTarget();
    }
    ImGui::DragFloat("Radius (m)", &scatterRadius, 0.2f, 1.f, 300.f, "%.0f");
    ImGui::DragFloat("Density (per 100 m2)", &scatterDensity, 0.02f, 0.01f, 50.f, "%.2f");
    ImGui::DragFloat("Spacing", &scatterSpacing, 0.01f, 0.1f, 2.f, "%.2f x bounds");
    ImGui::SetItemTooltip("1 = bounding footprints never overlap; lower lets canopies interleave, higher thins it out");
    ImGui::DragFloat2("Scale range", scatterScale, 0.01f, 0.1f, 5.f, "%.2f");
    ImGui::TextDisabled("Sits on the ground, sunk by 'Sink into ground' (Assets window). Random yaw.");
    float p[3];
    placePoint(p);
    if (ImGui::Button("Scatter at crosshair")) scatter(p);
    ImGui::SameLine();
    if (ImGui::Button("Clear mix assets in radius")) unscatter(p);
    ImGui::TextDisabled("Ctrl+Z undoes per cell");
    ImGui::SeparatorText("World forest");
    static float builtinW = 0.f;
    ImGui::SetNextItemWidth(120);
    ImGui::DragFloat("Built-in trees weight", &builtinW, 0.05f, 0.f, 20.f, "%.2f");
    ImGui::SetItemTooltip("0 = the mix replaces the procedural trees everywhere; > 0 keeps that share of them");
    if (ImGui::Button("Use this mix for the world's trees")) {
        std::string path = host.worldDir + "forest.txt";
        if (FILE* f = fopen(path.c_str(), "w")) {
            fprintf(f, "# world forest: <weight> <asset>, read at startup (ptx editor)\nbuiltin %g\nspacing %g\n", builtinW, scatterSpacing);
            for (auto& e : scatterMix) if (e.weight > 0) fprintf(f, "%g %s\n", e.weight, e.name.c_str());
            fclose(f);
            status = "wrote forest.txt: restart ptx to grow the new forest (variants + impostors are built at startup)";
        } else status = "couldn't write " + path;
    }
    ImGui::SameLine();
    if (ImGui::Button("Restore built-in")) {
        std::error_code ec;
        std::filesystem::remove(host.worldDir + "forest.txt", ec);
        status = "forest.txt removed: built-in trees after a restart";
    }
    ImGui::TextDisabled("Uses Spacing above; the world density slider still applies. Up to 12 assets.");
    // preview ring on the ground
    ImDrawList* dl = ImGui::GetBackgroundDrawList();
    ImVec2 prev;
    bool have = false;
    for (int i = 0; i <= 48; i++) {
        float a = i / 48.f * 6.2831853f, q[3] = {p[0] + cosf(a) * scatterRadius, 0, p[2] + sinf(a) * scatterRadius};
        q[1] = host.height(q[0], q[2]) + 0.2f;
        ImVec2 sc;
        bool ok = project(q, sc);
        if (ok && have) dl->AddLine(prev, sc, IM_COL32(120, 255, 140, 200), 2.f);
        prev = sc; have = ok;
    }
    ImGui::End();
}

}  // namespace editor

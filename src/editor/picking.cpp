// Picking: rays against items and the terrain, the mouse ray, screen projection, where new items go.
#include "editor/editor_internal.h"

namespace editor {

// ---------------------------------------------------------------- picking
void localBounds(const ECell& c, const Item& it, float mn[3], float mx[3]) {
    auto set = [&](float a, float b, float cc, float d, float e, float f) { mn[0] = a; mn[1] = b; mn[2] = cc; mx[0] = d; mx[1] = e; mx[2] = f; };
    switch (it.kind) {
    case K_PRIM: set(-it.size[0] * 0.5f, 0, -it.size[2] * 0.5f, it.size[0] * 0.5f, it.size[1], it.size[2] * 0.5f); return;
    case K_MESH:
        if (it.mesh >= 0 && it.mesh < (int)c.meshes.size()) { memcpy(mn, c.meshes[it.mesh].bmin, 12); memcpy(mx, c.meshes[it.mesh].bmax, 12); return; }
        break;
    case K_ASSET:
        if (Model* m = stream::findModel(assetFile(host.worldDir, it.asset))) {
            for (int k = 0; k < 3; k++) { mn[k] = 1e30f; mx[k] = -1e30f; }
            for (auto& in : m->data.instances) {
                if (in.mesh < 0) continue;
                float a[3], b[3];
                xfAabb(in.xf, m->data.meshes[in.mesh].bmin, m->data.meshes[in.mesh].bmax, a, b);
                for (int k = 0; k < 3; k++) { mn[k] = std::min(mn[k], a[k]); mx[k] = std::max(mx[k], b[k]); }
            }
            if (mn[0] <= mx[0]) return;
        }
        break;
    case K_LIGHT: set(-0.15f, -0.15f, -0.15f, 0.15f, 0.15f, 0.15f); return;
    case K_VOLUME: case K_ZONE: set(-1, -1, -1, 1, 1, 1); return;
    case K_TREE: set(-1, 0, -1, 1, 10, 1); return;
    case K_ROCK: set(-0.9f, -0.3f, -0.9f, 0.9f, 0.7f, 0.9f); return;
    default: break;
    }
    set(-0.5f, -0.5f, -0.5f, 0.5f, 0.5f, 0.5f);
}

Xf boundsXf(const Item& it) {
    if (it.kind == K_LIGHT) { float s[3] = {1, 1, 1}; return compose(it.pos, it.rot, s); }
    if (it.kind == K_VOLUME || it.kind == K_ZONE) { float r[3] = {0, 0, 0}; return compose(it.pos, r, it.scale); }
    if (it.kind == K_TREE || it.kind == K_ROCK) { float r[3] = {0, it.rot[1], 0}, s[3] = {it.scale[0], it.scale[0], it.scale[0]}; return compose(it.pos, r, s); }
    return itemXf(it);
}

// Put the item's base on the terrain under its centre, sunk by sinkDepth (its local bounds give where the base is,
// so a model with its origin mid-height still lands on its bottom).
void snapToGround(const ECell& c, Item& it) {
    float mn[3], mx[3], a[3], b[3];
    localBounds(c, it, mn, mx);
    xfAabb(boundsXf(it), mn, mx, a, b);
    it.pos[1] += host.height(it.pos[0], it.pos[2]) - sinkDepth - a[1];
}

bool groundable(const Item& it) { return it.kind == K_ASSET || it.kind == K_TREE || it.kind == K_ROCK; }  // not primitives: they stack

bool pickable(const Item& it) { return showVolumes || (it.kind != K_VOLUME && it.kind != K_ZONE); }

// ray vs the item's oriented box: entry distance (0 if inside), or < 0
float rayItem(const ECell& c, const Item& it, const float o[3], const float d[3]) {
    float mn[3], mx[3];
    localBounds(c, it, mn, mx);
    Xf inv = inverse(boundsXf(it));
    float lo[3], ld[3];
    for (int i = 0; i < 3; i++) {
        lo[i] = inv.m[i][0] * o[0] + inv.m[i][1] * o[1] + inv.m[i][2] * o[2] + inv.m[i][3];
        ld[i] = inv.m[i][0] * d[0] + inv.m[i][1] * d[1] + inv.m[i][2] * d[2];
    }
    float t0 = 0, t1 = 1e30f;
    for (int k = 0; k < 3; k++) {
        if (fabsf(ld[k]) < 1e-12f) { if (lo[k] < mn[k] || lo[k] > mx[k]) return -1; continue; }
        float a = (mn[k] - lo[k]) / ld[k], b = (mx[k] - lo[k]) / ld[k];
        if (a > b) std::swap(a, b);
        t0 = std::max(t0, a); t1 = std::min(t1, b);
        if (t0 > t1) return -1;
    }
    return t0;
}

Hit pick(const float o[3], const float d[3]) {
    Hit best{0, 0, -1, 1e30f};
    float bestVol = 1e30f;
    for (auto& [k, c] : cells)
        for (size_t i = 0; i < c.items.size(); i++) {
            const Item& it = c.items[i];
            if (!pickable(it)) continue;
            float t = rayItem(c, it, o, d);
            if (t < 0 || t > 500.f) continue;
            float mn[3], mx[3];
            localBounds(c, it, mn, mx);
            float vol = (mx[0] - mn[0]) * (mx[1] - mn[1]) * (mx[2] - mn[2]);
            // nearest entry wins; when the ray starts inside several (a volume around a room), the smallest
            if (t < best.t - 1e-3f || (t < best.t + 1e-3f && vol < bestVol)) { best = {k.first, k.second, (int)i, t}; bestVol = vol; }
        }
    return best;
}

float rayTerrain(const float o[3], const float d[3]) {
    for (int k = 0; k < 3; k++) if (!std::isfinite(o[k]) || !std::isfinite(d[k])) return -1;
    float t = 0, prev = 0;
    for (int i = 0; i < 400 && t < 400.f; i++) {
        float y = o[1] + d[1] * t, h = host.height(o[0] + d[0] * t, o[2] + d[2] * t);
        if (y < h) {
            for (int j = 0; j < 12; j++) {  // bisect
                float m = 0.5f * (prev + t);
                if (o[1] + d[1] * m < host.height(o[0] + d[0] * m, o[2] + d[2] * m)) t = m; else prev = m;
            }
            return t;
        }
        prev = t;
        t += std::max(0.05f, 0.02f * t);
    }
    return -1;
}

void mouseRay(float d[3]) {
    ImGuiIO& io = ImGui::GetIO();
    float W = io.DisplaySize.x, H = io.DisplaySize.y;
    float x = (2.f * io.MousePos.x / W - 1.f) * view.tanHalfFov * W / H, y = (1.f - 2.f * io.MousePos.y / H) * view.tanHalfFov;
    float l = 0;
    for (int k = 0; k < 3; k++) { d[k] = view.fwd[k] + view.right[k] * x + view.up[k] * y; l += d[k] * d[k]; }
    l = sqrtf(l);
    for (int k = 0; k < 3; k++) d[k] /= l;
}

bool project(const float p[3], ImVec2& s) {
    ImGuiIO& io = ImGui::GetIO();
    float d[3] = {p[0] - view.pos[0], p[1] - view.pos[1], p[2] - view.pos[2]};
    float z = d[0] * view.fwd[0] + d[1] * view.fwd[1] + d[2] * view.fwd[2];
    if (z < 0.05f) return false;
    float W = io.DisplaySize.x, H = io.DisplaySize.y;
    float x = (d[0] * view.right[0] + d[1] * view.right[1] + d[2] * view.right[2]) / (z * view.tanHalfFov * W / H);
    float y = (d[0] * view.up[0] + d[1] * view.up[1] + d[2] * view.up[2]) / (z * view.tanHalfFov);
    s = ImVec2((x + 1.f) * 0.5f * W, (1.f - y) * 0.5f * H);
    return true;
}

// where new things go: what the screen centre looks at (items, then terrain), within 60 m
void placePoint(float out[3]) {
    float t = 1e30f;
    Hit h = pick(view.pos, view.fwd);
    if (h.idx >= 0) t = h.t;
    float tt = rayTerrain(view.pos, view.fwd);
    if (tt >= 0) t = std::min(t, tt);
    if (t > 60.f) t = 10.f;
    for (int k = 0; k < 3; k++) out[k] = view.pos[k] + view.fwd[k] * t;
    if (t == 10.f) out[1] = host.height(out[0], out[2]);
}

float camYawSnapped() { return roundf(atan2f(view.fwd[0], view.fwd[2]) / DEG / 90.f) * 90.f; }

void addItem(Item it, bool atCursor) {
    if (atCursor) {
        float p[3];
        placePoint(p);
        it.pos[0] = snapv(p[0], grid); it.pos[2] = snapv(p[2], grid);
        it.pos[1] = p[1] + (it.kind == K_LIGHT ? 2.f : it.kind == K_VOLUME ? it.scale[1] : 0.f);
        if (it.kind == K_PRIM || it.kind == K_ASSET) it.rot[1] = camYawSnapped();
    }
    ECell* c = open(cellOf(it.pos[0]), cellOf(it.pos[2]));
    if (!c) { status = "that cell is still loading"; return; }
    pushUndo(*c);
    if (groundSnap && groundable(it)) snapToGround(*c, it);
    c->items.push_back(it);
    changed(*c, true);
    sel = {c->cx, c->cz, (int)c->items.size() - 1};
}

Item prim(int shape, float sx, float sy, float sz, const std::string& mat, const char* name) {
    Item it;
    it.kind = K_PRIM;
    it.shape = shape;
    it.size[0] = sx; it.size[1] = sy; it.size[2] = sz;
    it.mat = mat;
    it.name = name;
    if (auto* e = matlib::find(mat)) it.strength = e->strength;
    return it;
}

}  // namespace editor

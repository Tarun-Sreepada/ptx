#include "tools/test_hooks.h"

#include <cmath>
#include <cstdio>
#include <string>

#include <imgui.h>

#include "app/cli.h"
#include "app/frame_state.h"
#include "characters/character.h"
#include "core/math.h"
#include "destruction/throw.h"
#include "destruction/thrown.h"
#include "destruction/wood.h"
#include "editor/editor.h"
#include "scene/camera.h"
#include "scene/vegetation.h"
#include "vegetation/trees.h"
#include "world/terrain.h"

namespace {
constexpr uint32_t AUTO_THROW_START = 30, AUTO_THROW_EVERY = 20;   // --throw: frames
constexpr uint32_t TREE_THROW_START = 60, TREE_THROW_EVERY = 40;   // --tree-throw: frames
constexpr float TREE_THROW_BACK = 6.f, TREE_THROW_UP = 0.5f;       // camera offset from the target (m)
constexpr int MAP_CLICK_PRESS = 5, MAP_CLICK_RELEASE = 305;        // --map-click: frames after the start
constexpr float MAP_CLICK_Y = 0.6f;                                // screen height fraction of the click
}

void testHooksUploadWindow() {
    if ((int)frameIndex == editDemoFrame) editor::demoHouse();
    if ((int)frameIndex == editSaveFrame) {
        std::string err;
        if (!editor::saveAll(err)) fprintf(stderr, "editor: %s\n", err.c_str());
    }
}

void testHooksImGuiInput() {
    if (mapClickFrame < 0) return;  // synthetic input after the backend's, so it wins this frame
    ImGuiIO& io = ImGui::GetIO();
    int f = (int)frameIndex - mapClickFrame;
    if (f == 0 || f == 1) io.AddKeyEvent(ImGuiKey_M, f == 0);
    if (f >= 0) io.AddMousePosEvent(io.DisplaySize.x * 0.5f, io.DisplaySize.y * MAP_CLICK_Y);
    if (f == MAP_CLICK_PRESS || f == MAP_CLICK_RELEASE) io.AddMouseButtonEvent(0, f == MAP_CLICK_PRESS);
}

void testHooksSimulate() {
    if (autoThrow > 0 && frameIndex >= AUTO_THROW_START && frameIndex % AUTO_THROW_EVERY == 0) { throwRock(); autoThrow--; }
    if (treeThrow <= 0 || frameIndex < TREE_THROW_START || frameIndex % TREE_THROW_EVERY != 0 || treeList.empty()) return;
    const TreeInst* tb = nullptr;
    for (const TreeInst& tr : treeList)
        if (tr.v < (uint32_t)trees::VARIANTS && (!tb || dot(tr.p - cam.pos, tr.p - cam.pos) < dot(tb->p - cam.pos, tb->p - cam.pos))) tb = &tr;
    if (!tb) return;
    // a limb 3-10 cm thick (most of the throws), now and then the trunk
    trees::Wood w;
    trees::woodWhole((int)tb->v, w);
    WoodFrame F = treeFrame(*tb);
    V3 tgt = tb->p + V3{0, 1.2f * tb->scale, 0};
    float a[3], b[3], r;
    for (int k = (int)(hash2(treeThrow, 5) * 400); treeThrow % 4 && k < 4000; k++)
        if (trees::woodElem(w, k % 1000, a, b, r) && r * tb->scale > 0.03f && r * tb->scale < 0.1f && k % 1000 >= trees::TRUNK_ELEMS) {
            tgt = woodToWorld(F, (V3{a[0], a[1], a[2]} + V3{b[0], b[1], b[2]}) * 0.5f);
            break;
        }
    V3 p = tgt + V3{-TREE_THROW_BACK, TREE_THROW_UP, 0}, d = normalize(tgt - p);
    cam.pos = p;
    cam.yaw = atan2f(d.x, d.z);
    cam.pitch = asinf(d.y);
    throwRock();
    treeThrow--;
}

void testHooksAfterPhysics() {
    if (ragdollFrame < 0 || (int)frameIndex != ragdollFrame) return;
    for (character::Character* c : character::all()) {
        float j[3] = {0, 0, 0};
        character::ragdoll(*c, j, c->pos);
    }
}

#include "scene/camera.h"
#include <algorithm>
#include <cmath>
#define GLFW_INCLUDE_NONE
#include <GLFW/glfw3.h>
#include <imgui.h>
#include "core/input.h"
#include "physics/physics.h"
#include "editor/editor.h"
#include "app/frame_state.h"
#include "app/settings.h"
#include "core/math.h"
#include "destruction/throw.h"
#include "gpu/context.h"
#include "world/terrain_query.h"

namespace {
constexpr float EYE_HEIGHT = 1.65f;   // m above the feet
constexpr float STRIDE = 0.75f;       // m per step
constexpr float BOB_DIP = 0.035f;     // m the eye drops at a footfall, at walking pace
constexpr float BOB_SWAY = 0.012f;    // m it sways toward the planted foot
}  // namespace

Camera cam, prevCam;

// player / streaming settings
bool walkMode = false, grounded = false, walkReset = false;

float flySpeed = 1.f;  // fly-mode speed multiplier (UI), Ctrl = x8 on top

bool gyroAim = true;

float gyroSens = 1.0f;

V3 playerFeet;

void updateCamera(float dt) {
    ImGuiIO& io = ImGui::GetIO();
    static double lx, ly;
    static bool looking = false;
    double mx, my;
    glfwGetCursorPos(window, &mx, &my);
    bool rmb = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    if (rmb && (looking || !io.WantCaptureMouse)) {
        if (looking) {
            cam.yaw -= (float)(mx - lx) * 0.003f;
            cam.pitch = std::clamp(cam.pitch - (float)(my - ly) * 0.003f, -1.5f, 1.5f);
        }
        looking = true;
    } else looking = false;
    lx = mx; ly = my;
    // throw: G, left click (outside the UI) or right trigger, on the press edge
    static bool throwHeld = false;
    bool th = glfwGetKey(window, GLFW_KEY_G) == GLFW_PRESS ||
              (glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS && !io.WantCaptureMouse && !editor::wantsMouse()) || input::pad().rt > 0.6f;
    if (th && !throwHeld) throwRock();
    throwHeld = th;
    // P: primary visibility raster G-buffer <-> path traced camera rays (A/B comparison); lighting is traced either way
    static bool pHeld = false;
    bool pk = glfwGetKey(window, GLFW_KEY_P) == GLFW_PRESS && !io.WantCaptureKeyboard;
    if (pk && !pHeld) {
        S.rasterPrimary = !S.rasterPrimary;
        resetHistory = true;
        glfwSetWindowTitle(window, S.rasterPrimary ? "ptx - primary: raster G-buffer (P to toggle)" : "ptx - primary: path traced (P to toggle)");
    }
    pHeld = pk;
    // I: global illumination on / off (direct light only), to see what the bounced light adds
    static bool iHeld = false;
    bool ik = glfwGetKey(window, GLFW_KEY_I) == GLFW_PRESS && !io.WantCaptureKeyboard;
    if (ik && !iHeld) {
        S.giEnabled = !S.giEnabled;
        resetHistory = true;
        glfwSetWindowTitle(window, S.giEnabled ? "ptx - GI on (I to toggle)" : "ptx - GI off: direct light only (I to toggle)");
    }
    iHeld = ik;
    // gamepad look: stick (squared response for fine aim) + gyro (Steam Controller / DualSense)
    const input::Pad& gp = input::pad();
    if (gp.connected) {
        cam.yaw -= gp.look[0] * fabsf(gp.look[0]) * 2.8f * dt;
        cam.pitch += gp.look[1] * fabsf(gp.look[1]) * 2.0f * dt;
        if (gp.hasGyro && gyroAim) {
            cam.yaw += gp.gyro[1] * gyroSens * dt;
            cam.pitch += gp.gyro[0] * gyroSens * dt;
        }
        cam.pitch = std::clamp(cam.pitch, -1.5f, 1.5f);
    }
    if (io.WantCaptureKeyboard && !gp.connected) return;
    static bool fPrev = false;
    bool fKey = (!io.WantCaptureKeyboard && glfwGetKey(window, GLFW_KEY_F) == GLFW_PRESS) || gp.togglePressed;
    if (fKey && !fPrev) {
        walkMode = !walkMode;
        if (walkMode) {
            V3 feet = cam.pos - V3{0, EYE_HEIGHT, 0};
            feet.y = std::max(feet.y, heightFn(feet.x, feet.z) + 0.05f);
            physics::spawnPlayer(&feet.x);
            playerFeet = feet;
            walkReset = true;
        }
    }
    fPrev = fKey;
    if (walkMode) {
        // first person: horizontal wish velocity, physics resolves collisions, stairs and gravity
        float spd = (glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) || gp.sprint) ? 2.f * S.walkSpeed : S.walkSpeed;
        V3 fw = normalize(V3{sinf(cam.yaw), 0, cosf(cam.yaw)}), rt = cam.right();
        V3 w{0, 0, 0};
        if (glfwGetKey(window, GLFW_KEY_W)) w = w + fw;
        if (glfwGetKey(window, GLFW_KEY_S)) w = w - fw;
        if (glfwGetKey(window, GLFW_KEY_D)) w = w + rt;
        if (glfwGetKey(window, GLFW_KEY_A)) w = w - rt;
        if (dot(w, w) > 0) w = normalize(w) * spd;
        else w = (fw * gp.move[1] + rt * gp.move[0]) * spd;  // analog: partial tilt walks slower
        float wish[2] = {w.x, w.z};
        bool jump = glfwGetKey(window, GLFW_KEY_SPACE) == GLFW_PRESS || gp.jump;
        // fixed 120 Hz character step, rendered position interpolated between the last two steps:
        // a variable dt makes the controller's ground snapping / velocity resolve differently every frame
        const float step = 1.f / 120.f;
        static float acc = 0;
        static V3 feetPrev = playerFeet, feetCur = playerFeet, eye = playerFeet;
        if (walkReset) { acc = 0; feetPrev = feetCur = eye = playerFeet; walkReset = false; }
        acc += dt;
        while (acc >= step) {
            bool wasGrounded = grounded;
            feetPrev = feetCur;
            physics::movePlayer(step, wish, jump, walkMaxSpeed(), &feetCur.x, grounded);
            if (grounded && !wasGrounded) input::rumble(0.4f, 0.1f, 80);  // landing thump
            acc -= step;
            // clipped through the ground: once 50 m under the terrain, put the player back on top of it
            float gy = heightFn(feetCur.x, feetCur.z);
            if (feetCur.y < gy - 50.f) {
                feetCur.y = gy + 1.f;
                physics::spawnPlayer(&feetCur.x);
                feetPrev = eye = feetCur;
                grounded = false;
            }
        }
        float a = acc / step;
        playerFeet = feetPrev + (feetCur - feetPrev) * a;
        // stair step-up / stick-to-floor teleport the capsule vertically: ease the eye height over ~60 ms
        eye.x = playerFeet.x; eye.z = playerFeet.z;
        eye.y += (playerFeet.y - eye.y) * (1.f - expf(-dt / 0.06f));
        if (fabsf(playerFeet.y - eye.y) > 1.f) eye.y = playerFeet.y;  // falls / respawn: don't lag behind
        // head bob: the eye dips at every footfall and sways toward the planted foot, more the faster you go
        static V3 lastFeet = playerFeet;
        static float bobPhase = 0.f, bobAmp = 0.f;
        V3 moved = playerFeet - lastFeet;
        lastFeet = playerFeet;
        float speed = sqrtf(moved.x * moved.x + moved.z * moved.z) / std::max(dt, 1e-4f);
        bobAmp += ((grounded ? std::min(speed / S.walkSpeed, 2.f) : 0.f) - bobAmp) * (1.f - expf(-dt / 0.15f));
        bobPhase += std::min(speed, 12.f) * dt / STRIDE * 3.14159265f;  // a dip per step, a sway per two
        float dip = -BOB_DIP * bobAmp * S.headBob * 0.5f * (1.f - cosf(2.f * bobPhase));
        float sway = BOB_SWAY * bobAmp * S.headBob * sinf(bobPhase);
        cam.pos = eye + V3{0, EYE_HEIGHT + dip, 0} + cam.right() * sway;
        // speed widens the view: nothing at a jog, up to speedFov at the top speed (uncapped: at 60 m/s), along the
        // same sigmoid, eased over ~0.2 s so a landing or a wall doesn't snap it
        static float spdSmooth = 0.f;
        spdSmooth += (speed - spdSmooth) * (1.f - expf(-dt / 0.2f));
        float top = S.speedUncapped ? 60.f : S.maxSpeed, jog = 2.f * S.walkSpeed;
        cam.fovBoost = top > jog ? S.speedFov * sigmoid01((spdSmooth - jog) / (top - jog)) : 0.f;
        return;
    }
    cam.fovBoost = 0.f;
    float sp = ((glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) || gp.sprint) ? 12.f : 4.f) * flySpeed * dt;
    if (glfwGetKey(window, GLFW_KEY_LEFT_CONTROL) == GLFW_PRESS) sp *= 8.f;  // turbo: cross the map quickly
    V3 f = cam.fwd(), r = cam.right();
    cam.pos = cam.pos + f * (gp.move[1] * sp) + r * (gp.move[0] * sp);
    if (gp.up) cam.pos.y += sp;
    if (gp.down) cam.pos.y -= sp;
    if (glfwGetKey(window, GLFW_KEY_W)) cam.pos = cam.pos + f * sp;
    if (glfwGetKey(window, GLFW_KEY_S)) cam.pos = cam.pos - f * sp;
    if (glfwGetKey(window, GLFW_KEY_D)) cam.pos = cam.pos + r * sp;
    if (glfwGetKey(window, GLFW_KEY_A)) cam.pos = cam.pos - r * sp;
    if (glfwGetKey(window, GLFW_KEY_E)) cam.pos.y += sp;
    if (glfwGetKey(window, GLFW_KEY_Q)) cam.pos.y -= sp;
    // the flying camera stays above the ground (craters, sculpted hills): no seeing through the terrain from inside
    cam.pos.y = std::max(cam.pos.y, heightFn(cam.pos.x, cam.pos.z) + 0.3f);
}

float walkMaxSpeed() { return S.speedUncapped ? 1e6f : std::max(S.maxSpeed, 1.f); }

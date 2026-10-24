// Gamepad input via SDL3 (gamepad subsystem only; GLFW still owns the window and keyboard/mouse).
// Steam Controller (original + 2025 model) is read directly through SDL's HIDAPI driver when Steam isn't
// running; with Steam running it arrives as Steam Input's virtual pad. Gyro is exposed for gyro aiming.
#pragma once

namespace input {
struct Pad {
    bool connected = false;
    const char* name = "";
    float move[2] = {0, 0};  // left stick / left trackpad, x right, y forward, deadzoned
    float look[2] = {0, 0};  // right stick / right trackpad, x right, y up
    float gyro[3] = {0, 0, 0};  // rad/s (pitch, yaw, roll) when the pad has a gyro
    bool hasGyro = false;
    float lt = 0, rt = 0;
    bool jump = false, sprint = false, up = false, down = false;
    bool togglePressed = false;  // edge: walk/fly toggle
};

void init();
void shutdown();
void update();  // pump SDL events, refresh pad()
const Pad& pad();
void rumble(float low, float high, unsigned ms);
}  // namespace input

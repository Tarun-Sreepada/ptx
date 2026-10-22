#include "input.h"

#include <SDL3/SDL.h>

#include <cmath>
#include <cstdio>

namespace input {
namespace {
SDL_Gamepad* gp = nullptr;
Pad state;
bool prevToggle = false;

float axis(SDL_GamepadAxis a) { return SDL_GetGamepadAxis(gp, a) / 32767.f; }

// radial deadzone with rescale so motion starts smoothly at the edge
void stick(float x, float y, float out[2]) {
    const float dz = 0.15f;
    float m = sqrtf(x * x + y * y);
    if (m < dz) { out[0] = out[1] = 0; return; }
    float s = std::fmin(1.f, (m - dz) / (1.f - dz)) / m;
    out[0] = x * s;
    out[1] = y * s;
}

void open(SDL_JoystickID id) {
    if (gp) return;
    gp = SDL_OpenGamepad(id);
    if (!gp) return;
    state.hasGyro = SDL_GamepadHasSensor(gp, SDL_SENSOR_GYRO) && SDL_SetGamepadSensorEnabled(gp, SDL_SENSOR_GYRO, true);
    printf("gamepad: %s%s\n", SDL_GetGamepadName(gp), state.hasGyro ? " (gyro)" : "");
}
}  // namespace

void init() {
    // read Steam Controllers directly (SDL leaves them to Steam by default)
    SDL_SetHint(SDL_HINT_JOYSTICK_HIDAPI_STEAM, "1");
    SDL_SetHint(SDL_HINT_JOYSTICK_HIDAPI_STEAM_HORI, "1");
    SDL_SetHint(SDL_HINT_JOYSTICK_ALLOW_BACKGROUND_EVENTS, "1");
    if (!SDL_Init(SDL_INIT_GAMEPAD | SDL_INIT_SENSOR)) {
        fprintf(stderr, "SDL gamepad init failed: %s\n", SDL_GetError());
        return;
    }
    int n = 0;
    SDL_JoystickID* ids = SDL_GetGamepads(&n);
    if (ids && n > 0) open(ids[0]);
    SDL_free(ids);
}

void shutdown() {
    if (gp) SDL_CloseGamepad(gp);
    gp = nullptr;
    SDL_Quit();
}

void update() {
    SDL_Event e;
    while (SDL_PollEvent(&e)) {
        if (e.type == SDL_EVENT_GAMEPAD_ADDED) open(e.gdevice.which);
        else if (e.type == SDL_EVENT_GAMEPAD_REMOVED && gp && SDL_GetGamepadID(gp) == e.gdevice.which) {
            SDL_CloseGamepad(gp);
            gp = nullptr;
        }
    }
    bool had = state.hasGyro;
    state = {};
    state.hasGyro = had && gp;
    if (!gp) return;
    state.connected = true;
    state.name = SDL_GetGamepadName(gp);
    stick(axis(SDL_GAMEPAD_AXIS_LEFTX), -axis(SDL_GAMEPAD_AXIS_LEFTY), state.move);
    stick(axis(SDL_GAMEPAD_AXIS_RIGHTX), -axis(SDL_GAMEPAD_AXIS_RIGHTY), state.look);
    state.lt = std::fmax(0.f, axis(SDL_GAMEPAD_AXIS_LEFT_TRIGGER));
    state.rt = std::fmax(0.f, axis(SDL_GAMEPAD_AXIS_RIGHT_TRIGGER));
    state.jump = SDL_GetGamepadButton(gp, SDL_GAMEPAD_BUTTON_SOUTH);
    state.sprint = SDL_GetGamepadButton(gp, SDL_GAMEPAD_BUTTON_LEFT_STICK) || state.lt > 0.5f;
    state.up = SDL_GetGamepadButton(gp, SDL_GAMEPAD_BUTTON_RIGHT_SHOULDER);
    state.down = SDL_GetGamepadButton(gp, SDL_GAMEPAD_BUTTON_LEFT_SHOULDER);
    bool toggle = SDL_GetGamepadButton(gp, SDL_GAMEPAD_BUTTON_NORTH);
    state.togglePressed = toggle && !prevToggle;
    prevToggle = toggle;
    if (state.hasGyro) SDL_GetGamepadSensorData(gp, SDL_SENSOR_GYRO, state.gyro, 3);
}

const Pad& pad() { return state; }

void rumble(float low, float high, unsigned ms) {
    if (gp) SDL_RumbleGamepad(gp, (Uint16)(low * 65535.f), (Uint16)(high * 65535.f), ms);
}
}  // namespace input

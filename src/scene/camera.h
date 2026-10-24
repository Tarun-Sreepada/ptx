#pragma once
#include <cmath>
#include "core/math.h"

struct Camera {
    V3 pos{0, 2.2f, -18.f};
    float yaw = 0.f, pitch = -0.08f, fovDeg = 60.f;
    V3 fwd() const { return {cosf(pitch) * sinf(yaw), sinf(pitch), cosf(pitch) * cosf(yaw)}; }
    V3 right() const { return normalize(cross(fwd(), {0, 1, 0})); }
    V3 up() const { return cross(right(), fwd()); }
};

extern Camera cam, prevCam;

extern bool walkMode, grounded, walkReset;

extern float flySpeed;

extern bool gyroAim;

extern float gyroSens;

extern V3 playerFeet;

void updateCamera(float dt);

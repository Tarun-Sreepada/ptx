#pragma once
#include <cmath>
#include "core/math.h"

struct Camera {
    V3 pos{0, 2.2f, -18.f};
    float yaw = 0.f, pitch = -0.08f, fovDeg = 60.f;
    float fovBoost = 0.f;  // deg added on top of fovDeg (walking fast)
    float fov() const { return fovDeg + fovBoost; }
    V3 fwd() const { return {cosf(pitch) * sinf(yaw), sinf(pitch), cosf(pitch) * cosf(yaw)}; }
    V3 right() const { return normalize(cross(fwd(), {0, 1, 0})); }
    V3 up() const { return cross(right(), fwd()); }
};

extern Camera cam, prevCam;

extern bool walkMode, grounded, walkReset;

extern float flySpeed;

// 0..1 along a logistic curve over 0..1 (normalised so the ends are exactly 0 and 1): flat at the start, steepest
// in the middle, easing into the top. Speed gain and the speed FOV both follow it.
inline float sigmoid01(float t) {
    const float K = 8.f;
    auto s = [](float x) { return 1.f / (1.f + expf(-x)); };
    float lo = s(-0.5f * K), hi = s(0.5f * K);
    return (s(K * (std::fmin(std::fmax(t, 0.f), 1.f) - 0.5f)) - lo) / (hi - lo);
}
// the walking top speed (m/s): the setting, or effectively none when uncapped
float walkMaxSpeed();

extern bool gyroAim;

extern float gyroSens;

extern V3 playerFeet;

void updateCamera(float dt);

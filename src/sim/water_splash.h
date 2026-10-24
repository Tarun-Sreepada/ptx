#pragma once
#include <vector>
#include "core/math.h"

// water impacts for water.glsl (crown, cavity, rings, foam): every entry, skip and wake puff of anything thrown
struct SplashEvent { float x, z, t, impact, sc, speed; };

inline constexpr int MAX_SPLASH_EV = 64;

inline constexpr float SPLASH_LIFE = 40.f, SPLASH_FOAM_TAU = 12.f;  // s, water.glsl dropLocal / dropFoam

extern std::vector<SplashEvent> splashEvents;

void waterImpact(V3 at, float sc, const float v[3], float scale = 1.f);

// Physical constants (SI units) the simulation shares. Tuning values that aren't nature's live with the code that
// uses them, as named constants at the top of its file.
#pragma once

namespace phys {
inline constexpr float GRAVITY = 9.81f;              // m/s^2
inline constexpr float WATER_DENSITY = 1000.f;       // kg/m^3
inline constexpr float AIR_DENSITY = 1.225f;         // kg/m^3 at sea level, 15 C
inline constexpr float ROCK_DENSITY = 2600.f;        // kg/m^3 (granite ~2650, sandstone ~2300)
inline constexpr float SPEED_OF_SOUND = 343.f;       // m/s in air at 20 C (thunder delay)
inline constexpr float WATER_SURFACE_TENSION = 0.0728f;  // N/m
inline constexpr float PI = 3.14159265358979f;
inline constexpr float DEG = PI / 180.f;             // degrees -> radians
}  // namespace phys

#include "sky/day_cycle.h"

#include <algorithm>
#include <cmath>

#include "app/settings.h"
#include "core/math.h"

namespace {
constexpr float DEG = 3.14159265f / 180.f;
constexpr float AXIAL_TILT = 23.44f;      // degrees
constexpr float DAYS_PER_YEAR = 365.25f;
constexpr float SOLSTICE_DAY = 172.f;     // day of year of the June solstice
constexpr float LUNAR_MONTH = 29.53f;     // days from new moon to new moon
constexpr float COLDEST_DAY = 20.f;       // day of year (late January)
constexpr float WINTER_SNOW_LINE = -30.f, SUMMER_SNOW_LINE = 3500.f;  // m
}  // namespace

void sunAt(float hour, float dayOfYear, float latitude, float& elevDeg, float& azimDeg) {
    float phi = latitude * DEG;
    float decl = AXIAL_TILT * DEG * cosf(2.f * 3.14159265f * (dayOfYear - SOLSTICE_DAY) / DAYS_PER_YEAR);
    // at noon the sun stands on the meridian toward the equator (+z here, north is -z), 90 - latitude + declination
    // high; through the day it turns westward about the celestial pole (hour angle 15 degrees per hour)
    V3 noon{0.f, cosf(phi - decl), sinf(phi - decl)};
    V3 pole{0.f, sinf(phi), -cosf(phi)};
    V3 d = rotateAbout(noon, pole, -(hour - 12.f) * 15.f * DEG);
    elevDeg = asinf(std::clamp(d.y, -1.f, 1.f)) / DEG;
    azimDeg = atan2f(d.x, d.z) / DEG;
    if (azimDeg < 0.f) azimDeg += 360.f;
}

float seasonSnowLine(float dayOfYear) {
    float warmth = 0.5f - 0.5f * cosf(2.f * 3.14159265f * (dayOfYear - COLDEST_DAY) / DAYS_PER_YEAR);  // 0 coldest .. 1 warmest
    float k = std::clamp((warmth - 0.15f) / 0.4f, 0.f, 1.f);
    return WINTER_SNOW_LINE + (SUMMER_SNOW_LINE - WINTER_SNOW_LINE) * k * k * (3.f - 2.f * k);
}

void advanceDay(float dt) {
    if (!S.dayCycle) return;
    float hours = dt * 24.f / (std::max(S.dayMinutes, 0.1f) * 60.f);
    S.timeOfDay += hours;
    S.moonPhase = fmodf(S.moonPhase + hours / 24.f / LUNAR_MONTH, 1.f);
    while (S.timeOfDay >= 24.f) { S.timeOfDay -= 24.f; S.dayOfYear = fmodf(S.dayOfYear + 1.f, DAYS_PER_YEAR); }
    sunAt(S.timeOfDay, S.dayOfYear, S.latitude, S.sunElev, S.sunAzim);
}

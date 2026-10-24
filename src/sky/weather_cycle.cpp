#include "sky/weather_cycle.h"

#include <algorithm>
#include <cmath>

#include "app/settings.h"
#include "core/random.h"

namespace {
constexpr float HOLD_MIN = 240.f, HOLD_MAX = 720.f;  // s a weather lasts (real time, at weatherChange 1)
constexpr float TRANSITION = 150.f;                  // s to change over
constexpr float FOG_ON = 0.0012f;                    // fog density (1/m) from which the volumetric fog is switched on

struct Weather {
    const char* name;
    float weight;  // how common
    float coverage, cumulonimbus, stratus, cirrus, rain, wind, fog;
    bool banks;
};
const Weather KINDS[] = {
    {"Clear", 1.0f, 0.05f, 0.0f, 0.0f, 0.35f, 0.f, 0.2f, 0.0002f, false},
    {"Fair", 1.6f, 0.35f, 0.1f, 0.25f, 0.6f, 0.f, 0.3f, 0.0003f, false},
    {"Cloudy", 1.0f, 0.7f, 0.2f, 0.7f, 0.4f, 0.f, 0.45f, 0.0005f, false},
    {"Overcast", 0.7f, 0.95f, 0.1f, 1.0f, 0.2f, 0.05f, 0.4f, 0.0008f, false},
    {"Rain", 0.7f, 0.9f, 0.4f, 0.9f, 0.1f, 0.55f, 0.55f, 0.001f, false},
    {"Thunderstorm", 0.35f, 0.85f, 1.0f, 0.5f, 0.1f, 0.9f, 0.8f, 0.0008f, false},
    {"Fog", 0.5f, 0.5f, 0.0f, 0.8f, 0.2f, 0.f, 0.1f, 0.004f, true},
};
constexpr int KIND_COUNT = (int)(sizeof KINDS / sizeof KINDS[0]);
enum { CLEAR, FAIR, CLOUDY, OVERCAST, RAIN, STORM, FOG };

struct State { float coverage, cumulonimbus, stratus, cirrus, rain, wind, fog; };
State from{}, to{};
int target = FAIR;
float clock = 0.f, hold = 0.f;
bool running = false;

State current() { return {S.cloudCoverage, S.cumulonimbus, S.stratus, S.cirrus, S.rain, S.windStrength, S.fogDensity}; }

int pickNext(int now) {
    float w[KIND_COUNT];
    float morning = S.dayCycle ? expf(-(S.timeOfDay - 7.f) * (S.timeOfDay - 7.f) / 4.f) : 0.3f;
    float summer = S.dayCycle ? std::clamp(1.f - fabsf(S.dayOfYear - 200.f) / 80.f, 0.f, 1.f) : 0.5f;
    float afternoon = S.dayCycle ? expf(-(S.timeOfDay - 16.f) * (S.timeOfDay - 16.f) / 8.f) : 0.5f;
    float sum = 0.f;
    for (int k = 0; k < KIND_COUNT; k++) {
        w[k] = KINDS[k].weight;
        if (k == FOG) w[k] *= 0.3f + 3.f * morning;
        if (k == STORM) w[k] *= 0.3f + 2.f * summer * afternoon;
        if (abs(k - now) == 1 && k != FOG && now != FOG) w[k] *= 2.f;  // weather mostly drifts to its neighbours
        if (k == now) w[k] *= 0.3f;
        sum += w[k];
    }
    float r = rnd01() * sum;
    for (int k = 0; k < KIND_COUNT; k++) if ((r -= w[k]) <= 0.f) return k;
    return FAIR;
}

void beginFront(int k) {
    target = k;
    from = current();
    const Weather& w = KINDS[k];
    to = {w.coverage, w.cumulonimbus, w.stratus, w.cirrus, w.rain, w.wind, w.fog};
    clock = 0.f;
    hold = (HOLD_MIN + (HOLD_MAX - HOLD_MIN) * rnd01()) / std::max(S.weatherChange, 0.05f);
}
}  // namespace

void advanceWeather(float dt) {
    if (!S.weatherCycle) { running = false; return; }
    if (!running) { running = true; beginFront(pickNext(FAIR)); }
    clock += dt;
    float k = std::clamp(clock / (TRANSITION / std::max(S.weatherChange, 0.05f)), 0.f, 1.f);
    k = k * k * (3.f - 2.f * k);
    auto mix = [&](float a, float b) { return a + (b - a) * k; };
    S.cloudCoverage = mix(from.coverage, to.coverage);
    S.cumulonimbus = mix(from.cumulonimbus, to.cumulonimbus);
    S.stratus = mix(from.stratus, to.stratus);
    S.cirrus = mix(from.cirrus, to.cirrus);
    S.rain = mix(from.rain, to.rain);
    S.windStrength = mix(from.wind, to.wind);
    S.fogDensity = mix(from.fog, to.fog);
    S.fog = S.fogDensity > FOG_ON;
    S.fogBanks = KINDS[target].banks && k > 0.3f;
    S.clouds = true;
    if (clock > hold) beginFront(pickNext(target));
}

const char* weatherName() { return KINDS[target].name; }

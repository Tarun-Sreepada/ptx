#include "app/settings_io.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <type_traits>
#include <vector>

#include "app/settings.h"
#include "vegetation/trees.h"

namespace {
constexpr const char* SETTINGS_FILE = "settings.ini";
constexpr double SAVE_INTERVAL = 5.0;  // s between checks for changes

struct Field { const char* name; char type; int count; size_t offset; };  // type: f float, i int, b bool, u uint32
template <class T> constexpr char typeOf() {
    using E = std::remove_extent_t<T>;
    if constexpr (std::is_same_v<E, float>) return 'f';
    else if constexpr (std::is_same_v<E, bool>) return 'b';
    else if constexpr (std::is_same_v<E, uint32_t>) return 'u';
    else { static_assert(std::is_same_v<E, int>, "settings fields are float, int, bool or uint32_t"); return 'i'; }
}
#define FIELD(T, x) Field{#x, typeOf<decltype(T::x)>(), (int)std::max<size_t>(std::extent_v<decltype(T::x)>, 1), offsetof(T, x)}
#define SF(x) FIELD(Settings, x)
#define TF(x) FIELD(trees::TreeParams, x)
#define BF(x) FIELD(trees::BushParams, x)

// every Settings field but the debug view mode
const Field SETTINGS_FIELDS[] = {
    SF(raysFirst), SF(bounces), SF(rasterPrimary), SF(restir), SF(restirTemporal), SF(restirSpatial),
    SF(candidates), SF(spatialSamples), SF(spatialRadius), SF(restirGI), SF(giTemporal), SF(giSpatial),
    SF(giEnabled), SF(giCache), SF(giChecker), SF(giProbes), SF(giSpatialSamples), SF(giRadius),
    SF(giMaxM), SF(accumulate), SF(denoise), SF(historyMax), SF(atrousIters), SF(fireflyClamp),
    SF(sunElev), SF(sunAzim), SF(sunIntensity), SF(sunRadiusDeg), SF(skyIntensity), SF(fogScale),
    SF(dayCycle), SF(timeOfDay), SF(dayMinutes), SF(dayOfYear), SF(weatherCycle), SF(weatherChange),
    SF(clouds), SF(groundTextures), SF(parallax), SF(groundNormal), SF(parallaxDist), SF(groundMode),
    SF(vtBudget), SF(fractalRelief), SF(fractalPebbles), SF(fractalFinest), SF(windMap), SF(gustStrength),
    SF(gustSpreadDeg), SF(windWaveSpeed), SF(windDamping), SF(windRelax), SF(cloudCoverage), SF(cloudDensity),
    SF(cloudSpeed), SF(cloudBase), SF(cloudThickness), SF(cloudShadow), SF(cumulus), SF(stratus),
    SF(stratusLumpy), SF(stratusBase), SF(stratusThick), SF(cumulonimbus), SF(cbTop), SF(cirrus),
    SF(cirrusAlt), SF(cirrusLength), SF(cirrusFine), SF(cirrusPuff), SF(grassClump), SF(grassClumpSize),
    SF(grassWidth), SF(grassCurve), SF(grassTwist), SF(grassTilt), SF(grassHeightVar), SF(grassDead),
    SF(grassColorVar), SF(groundWarp), SF(groundMottle), SF(groundMottleScale), SF(darkSpots), SF(darkSpotSize),
    SF(groundGrit), SF(groundCavity), SF(tileBreakup), SF(groundSat), SF(bushAmount), SF(berryShare),
    SF(fovMargin), SF(keepRadius), SF(offscreenDensity), SF(headBob), SF(wildlife), SF(critters), SF(ambience), SF(grassDensity), SF(grassLength), SF(lodStart),
    SF(drawDist), SF(craterInflow), SF(riverRainRise), SF(groundFriction), SF(groundBounce), SF(worldSim),
    SF(snowFollowsClouds), SF(snowLine), SF(snowfall), SF(meltRate), SF(reposeDeg), SF(avalanche),
    SF(riverRange), SF(dryRate), SF(waterMobility), SF(mapTexTile), SF(simSpeed), SF(snowMeshEvery),
    SF(viewDist), SF(treeDensity), SF(impostorDist), SF(clutter), SF(clutterRange), SF(lightCount),
    SF(orbRange), SF(lightIntensity), SF(windStrength), SF(windSpeed), SF(windDirDeg), SF(exposure),
    SF(autoExposure), SF(evComp), SF(adaptUp), SF(adaptDown), SF(physCam), SF(fStop),
    SF(shutterDen), SF(iso), SF(sceneLum), SF(dof), SF(motionBlur), SF(focusDist),
    SF(mbShutter), SF(vignette), SF(chromAb), SF(grain), SF(tonemapper), SF(bloom),
    SF(sharpen), SF(hdrMode), SF(paperWhite), SF(peakNits), SF(grassRefit), SF(fog),
    SF(fogLights), SF(fogDiv), SF(fogDensity), SF(fogFalloff), SF(fogAniso), SF(fogNoise),
    SF(fogDist), SF(fogBlend), SF(rain), SF(wetness), SF(puddles), SF(rainWets),
    SF(soak), SF(drip), SF(rivulets), SF(caustics), SF(decals), SF(foamTrail),
    SF(flow), SF(clarity), SF(foam), SF(splash), SF(absorb), SF(murk),
    SF(throwSize), SF(throwSpeed), SF(sss), SF(snowSSS), SF(microShadow),
    SF(sssSteps), SF(sssScale), SF(moonPhase), SF(latitude), SF(nightBoost), SF(stars),
    SF(milkyWay), SF(airglow), SF(twinkle), SF(purkinje), SF(lightning), SF(strikesPerMin),
    SF(rainbow), SF(dropMM), SF(fogBanks), SF(bankDensity), SF(bankScale), SF(bankTop),
    SF(bankCover), SF(bankDetail),
};
const Field TREE_FIELDS[] = {
    TF(height), TF(trunkRadius), TF(bole), TF(lean), TF(gnarl), TF(flare),
    TF(trunkWave), TF(levels), TF(branches), TF(children), TF(branchAngle), TF(angleVar),
    TF(lengthRatio), TF(childLength), TF(radiusRatio), TF(gravity), TF(upturn), TF(crown),
    TF(leafSize), TF(leafWidth), TF(leafDensity), TF(leafSpread), TF(leafDroop), TF(leafColor),
    TF(hueVar), TF(barkColor), TF(density), TF(strength), TF(growth), TF(seed),
};
const Field BUSH_FIELDS[] = {
    BF(size), BF(stems), BF(spread), BF(branchAngle), BF(gnarl), BF(levels),
    BF(leafSize), BF(leafDensity), BF(leafColor), BF(fruitDensity), BF(fruitSize), BF(fruitColor),
    BF(seed),
};

bool persist = true;
std::map<std::string, std::string> loaded;  // key -> value text as read
Settings fileS, flagS;                      // S as the file left it, and after the command line
std::string lastWritten;
double nextCheck = 0.0;

std::string format(const Field& f, const void* base) {
    const char* p = (const char*)base + f.offset;
    std::string out;
    char buf[48];
    for (int k = 0; k < f.count; k++) {
        if (k) out += ' ';
        switch (f.type) {
        case 'f': snprintf(buf, sizeof buf, "%.9g", (double)((const float*)p)[k]); break;
        case 'b': snprintf(buf, sizeof buf, "%d", ((const bool*)p)[k] ? 1 : 0); break;
        case 'u': snprintf(buf, sizeof buf, "%u", ((const uint32_t*)p)[k]); break;
        default: snprintf(buf, sizeof buf, "%d", ((const int*)p)[k]); break;
        }
        out += buf;
    }
    return out;
}

void parse(const Field& f, void* base, const std::string& text) {
    char* p = (char*)base + f.offset;
    const char* s = text.c_str();
    for (int k = 0; k < f.count; k++) {
        char* end;
        double v = strtod(s, &end);
        if (end == s) return;  // fewer values than expected: keep the rest
        switch (f.type) {
        case 'f': ((float*)p)[k] = (float)v; break;
        case 'b': ((bool*)p)[k] = v != 0.0; break;
        case 'u': ((uint32_t*)p)[k] = (uint32_t)v; break;
        default: ((int*)p)[k] = (int)v; break;
        }
        s = end;
    }
}

bool same(const Field& f, const void* a, const void* b) {
    size_t n = (size_t)f.count * (f.type == 'b' ? sizeof(bool) : 4);
    return !memcmp((const char*)a + f.offset, (const char*)b + f.offset, n);
}

std::string compose() {
    std::string out = "# ptx settings, written by the game (delete this file to go back to the defaults)\n";
    for (const Field& f : SETTINGS_FIELDS) {
        // a flag's value isn't remembered: unless it was changed again since, write what the file had
        bool flagOnly = same(f, &S, &flagS) && !same(f, &flagS, &fileS);
        out += std::string(f.name) + " = " + format(f, flagOnly ? (const void*)&fileS : (const void*)&S) + "\n";
    }
    for (int s = 0; s < trees::SPECIES; s++)
        for (const Field& f : TREE_FIELDS) out += "tree." + std::to_string(s) + "." + f.name + " = " + format(f, &trees::treeParams[s]) + "\n";
    for (int k = 0; k < trees::BUSH_KINDS; k++)
        for (const Field& f : BUSH_FIELDS) out += "bush." + std::to_string(k) + "." + f.name + " = " + format(f, &trees::bushParams[k]) + "\n";
    return out;
}
}  // namespace

void setSettingsPersistence(bool on) { persist = on; }

void loadSettings() {
    fileS = flagS = S;
    if (!persist) return;
    FILE* fp = fopen(SETTINGS_FILE, "r");
    if (!fp) return;
    char line[512];
    std::string text;
    while (fgets(line, sizeof line, fp)) {
        std::string l = line;
        text += l;
        if (l.empty() || l[0] == '#') continue;
        size_t eq = l.find('=');
        if (eq == std::string::npos) continue;
        auto trim = [](std::string s) {
            size_t a = s.find_first_not_of(" \t\r\n"), b = s.find_last_not_of(" \t\r\n");
            return a == std::string::npos ? std::string() : s.substr(a, b - a + 1);
        };
        loaded[trim(l.substr(0, eq))] = trim(l.substr(eq + 1));
    }
    fclose(fp);
    lastWritten = text;
    for (const Field& f : SETTINGS_FIELDS) {
        auto it = loaded.find(f.name);
        if (it != loaded.end()) parse(f, &S, it->second);
    }
    fileS = flagS = S;
    printf("settings: %s (%zu values)\n", SETTINGS_FILE, loaded.size());
}

void markFlagsApplied() { flagS = S; }

void applyVegetationSettings() {
    if (!persist) return;
    for (int s = 0; s < trees::SPECIES; s++)
        for (const Field& f : TREE_FIELDS) {
            auto it = loaded.find("tree." + std::to_string(s) + "." + f.name);
            if (it != loaded.end()) parse(f, &trees::treeParams[s], it->second);
        }
    for (int k = 0; k < trees::BUSH_KINDS; k++)
        for (const Field& f : BUSH_FIELDS) {
            auto it = loaded.find("bush." + std::to_string(k) + "." + f.name);
            if (it != loaded.end()) parse(f, &trees::bushParams[k], it->second);
        }
}

void saveSettings() {
    if (!persist) return;
    std::string text = compose();
    if (text == lastWritten) return;
    std::string tmp = std::string(SETTINGS_FILE) + ".tmp";
    FILE* fp = fopen(tmp.c_str(), "w");
    if (!fp) return;
    bool ok = fwrite(text.data(), 1, text.size(), fp) == text.size();
    ok = fclose(fp) == 0 && ok;
    if (ok && rename(tmp.c_str(), SETTINGS_FILE) == 0) lastWritten = text;  // whole file or nothing
}

void saveSettingsIfChanged(double now) {
    if (!persist || now < nextCheck) return;
    nextCheck = now + SAVE_INTERVAL;
    saveSettings();
}

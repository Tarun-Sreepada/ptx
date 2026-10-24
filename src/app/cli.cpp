#include "app/cli.h"
#include "sim/fire.h"
#include "sky/day_cycle.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>

#include "app/settings.h"
#include "app/settings_io.h"
#include "app/window.h"
#include "core/math.h"
#include "core/paths.h"
#include "destruction/thrown.h"
#include "editor/editor.h"
#include "gpu/dlss.h"
#include "gpu/swapchain.h"
#include "render/bake.h"
#include "render/frame_graph.h"
#include "render/ground_textures.h"
#include "render/targets.h"
#include "scene/camera.h"
#include "scene/characters.h"
#include "scene/grass.h"
#include "scene/stream_sync.h"
#include "scene/vegetation.h"
#include "tools/movelog.h"
#include "world/texcomp.h"
#include "world/terrain.h"

int profFrames = 0;
bool mapTest = false;
int ragdollFrame = -1;
int mapClickFrame = -1;
int editDemoFrame = -1, editSaveFrame = -1;

namespace {
constexpr float DEG = 3.14159265f / 180.f;
constexpr float CELL_METRES = 64.f;   // streamed cell size: view distance -> load radius in cells
constexpr int MAX_LOAD_RADIUS = 96;

float f(const char* s) { return (float)atof(s); }
int i(const char* s) { return atoi(s); }

struct Flag {
    const char* name;
    int args;            // values that follow the flag
    const char* help;
    std::function<void(char** a)> apply;  // a[0..args-1] = the values
};

std::vector<Flag> flags(LaunchOptions& o) {
    return {
        // ---- run modes / output
        {"--defaults", 0, "start from the default settings, don't touch settings.ini", [](char**) {}},
        {"--world", 1, "<dir>  world folder (cells/, assets/, terrain/)", [&](char** a) { o.worldDir = a[0]; if (o.worldDir.back() != '/') o.worldDir += '/'; }},
        {"--validate", 0, "Vulkan validation layer", [&](char**) { o.validate = true; }},
        {"--shot", 1, "<file.png>  screenshot after --shot-frames frames, then exit", [&](char** a) { o.shotPath = a[0]; }},
        {"--shot-frames", 1, "<n>  frames before --shot (default 240)", [&](char** a) { o.shotFrames = i(a[0]); }},
        {"--shot-seq", 1, "<n>  also save the n frames before the shot (<path>.<k>)", [&](char** a) { o.shotSeq = i(a[0]); }},
        {"--bench", 0, "run the benchmark, print the results, exit", [&](char**) { o.benchOnly = true; vsync = false; }},
        {"--prof", 1, "<n>  capture n frames from frame 30, export a Chrome trace at exit", [](char** a) { profFrames = i(a[0]); }},
        {"--from-shot", 1, "<file.png|txt>  reopen a screenshot's scene", [](char** a) { loadShotInfo(a[0]); }},
        {"--replay", 1, "<log>  drive the camera from a movement log", [](char** a) { loadReplay(a[0]); }},
        {"--replay-shots", 0, "retake the log's screenshots while replaying", [](char**) { replayShots = true; }},
        {"--replay-exit", 0, "quit at the end of the replay", [](char**) { replayExit = true; }},
        // ---- window / display
        {"--borderless", 0, "borderless window filling the screen", [](char**) { windowMode = 1; }},
        {"--fullscreen", 0, "exclusive fullscreen", [](char**) { windowMode = 2; }},
        {"--hdr", 0, "HDR10 output when the display offers it", [](char**) { S.hdrMode = 1; }},
        {"--dlss", 1, "<0..5>  DLSS-RR mode (0 off, 1 DLAA, 2 quality, 3 balanced, 4 performance, 5 ultra perf.)",
         [](char** a) { upscaler = (dlss::Mode)std::clamp(i(a[0]), 0, (int)dlss::MODE_COUNT - 1); }},
        // ---- camera
        {"--pos", 3, "<x y z>  camera position", [](char** a) { cam.pos = {f(a[0]), f(a[1]), f(a[2])}; }},
        {"--look", 2, "<yaw pitch>  camera direction (degrees)", [](char** a) { cam.yaw = f(a[0]) * DEG; cam.pitch = f(a[1]) * DEG; }},
        {"--walk", 0, "start walking (first person)", [](char**) { walkMode = true; }},
        // ---- rendering
        {"--rays", 1, "<n>  rays per pixel on the first bounce", [](char** a) { S.raysFirst = i(a[0]); }},
        {"--view", 1, "<n>  debug view (0 final, 1 raw, 2 direct, 3 indirect, 4 albedo, 5 normals, ...)", [](char** a) { S.viewMode = i(a[0]); }},
        {"--traced-primary", 0, "path-traced camera rays instead of the raster G-buffer", [](char**) { S.rasterPrimary = false; }},
        {"--no-restir-gi", 0, "", [](char**) { S.restirGI = false; }},
        {"--no-gi", 0, "direct light only", [](char**) { S.giEnabled = false; }},
        {"--no-gi-cache", 0, "", [](char**) { S.giCache = false; }},
        {"--no-gi-checker", 0, "", [](char**) { S.giChecker = false; }},
        {"--no-gi-probes", 0, "", [](char**) { S.giProbes = false; }},
        {"--no-baked-gi", 0, "ignore the baked interior probes", [](char**) { bakedGI = false; }},
        {"--no-async", 0, "no async compute (BVH build on the graphics queue)", [](char**) { asyncCompute = false; }},
        {"--no-async-fog", 0, "", [](char**) { asyncFog = false; }},
        {"--no-sss", 0, "no subsurface scattering", [](char**) { S.sss = false; }},
        {"--no-micro", 0, "no micro-shadowing / specular occlusion", [](char**) { S.microShadow = false; }},
        {"--no-grass-refit", 0, "rebuild the grass BLAS every frame", [](char**) { S.grassRefit = false; }},
        {"--no-bc7", 0, "upload textures as RGBA8 (no BC7 compression)", [](char**) { texcomp::setEnabled(false); }},
        {"--max-blades", 1, "<n>  grass blade capacity", [](char** a) { maxBlades = (uint32_t)i(a[0]); }},
        {"--ground-tex", 1, "<n>  generated ground texture size", [](char** a) { groundTexSize = std::clamp((uint32_t)i(a[0]), 256u, 8192u); }},
        {"--ground-mode", 1, "<n>  0 tiled textures, 1 fractal, 2 hybrid, 3 layered virtual texture", [](char** a) { S.groundMode = i(a[0]); }},
        {"--view-dist", 1, "<m>  view distance", [](char** a) {
             S.viewDist = f(a[0]);
             loadRadius = std::clamp((int)ceilf(S.viewDist / CELL_METRES), 1, MAX_LOAD_RADIUS);
         }},
        // ---- camera / lens
        {"--dof", 2, "<focus m (0 auto)> <f-stop>  depth of field", [](char** a) { S.dof = true; S.focusDist = f(a[0]); S.fStop = f(a[1]); }},
        {"--motion-blur", 1, "<shutter fraction>", [](char** a) { S.motionBlur = true; S.mbShutter = f(a[0]); }},
        {"--phys-cam", 3, "<f-stop> <shutter 1/x s> <ISO>  physical camera exposure",
         [](char** a) { S.physCam = true; S.fStop = f(a[0]); S.shutterDen = f(a[1]); S.iso = f(a[2]); }},
        {"--lens-fx", 3, "<vignette> <chromatic aberration px> <grain>", [](char** a) { S.vignette = f(a[0]); S.chromAb = f(a[1]); S.grain = f(a[2]); }},
        // ---- sky / weather
        {"--sun", 2, "<elevation> <azimuth>  degrees", [](char** a) { S.sunElev = f(a[0]); S.sunAzim = f(a[1]); }},
        {"--night", 0, "sun below the horizon, orbs brighter", [](char**) { S.sunElev = -8; S.lightIntensity = 3; S.exposure = 3; }},
        {"--time", 1, "<hours>  day / night cycle running from this time of day", [](char** a) {
             S.dayCycle = true; S.timeOfDay = f(a[0]); sunAt(S.timeOfDay, S.dayOfYear, S.latitude, S.sunElev, S.sunAzim);
         }},
        {"--day-length", 1, "<minutes>  real minutes per game day", [](char** a) { S.dayMinutes = f(a[0]); }},
        {"--weather-cycle", 0, "weather changes by itself (fronts: clear, cloud, rain, storms, fog)", [](char**) { S.weatherCycle = true; }},
        {"--rain", 1, "<0..1>  precipitation (wets everything at once)", [](char** a) {
             S.rain = f(a[0]);
             S.wetness = std::min(S.rain * 1.5f, 1.f); S.soak = std::min(S.rain * 2.f, 1.f); S.drip = std::min(S.rain * 3.f, 1.f);
         }},
        {"--wet", 3, "<film> <soak> <drip>  surface water without rain (drying state)",
         [](char** a) { S.wetness = f(a[0]); S.soak = f(a[1]); S.drip = f(a[2]); S.rainWets = false; }},
        {"--fog", 1, "<density /m>  volumetric fog", [](char** a) { S.fogDensity = f(a[0]); S.fog = S.fogDensity > 0.f; }},
        {"--fog-div", 1, "<1..4>  fog resolution divisor", [](char** a) { S.fogDiv = i(a[0]); }},
        {"--snow-line", 1, "<m>  fixed snow line", [](char** a) { S.snowLine = f(a[0]); S.snowFollowsClouds = false; }},
        {"--snowfall", 1, "<m/h>  at full precipitation", [](char** a) { S.snowfall = f(a[0]); }},
        {"--sim-speed", 1, "<x>  world simulation time scale", [](char** a) { S.simSpeed = f(a[0]); }},
        // ---- world / interaction tests
        {"--edit", 0, "open the editor", [](char**) { editor::setActive(true); }},
        {"--edit-demo", 1, "<frame>  build the example house", [](char** a) { editDemoFrame = i(a[0]); }},
        {"--edit-save", 1, "<frame>  save editor changes", [](char** a) { editSaveFrame = i(a[0]); }},
        {"--map-test", 0, "sculpt a hill and carve a river on the world map", [](char**) { mapTest = true; }},
        {"--map-click", 1, "<frame>  raise ground through the editor UI", [](char** a) { mapClickFrame = i(a[0]); editor::setActive(true); }},
        {"--crater", 3, "<x z r>  a crater (repeatable)", [](char** a) {
             int n = gCraterCount.load();
             float r = f(a[2]);
             if (n < MAX_CRATERS) { gCraters[n] = Crater{f(a[0]), f(a[1]), r, CRATER_DEPTH_RATIO * r}; gCraterCount.store(n + 1); }
         }},
        {"--throw", 1, "<n>  throw n rocks, one every 20 frames", [](char** a) { autoThrow = i(a[0]); }},
        {"--tree-throw", 1, "<n>  throw n rocks at the nearest tree, log every hit", [](char** a) { treeThrow = i(a[0]); treeLog = true; }},
        {"--throw-size", 1, "<m>", [](char** a) { S.throwSize = f(a[0]); }},
        {"--throw-speed", 1, "<m/s>", [](char** a) { S.throwSpeed = f(a[0]); }},
        {"--fire", 3, "<x z radius>  a fire burning from the start (repeatable)", [](char** a) {
             requestFire(f(a[0]), f(a[1]), f(a[2]), 600.f);
         }},
        {"--veg-regen", 1, "<frame>  regrow the vegetation", [](char** a) { vegRegenFrame = i(a[0]); }},
        {"--lights-off", 0, "start with interior lights off", [](char**) { startLightsOff = true; }},
        {"--character", 3, "<asset> <x> <z>  spawn a character (--character-* below apply to the last one)", [](char** a) {
             charSpawns.push_back({a[0], "", {f(a[1]), 0.f, f(a[2])}, 0.f, 0.f, 0.f});
         }},
        {"--character-clip", 1, "<name>", [](char** a) { if (!charSpawns.empty()) charSpawns.back().clip = a[0]; }},
        {"--character-walk", 1, "<m/s>", [](char** a) { if (!charSpawns.empty()) charSpawns.back().walk = f(a[0]); }},
        {"--character-yaw", 1, "<degrees>", [](char** a) { if (!charSpawns.empty()) charSpawns.back().yaw = f(a[0]) * DEG; }},
        {"--character-height", 1, "<m>", [](char** a) { if (!charSpawns.empty()) charSpawns.back().height = f(a[0]); }},
        {"--ragdoll-at", 1, "<frame>  every character goes limp", [](char** a) { ragdollFrame = i(a[0]); }},
    };
}
}  // namespace

// Runs that must start from the defaults and leave settings.ini alone: tests, captures, replays.
const char* const NO_SETTINGS_FLAGS[] = {"--defaults", "--fire", "--shot", "--bench", "--prof", "--from-shot", "--replay", "--map-test",
                                         "--map-click", "--edit-demo", "--edit-save", "--crater", "--throw", "--tree-throw",
                                         "--veg-regen", "--ragdoll-at"};

bool parseArgs(int argc, char** argv, LaunchOptions& o) {
    o.worldDir = WORLD_DIR;
    std::vector<Flag> table = flags(o);
    // remembered settings first, so the flags override them
    for (int k = 1; k < argc; k++)
        for (const char* nf : NO_SETTINGS_FLAGS)
            if (!strcmp(argv[k], nf)) setSettingsPersistence(false);
    loadSettings();
    loadRadius = std::clamp((int)ceilf(S.viewDist / CELL_METRES), 1, MAX_LOAD_RADIUS);  // streamed cells follow the view distance
    for (int k = 1; k < argc; k++) {
        if (!strcmp(argv[k], "--help") || !strcmp(argv[k], "-h")) {
            printf("usage: ptx [flags]\n");
            for (const Flag& fl : table) printf("  %-20s %s\n", fl.name, fl.help);
            return false;
        }
        const Flag* hit = nullptr;
        for (const Flag& fl : table)
            if (!strcmp(argv[k], fl.name)) { hit = &fl; break; }
        if (!hit) { fprintf(stderr, "unknown flag %s (--help lists them)\n", argv[k]); continue; }
        if (k + hit->args >= argc) { fprintf(stderr, "%s needs %d value(s)\n", hit->name, hit->args); break; }
        hit->apply(argv + k + 1);
        k += hit->args;
    }
    markFlagsApplied();
    return true;
}

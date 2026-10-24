#include "tools/movelog.h"
#include <vector>
#include <string>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <sstream>
#include <cstdint>
#include <ctime>
#include <cstddef>
#include "world/terrain.h"
#include "app/frame_state.h"
#include "app/settings.h"
#include "gpu/context.h"
#include "render/targets.h"
#include "scene/camera.h"

FILE* moveLog = nullptr;

static std::string moveLogPath;

static uint32_t moveLogFrames = 0;

std::vector<MoveFrame> replayFrames;

std::vector<MoveEvent> replayEvents;

size_t replayAt = 0, replayEvAt = 0;

bool replaying = false, replayShots = false, replayExit = false;

static void writeSceneHeader(FILE* f);

void moveLogEvent(const char* kind, const std::string& arg) {
    if (moveLog) fprintf(moveLog, "e %u %s %s\n", moveLogFrames, kind, arg.c_str());
}

void openMoveLog() {
    std::error_code ec;
    std::filesystem::create_directories("logs", ec);
    char name[96];
    time_t t = time(nullptr);
    strftime(name, sizeof name, "logs/ptx_%Y%m%d_%H%M%S.movelog", localtime(&t));
    moveLogPath = std::filesystem::absolute(name).string();
    moveLog = fopen(name, "w");
    if (!moveLog) return;
    fprintf(moveLog, "# ptx movement log (replay: ./build/ptx --replay %s)\n", moveLogPath.c_str());
    writeSceneHeader(moveLog);
    fprintf(moveLog, "# f <frame> <time> <dt> <x y z> <yaw pitch deg> <walk>;  e <frame> <event> <args>\n");
}

void moveLogFrame(float dt) {
    if (!moveLog || replaying) return;
    fprintf(moveLog, "f %u %.4f %.5f %.4f %.4f %.4f %.3f %.3f %d\n", moveLogFrames, simTime, dt, cam.pos.x, cam.pos.y, cam.pos.z,
            cam.yaw * 180.f / 3.14159265f, cam.pitch * 180.f / 3.14159265f, walkMode ? 1 : 0);
    moveLogFrames++;
    if (moveLogFrames % 300 == 0) fflush(moveLog);  // a crash loses at most a few seconds
}

bool loadReplay(const std::string& path) {
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "--replay: can't read %s\n", path.c_str()); return false; }
    std::string line;
    while (std::getline(f, line)) {
        std::istringstream ls(line);
        std::string k;
        ls >> k;
        if (k == "f") {
            MoveFrame m;
            ls >> m.f >> m.t >> m.dt >> m.x >> m.y >> m.z >> m.yaw >> m.pitch >> m.walk;
            replayFrames.push_back(m);
        } else if (k == "e") {
            MoveEvent e;
            ls >> e.f >> e.kind;
            std::getline(ls >> std::ws, e.arg);
            std::istringstream as(e.arg);
            as >> e.a >> e.b;
            replayEvents.push_back(e);
        }
    }
    // the starting scene (settings, river, craters) is in the header: same format as a screenshot's .txt
    extern bool loadShotInfoPath(const std::string&);
    loadShotInfoPath(path);
    replaying = !replayFrames.empty();
    printf("replay: %zu frames, %zu events from %s\n", replayFrames.size(), replayEvents.size(), path.c_str());
    return replaying;
}

// The scene beside each screenshot (<shot>.txt): camera, time, river, craters and every setting, so
// ./build/ptx --from-shot <shot.png|txt> reopens exactly this view.
void writeShotInfo(const std::string& png) {
    std::string txt = png.substr(0, png.size() - 4) + ".txt";
    FILE* f = fopen(txt.c_str(), "w");
    if (!f) return;
    fprintf(f, "# ptx scene for %s\n# reopen: ./build/ptx --from-shot %s\n", png.c_str(), txt.c_str());
    if (!moveLogPath.empty())
        fprintf(f, "# movement log: %s at frame %u (replay up to here: ./build/ptx --replay %s)\nmovelog %s %u\n", moveLogPath.c_str(),
                moveLogFrames, moveLogPath.c_str(), moveLogPath.c_str(), moveLogFrames);
    writeSceneHeader(f);
    fclose(f);
    moveLogEvent("shot", png);
}

static void writeSceneHeader(FILE* f) {
    float yawD = cam.yaw * 180.f / 3.14159265f, pitchD = cam.pitch * 180.f / 3.14159265f;
    fprintf(f, "# GPU: %s, render %ux%u -> output %ux%u, %.1f ms GPU, %.1f ms CPU frame\n", gpuName, W, H, OW, OH, gpuEmaTotal, cpuFrameMs);
    fprintf(f, "pos %.4f %.4f %.4f\nlook %.4f %.4f\nfov %.3f\ntime %.4f\nwalk %d\n", cam.pos.x, cam.pos.y, cam.pos.z, yawD, pitchD,
            cam.fovDeg, simTime, walkMode ? 1 : 0);
    fprintf(f, "river %g %g %g %g %g\n", gRiver.width, gRiver.depth, gRiver.level, gRiver.slope, gRiver.bumps);
    for (int i = 0; i < std::min(gCraterCount.load(), MAX_CRATERS); i++)
        fprintf(f, "crater %g %g %g %g %g %g %g\n", gCraters[i].x, gCraters[i].z, gCraters[i].r, gCraters[i].depth, gCraters[i].base, gCraters[i].gx, gCraters[i].gz);
    fprintf(f, "settings %zu ", sizeof(Settings));
    const uint8_t* b = (const uint8_t*)&S;
    for (size_t i = 0; i < sizeof(Settings); i++) fprintf(f, "%02x", b[i]);
    fprintf(f, "\n# readable: sun %.1f / %.1f deg, precipitation %.2f, fog %g, clouds %.2f, view %.0f m, trees %.2f, ground mode %d\n",
            S.sunElev, S.sunAzim, S.rain, S.fogDensity, S.cloudCoverage, S.viewDist, S.treeDensity, S.groundMode);
    fprintf(f, "# command: ./build/ptx --pos %.3f %.3f %.3f --look %.2f %.2f\n", cam.pos.x, cam.pos.y, cam.pos.z, yawD, pitchD);
}

// --from-shot: apply a saved scene (before the world starts)
bool loadShotInfoPath(const std::string& path);

bool loadShotInfo(std::string path) { return loadShotInfoPath(path); }

bool loadShotInfoPath(const std::string& p0) {
    std::string path = p0;
    if (path.size() > 4 && path.compare(path.size() - 4, 4, ".png") == 0) path = path.substr(0, path.size() - 4) + ".txt";
    std::ifstream f(path);
    if (!f) { fprintf(stderr, "--from-shot: can't read %s\n", path.c_str()); return false; }
    std::string line;
    int nc = 0;
    while (std::getline(f, line)) {
        std::istringstream ls(line);
        std::string k;
        ls >> k;
        if (k == "pos") ls >> cam.pos.x >> cam.pos.y >> cam.pos.z;
        else if (k == "look") { float y, p; ls >> y >> p; cam.yaw = y * 3.14159265f / 180.f; cam.pitch = p * 3.14159265f / 180.f; }
        else if (k == "fov") ls >> cam.fovDeg;
        else if (k == "time") ls >> simTime;
        else if (k == "walk") { int w; ls >> w; walkMode = w != 0; }
        else if (k == "river") ls >> gRiver.width >> gRiver.depth >> gRiver.level >> gRiver.slope >> gRiver.bumps;
        else if (k == "crater" && nc < MAX_CRATERS) {
            Crater c;
            ls >> c.x >> c.z >> c.r >> c.depth;
            if (!(ls >> c.base >> c.gx >> c.gz)) { c.base = NAN; c.gx = c.gz = 0.f; }  // older files: settled at startup
            gCraters[nc++] = c;
        }
        else if (k == "settings") {
            size_t n;
            std::string hex;
            ls >> n >> hex;
            if (n != sizeof(Settings) || hex.size() != 2 * n) { fprintf(stderr, "--from-shot: settings from another build, camera only\n"); continue; }
            uint8_t* b = (uint8_t*)&S;
            for (size_t i = 0; i < n; i++) b[i] = (uint8_t)strtoul(hex.substr(2 * i, 2).c_str(), nullptr, 16);
        }
    }
    gCraterCount.store(nc);
    printf("scene from %s\n", path.c_str());
    return true;
}

void closeMoveLog() {
    if (moveLog) fclose(moveLog);
    moveLog = nullptr;
}

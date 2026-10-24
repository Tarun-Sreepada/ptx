#pragma once
#include <vector>
#include <string>
#include <cstdio>
#include <cstdint>
#include <cstddef>

// ---------------------------------------------------------------- movement log + replay
// Every session records logs/ptx_<start>.movelog: the starting scene (settings, river, craters), then one line per
// frame (frame, time, step, camera position / yaw / pitch, walking) and events (rocks thrown, screenshots). A
// screenshot's .txt names the log and frame it was taken at. --replay <log> drives the camera from it frame by frame
// with the recorded steps and re-throws the rocks; --replay-shots retakes its screenshots.
struct MoveFrame { uint32_t f; double t; float dt, x, y, z, yaw, pitch; int walk; };

struct MoveEvent { uint32_t f; std::string kind, arg; float a = 0, b = 0; };

extern FILE* moveLog;

extern std::vector<MoveFrame> replayFrames;

extern std::vector<MoveEvent> replayEvents;

extern size_t replayAt, replayEvAt;

extern bool replaying, replayShots, replayExit;

void moveLogEvent(const char* kind, const std::string& arg);

void openMoveLog();
void closeMoveLog();

void moveLogFrame(float dt);

bool loadReplay(const std::string& path);

void writeShotInfo(const std::string& png);

bool loadShotInfo(std::string path);

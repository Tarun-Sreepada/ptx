#pragma once
#include <vector>
#include "app/settings.h"
#include "scene/camera.h"

struct Bench {
    static constexpr int WARM = 60, MEAS = 240;
    bool running = false, done = false;
    int tier = 0, frame = 0;
    double vt = 0;
    std::vector<float> samples[5];
    float avg[5] = {}, p95[5] = {};
    Settings saved;
    Camera savedCam;
};

extern Bench bench;

extern int targetFps;

int raysPerPixel(int n, int b);

void readTimestamps();

void startBenchmark();

void benchStep();

int benchBestTier(int fps);

void autoScale();

// --bench: results table on stdout
void printBenchmark();

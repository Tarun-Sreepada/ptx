// One frame, in order: wait for the GPU, read back last frame's results, the upload window (streaming, edits,
// destruction), the UI, simulation (camera, physics, water, weather), frame data + culling, then the render graph,
// present and the per-frame bookkeeping (screenshots, benchmark, frame limiter).
#pragma once
#include "app/cli.h"

// Returns false when the program should exit.
bool runFrame(const LaunchOptions& opts);

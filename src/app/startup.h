// Startup: everything from an empty process to the first frame. Independent work runs side by side on the job pool
// (all cores but one): asset and world loading, procedural vegetation / rocks, audio, physics and the pipeline
// compiles, while the main thread brings up the window, the device and the GPU resources that need them.
#pragma once
#include "app/cli.h"

void startup(const LaunchOptions& opts);

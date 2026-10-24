#include "app/frame_state.h"
#include <cstdint>

float frameDt = 1.f / 60.f;

uint32_t frameIndex = 0;

double simTime = 0.0;

bool resetHistory = true;

bool resetRR = true;

// ---------------------------------------------------------------- timing / benchmark
float gpuTotal = 0, gpuEmaTotal = 0, gpuEmaGI = 0;

float cpuFrameMs = 16.f;

bool haveTimestamps = false;

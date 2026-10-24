#pragma once
#include <cstdint>

extern float frameDt;

extern uint32_t frameIndex;

extern double simTime;

extern bool resetHistory;

extern bool resetRR;

extern float gpuTotal, gpuEmaTotal, gpuEmaGI;

extern float cpuFrameMs;

extern bool haveTimestamps;

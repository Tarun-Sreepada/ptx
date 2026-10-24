#pragma once
#include <cstdint>

extern float jitter[2], prevJitter[2];

uint32_t flags();

void updateFrameData();

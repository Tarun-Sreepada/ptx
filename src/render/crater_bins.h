#pragma once
#include "gpu/vk.h"

// Crater bins (material.glsl craterRange): 32 x 32 cells of 16 m around the camera, each listing the craters whose
// reach touches it, so terrain / shadow lookups loop over a few craters instead of all of them.
inline constexpr int CB_N = 32, CB_BINS = CB_N * CB_N, CB_LIST = 16384;

extern Buffer craterBinBuf;

extern int cbOrigin[2];

extern bool cbValid;

void buildCraterBins(float cx, float cz);

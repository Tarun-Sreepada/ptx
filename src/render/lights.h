#pragma once
#include <vector>
#include <cstdint>
#include "core/math.h"
#include "gpu/vk.h"

struct LightBase { V3 p; float phase; V3 col; float inten; };

extern std::vector<LightBase> lightBase;

extern float orbFade[64];

extern uint32_t numLights, numWorldLights;

extern Buffer triLightBuf, lightTreeBuf, blueNoiseBuf, lightAliasBuf;

extern uint32_t numTriLights, numTreeNodes;

extern bool allEmittersListed;

extern float lightScale;

extern std::vector<float> wlx, wly, wlz, wlLum, wlScore;

void rebuildWorldLights();

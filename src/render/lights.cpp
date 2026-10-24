#include "render/lights.h"
#include <vector>
#include <cstdint>
#include <cstddef>
#include "world/world.h"
#include "gpu/vk.h"

std::vector<LightBase> lightBase;

float orbFade[64];  // per orb, 0 at the wrap edge of the tile (MAX_ORBS)

uint32_t numLights = 0, numWorldLights = 0;

Buffer triLightBuf, lightTreeBuf, blueNoiseBuf, lightAliasBuf;  // lights.glsl

uint32_t numTriLights = 0, numTreeNodes = 0;

bool allEmittersListed = true;

float lightScale = 4.66e-4f;  // glTF candela -> renderer intensity: 1 / (683 pi), so Blender watts match the sun scale

// Light SoA of every streamed light; each frame the strongest (by luminance / distance^2) fill the slots after the orbs.
std::vector<float> wlx, wly, wlz, wlLum, wlScore;

void rebuildWorldLights() {
    auto& L = stream::lights();
    wlx.resize(L.size()); wly.resize(L.size()); wlz.resize(L.size()); wlLum.resize(L.size()); wlScore.resize(L.size());
    for (size_t i = 0; i < L.size(); i++) {
        wlx[i] = L[i].pos[0]; wly[i] = L[i].pos[1]; wlz[i] = L[i].pos[2];
        wlLum[i] = 0.2126f * L[i].color[0] + 0.7152f * L[i].color[1] + 0.0722f * L[i].color[2];
    }
}

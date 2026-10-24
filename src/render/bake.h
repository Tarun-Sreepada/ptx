#pragma once
#include <vector>
#include <string>
#include <cstdint>
#include "world/world.h"
#include "gpu/vk.h"

// baked interiors (probes.glsl): a probe grid per bake volume, baked progressively on the GPU for both switch
// states and cached in <world>/bake/
inline constexpr uint32_t MAX_VOLUMES = 64, MAX_PROBES = 400000, BAKE_PASSES = 16, BAKE_CHUNK = 8192, BAKE_FILTERS = 2;

struct BakeVol {
    ProbeVolume v;
    float spacing = 0.25f;
    uint32_t dims[3] = {}, first = 0, count = 0;
    bool on = true;
    bool ready[2] = {}, saved[2] = {};
    uint32_t pass[2] = {}, next[2] = {};
};

extern std::vector<BakeVol> bakeVols;

extern Buffer volBuf, probeBuf, probeTmp;

extern bool bakedGI, bakeEnabled, startLightsOff;

extern uint64_t bakeEnv;

extern std::string bakeDir;

extern uint32_t bakeOffLights;

int volumeAt(float x, float y, float z);

uint64_t bakeEnvHash();

void rebuildBakeVolumes(bool envChanged);

bool lightOff(const float p[3]);

void writeBakeVolumes();

// Next chunk of bake work: the volume the camera is in first (its current state first), then the rest.
struct BakeJob { int vol = -1, st = 0; uint32_t first = 0, n = 0, pass = 0; bool filter = false; uint32_t count = 0, base = 0; };

BakeJob nextBakeJob();

void saveFinishedBakes();

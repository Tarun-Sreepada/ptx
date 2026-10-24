#include "scene/grass.h"
#include <cstdint>

uint32_t maxBlades = 400000;  // buffer / BLAS capacity

uint32_t bladeCap = 0;        // slots built this frame (from last frame's count + headroom)

uint32_t lastBladeCount = 0, grassGridN = 0;

float terrainCenter[2] = {1e9f, 1e9f};  // finest clipmap level's centre (units of twice its spacing)
float terrainBuildPos[2] = {0.f, 0.f};  // camera xz the current terrain mesh was built around
uint32_t terrainLevels = 1;              // clipmap levels in use (view distance)

bool terrainDirty = true;

bool grassRefit = false, grassBuilt = false;

int grassFrames = 0;

uint32_t grassBuildCap = 0;

float grassBuildPos[3], grassBuildFwd[3];

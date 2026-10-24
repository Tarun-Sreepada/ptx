#include "sim/world_sim.h"
#include "scene/grass.h"
#include "app/frame_state.h"
#include "app/settings.h"
#include "gpu/vk.h"

// world map (worldmap.h): edits uploaded from the CPU, and the GPU-owned simulation grid (worldsim.comp)
Image mapHImg, mapPaintImg, mapWaterImg;  // height delta (r32f), paint (rgba8), drawn water (rgba32f, half res)

Image simImg, fluxWImg, fluxSImg, simGroundImg;

float gSimBrush[4] = {0, 0, 0, 0};  // editor snow brush this frame (x, z, radius, m/s)

bool simClear = false;

int simActiveFrames = 0;

Buffer simRB;

int simRBx = 0, simRBz = 0, simSnapX = 0, simSnapZ = 0;

float simSnap[SIM_RB * SIM_RB][2];  // snow depth, wetness

bool simSnapValid = false;

void keepSnowTerrainFresh() {
    if (S.worldSim && S.snowMeshEvery > 0 && frameIndex % (uint32_t)S.snowMeshEvery == 0 && (S.rain > 0.f || gSimBrush[2] > 0.f || simActiveFrames > 0)) {
        terrainDirty = true;
        if (simActiveFrames > 0) simActiveFrames--;
    }
}

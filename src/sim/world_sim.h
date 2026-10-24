#pragma once
#include "gpu/vk.h"

extern Image mapHImg, mapPaintImg, mapWaterImg;

extern Image simImg, fluxWImg, fluxSImg, simGroundImg;

extern float gSimBrush[4];

extern bool simClear;

extern int simActiveFrames;

// a 64 x 64 cell (128 m) patch of the simulation around the camera, read back each frame: the physics' view of
// snow and wetness (ground friction / bounce)
inline constexpr int SIM_RB = 64;

extern Buffer simRB;

extern int simRBx, simRBz, simSnapX, simSnapZ;

extern float simSnap[SIM_RB * SIM_RB][2];

extern bool simSnapValid;

// Snow changes the ground's shape: while it falls (or is painted) the terrain mesh is refreshed every few frames.
void keepSnowTerrainFresh();

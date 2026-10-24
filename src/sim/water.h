#pragma once
#include "world/terrain.h"

extern float riverSwell;

struct CraterWater { bool pond = false; float level = 0, fill = 1, divert = 0, inflowSpeed = 0, lastR = 0, ground[3] = {}; };  // updateCraterWater

extern CraterWater craterWater[MAX_CRATERS];

void updateRiverSwell(float dt);

float waterLevelAt(float x, float z);

void updateCraterWater(float dt);

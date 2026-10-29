#pragma once


void groundSurfaceAt(const float p[3], float& f, float& e);

// What a foot lands on, for the footstep sounds: feetAbove = how far the feet are above the terrain (standing on a
// building, a primitive, a rock: hard), wade = water depth at the feet.
enum FootSurface { FOOT_GRASS, FOOT_DIRT, FOOT_CONCRETE, FOOT_WOOD, FOOT_SNOW, FOOT_WATER, FOOT_COUNT };
inline const char* FOOT_NAMES[FOOT_COUNT] = {"grass", "dirt", "concrete", "wood", "snow", "water"};
int footSurfaceAt(const float p[3], float feetAbove, float wade);

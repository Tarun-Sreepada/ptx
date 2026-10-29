#pragma once


float heightFn(float x, float z);

#include <cstdint>
// World seed: the base hash mix and the river's meander (seed 0 = the original world). Callers then rebuild what
// depends on it (terrain, placement, vegetation): see the World panel.
void setWorldSeed(uint32_t seed);
// S's world generation values (seed, hilliness, river variation) -> terrain.h's globals. Startup and the World panel.
void applyWorldGen();

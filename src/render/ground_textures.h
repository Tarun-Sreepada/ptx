// Tiled ground texture sets (texgen.cpp, cached in texcache/): albedo + detail for bare ground and soil. Only the
// tiled and hybrid ground modes (0, 2) sample them, so they load in the background the first time one of those is
// active and upload in the following upload window; the layered virtual texture (mode 3, default) never needs them.
#pragma once
#include <cstdint>

extern uint32_t groundSlots[4];  // bindless slots: bare albedo, bare detail, soil albedo, soil detail
extern float groundMaxH[2];      // metres of relief at height 1, per set
extern bool groundReady;
extern uint32_t groundTexSize;   // texels per side (--ground-tex)

// Upload window, every frame: starts / finishes the background load when the ground mode wants the textures.
void updateGroundTextures();

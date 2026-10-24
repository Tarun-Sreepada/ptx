// Parts of the frame uniform buffer (frame_ubo.h), filled every frame by updateFrameData in this order.
#pragma once
#include "render/frame_ubo.h"

void fillCameraSky(UBO& u);      // camera (this + last frame), sun / moon / night sky, wind, clouds, render settings
void fillSurfaceParams(UBO& u);  // grass blade shape, ground detail, subsurface, ground textures / VT, wind map window
void fillGrassParams(UBO& u);    // grass grid + view cone, BLAS refit or rebuild (scene/grass.cpp)
void fillWorldState(UBO& u);     // terrain grid, DLSS jitter, river / water, thrown rocks, splashes, craters
void fillWeather(UBO& u);        // fog, wetness (eased with the rain), lightning, rainbow, fog banks, world sim, decals, dust
void fillPost(UBO& u);           // exposure, lens, tonemap, display

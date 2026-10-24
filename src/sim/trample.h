// Feet on the ground. Grass bends away from whatever stands in it (the player, characters, rocks lying in the
// grass) and stays flattened along a trail of recent footsteps, rising again over a minute or so (grass.comp
// pushers). Every step on open ground leaves a footprint, pressed into soft soil and snow (surface_fx.comp decal
// kind 3), fading after a few minutes.
#pragma once
#include "core/math.h"
#include "render/frame_ubo.h"

// A footstep at p (on the ground) heading yaw (radians, 0 = +z), the left or right foot.
void footstep(V3 p, float yaw, bool left);
void writeTrample(UBO& u);

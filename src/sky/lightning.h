#pragma once
#include <vector>
#include <array>
#include "core/math.h"
#include "render/frame_ubo.h"

// ---------------------------------------------------------------- lightning (lightning.glsl)
// Strikes in thunderstorms: Poisson in time (rain under cumulonimbus), at 150 m .. 6 km. The channel is a jagged
// polyline from the cloud base to the ground with a few side branches; 3-4 return strokes flicker over ~0.5 s.
// Thunder is played after distance / 343 m/s.
struct Bolt { std::vector<std::array<float, 8>> seg; V3 ground, top; double t0 = -1e9; float strokes[4]; int nStrokes = 0; bool thunder = false; float dist = 0; };

extern Bolt bolt;

extern bool boltRequest;

void updateLightning(float dt);

void writeLightning(UBO& u);

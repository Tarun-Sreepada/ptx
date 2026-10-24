// Small animals around the camera, drawn by the raster pass only (too small to matter to the ray tracer): flocks of
// birds wheeling high over the land by day, flapping and gliding, scattering at loud bangs; butterflies fluttering
// over the flowers on warm days; fireflies blinking over the meadows on warm, dry nights.
#pragma once
#include <vector>

#include "core/math.h"

struct CritterDraw {
    int mesh;                       // trees::PLANT_FIREFLY ...
    float m[3][4], prev[3][4];      // transform now and last frame
    float flapHz = 0.f, flapPhase = 0.f, flapAmp = 0.f;  // wings (raster_model.vert): beats per second, 0..1, radians
};

void updateCritters(float dt);
// Something loud at at (loudness 0..1+): birds nearby take off and wheel away higher.
void critterAlarm(V3 at, float loudness);
const std::vector<CritterDraw>& critterDraws();

#pragma once


void cullTrees();

// how much a plant variant gives to the wind (-1: rigid clutter) and a per-position sway phase (sway.glsl)
int plantFlex(int v);
float swayPhase(float x, float z);

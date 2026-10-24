// Fires: a burning patch of ground or wood with flames, smoke, embers, a flickering light and its crackle. Started by
// lightning striking near trees, by very energetic impacts, by embers landing on dry ground, or by hand (the
// Destruction panel, --fire). A fire grows to full heat, burns its fuel and dies down; rain puts it out sooner.
// The GPU draws flames and smoke as volumes (shaders/scene/fire.glsl); the light joins the light list, hidden from
// the camera (the flames are what you see).
#pragma once
#include <cstdint>
#include <vector>

#include "core/math.h"
#include "render/frame_ubo.h"

struct Fire {
    V3 p;                         // base, on the ground
    float radius;                 // m
    float fuel;                   // s of burning left at full heat
    float heat = 0.f;             // 0..1: grows from a spark, dies with the fuel
    float seed = 0.f;
    float emberDue = 0.f;         // embers owed to the air (fractional)
    uint32_t sound = UINT32_MAX;  // crackle emitter
};
extern std::vector<Fire> fires;

void startFire(V3 at, float radius, float burnSeconds);
// A fire on the ground at x, z, lit once the world is up (command line).
void requestFire(float x, float z, float radius, float burnSeconds);
void putOutFires();
// A fire on the ground the camera looks at (within reach m); false if the view doesn't meet the ground.
bool fireInView(float radius, float burnSeconds, float reach = 200.f);
void updateFires(float dt);
void writeFires(UBO& u);
// Flickering point lights (8 floats each, hidden from the camera) at L; returns how many (at most room).
int writeFireLights(float* L, int room);
// A rising ember from a fire, or one landing: may light dry ground.
void emberLanded(V3 at);

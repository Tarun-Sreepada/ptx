#pragma once
#include <vector>
#include <memory>
#include <cstdint>
#include <cstddef>
#include "vegetation/trees.h"
#include "physics/fracture.h"
#include "world/world.h"
#include "core/math.h"
#include "sim/hydro.h"

// thrown rocks: dynamic physics bodies sharing the rock mesh (hull for collision), pushed around by the river
// Fragments of broken rocks live in a FragSet: one uploaded model per fracture event, one mesh per piece.
struct FragSet { Model model; int live = 0; };

extern std::vector<std::unique_ptr<FragSet>> fragSets;

struct Thrown {
    uint32_t body; float sc; Xf xf; bool wet = false; float v[3] = {}; float cooldown = 0.f, trail = 0.f;
    float w[3] = {}, w0[3] = {};                 // angular velocity this frame / the one before (wood landings)
    float radius = 0.f;                          // bounding radius (m)
    FragSet* frag = nullptr; uint32_t fragMesh = 0;  // a fragment: own mesh, own hull; else the rock mesh at scale sc
    uint32_t rockVar = 0;                        // which generated rock mesh (not a fragment)
    void* shape = nullptr;
    std::shared_ptr<fracture::Poly> poly;        // the solid in body space (unscaled), what breaks next time
    float age = 0.f;
    float density = 2600.f;                      // kg/m^3 of its hull (buoyancy)
    bool wood = false;                           // a piece of a broken tree: no craters, floats
    Xf prevXf{}; bool hasPrev = false;           // last frame's pose (motion vectors)
    float rest = 0.f;                            // seconds at rest (retire settled debris)
    std::shared_ptr<trees::Wood> woodPart;       // a loose part of a procedural tree: breaks again on landing
    float woodScale = 1.f; V3 woodCtr{};         // its frame: body space = woodScale * tree local - woodCtr
    uint32_t woodSeed = 0;                       // its tree's (splinters stay the same on every rebuild)
    std::shared_ptr<const HydroShape> hydro;     // its surface elements for buoyancy / drag (sim/hydro.h), built on first contact
};

extern uint32_t numThrownRock;

extern uint32_t fragsDrawn;

extern std::vector<Thrown> thrown;

extern uint32_t numThrown;

extern int autoThrow;

extern bool treeLog;

extern int treeThrow;

void releaseFrag(FragSet* fs);

void dropThrown(size_t i);

void updateThrown(float dt);

void writeThrown();

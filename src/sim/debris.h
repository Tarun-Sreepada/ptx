#pragma once
#include <vector>
#include <cstdint>
#include "vegetation/trees.h"
#include "core/math.h"

// debris: ballistic particles (CPU), drawn and ray traced as instances of the plant model's debris meshes
enum DebrisKind : uint8_t { DEB_WATER, DEB_DIRT, DEB_WOOD, DEB_LEAF, DEB_EMBER, DEB_KINDS };

inline const int debrisMesh[DEB_KINDS] = {trees::PLANT_SPRAY, trees::PLANT_CLOD, trees::PLANT_SPLINTER, trees::PLANT_LEAF, trees::PLANT_EMBER};

struct Droplet { V3 p, v; float size, life, delay; uint8_t kind = DEB_WATER; V3 spin{0, 1, 0}; V3 prevP{0, 0, 0}; bool hasPrev = false; };

extern std::vector<Droplet> spray;

extern uint32_t numSprayInst;

void dropletXf(const Droplet& d, float m[3][4]);

void addDebris(uint8_t kind, V3 p, V3 v, float size, float delay = 0.f);

void updateSpray(float dt);

#pragma once
#include <unordered_map>
#include <unordered_set>
#include <cmath>
#include <cstdint>
#include "world/world.h"
#include "core/math.h"

// Scattered ground rocks get static colliders near thrown rocks (like tree trunks); a hard enough hit knocks one
// loose: it leaves the placement (brokenRocks) and becomes a thrown rock that fractures by the same energy rule.
struct RockBody { uint32_t body; Xf xf; float sc; uint32_t var; };

extern std::unordered_map<uint64_t, RockBody> rockBodies;

extern std::unordered_set<uint64_t> brokenRocks;

extern uint32_t numBreakableRocks;

static inline uint64_t rockKey(float x, float z) { return ((uint64_t)(uint32_t)(int)lroundf(x * 8.f) << 32) | (uint32_t)(int)lroundf(z * 8.f); }

bool knockRockLoose(V3 p, float reach, V3 J, float E);

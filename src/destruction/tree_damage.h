#pragma once
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <cstddef>
#include <utility>
#include "vegetation/trees.h"
#include "world/world.h"
#include "destruction/thrown.h"
#include "scene/vegetation.h"

// broken trees: the pieces above the snap are thrown (wood = true); the stump stays, a static body in the
// placed tree's grid cell
struct Stump { FragSet* frag; uint32_t mesh, body; void* shape; Xf xf; uint64_t key = 0; };  // key: damaged procedural tree

extern std::vector<Stump> stumps;

extern std::unordered_set<uint64_t> brokenTrees;

extern std::unordered_map<uint64_t, std::pair<TreeInst, trees::Wood>> damagedTrees;

extern uint32_t numFallen;

static inline uint64_t treeKey(int ix, int iz) { return ((uint64_t)(uint32_t)ix << 32) | (uint32_t)iz; }

void dropStump(size_t i);

#include "destruction/tree_damage.h"
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <cstdint>
#include <cstddef>
#include <utility>
#include "physics/physics.h"
#include "vegetation/trees.h"
#include "destruction/thrown.h"
#include "scene/vegetation.h"

std::vector<Stump> stumps;

std::unordered_set<uint64_t> brokenTrees;  // grid cells of broken trees (placement skips them)

// procedural trees hit but standing (what's left of them, a Stump with this key): processWood
std::unordered_map<uint64_t, std::pair<TreeInst, trees::Wood>> damagedTrees;

uint32_t numFallen = 0;

void dropStump(size_t i) {
    Stump& s = stumps[i];
    if (s.key) damagedTrees.erase(s.key);
    physics::removeBodies(&s.body, 1);
    physics::releaseShape(s.shape);
    releaseFrag(s.frag);
    stumps.erase(stumps.begin() + i);
}

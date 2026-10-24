// Tree growth: what a tree is made of before it becomes a mesh. A Plan holds the trunk, every limb as a chain of
// tube segments (each segment knowing the one it grows from, for the wood mechanics) and the foliage clusters.
// Generator (Weber-Penn style): a straight bole, then a leader that bends as it climbs, branches spiralling off it
// (phyllotaxis) whose length follows the crown envelope, sub-branches recursively down to twigs that carry the
// leaf clusters. Limbs bend toward the light (upturn), sag under their weight (gravity) and wander (gnarl).
#pragma once
#include <vector>

#include "vegetation/trees.h"
#include "vegetation/trees/tree_math.h"

namespace trees {
// Tapered tube a..b. parent: the limb segment it grows from (-1: the trunk); chain: the limb it belongs to.
struct Limb { V a, b; float ra, rb; int sides, parent, level = 0, chain = -1; };
// One limb grown in one go: its segments base to tip, swept as one surface (limb_mesh.h).
struct Chain {
    std::vector<int> segs;
    int sides = 6;
    float parentR = 0.f;    // radius of the limb it grows out of (its collar fills the joint); 0 = none
    V parentDir{0, 1, 0};   // that limb's direction there
};
enum { F_LEAVES, F_FRUIT };
// A cluster at c on its twig (the twig going dir): n leaves on a few shoots about spread long, or n fruit.
struct Foliage { int kind; V c, dir; float spread, size; int n; float seed; int owner; };
struct Plan {
    Limb trunk;
    std::vector<Limb> branches;
    std::vector<Chain> chains;   // trees: chain 0 is the leader, continuing the trunk
    std::vector<Foliage> leaves;
    uint32_t leafMat = 0, woodMat = 0, fruitMat = 0;
    float leafWidth = 0.5f, droop = 0.3f, flare = 0.f, wave = 0.f, seed = 0.f, upturn = 0.f;
    bool needles = false;        // conifer: whorls of needles on the shoots instead of leaves
    bool trunkPath = true;       // chain 0 continues the trunk (trees); bushes grow stems out of the ground
};

Plan growTree(const TreeParams& P, uint32_t seed, uint32_t leafMat, uint32_t woodMat);
// Space colonization (colonize.cpp): branches grow toward the free space in the crown envelope.
Plan colonizeTree(const TreeParams& P, uint32_t seed, uint32_t leafMat, uint32_t woodMat);
// Over the per-tree triangle budget: thin every leaf cluster evenly (bigger leaves keep the crown full).
void fitFoliageBudget(Plan& p);
// Bush: stems out of the ground (no bole), the same branching, fruit hanging under the twig ends.
Plan growBush(const BushParams& B, uint32_t seed, uint32_t leafMat, uint32_t woodMat, uint32_t fruitMat);
}  // namespace trees

#pragma once
#include <vector>
#include <unordered_map>
#include <cstdint>
#include "vegetation/trees.h"
#include "world/world.h"
#include "core/math.h"
#include "gpu/vk.h"

// procedural trees (trees.h): one non-streamed model, mesh v = variant v
struct TreeInst { V3 p; float yaw, scale; uint32_t v; int ix, iz; };

extern Model treeModel;

// World forest: imported assets join the procedural variants (trees::VARIANTS..numTreeVariants-1), so placement,
// the BVH, impostors, trunk colliders and breaking treat them alike. <world>/forest.txt, one "<weight> <asset>" per
// line, plus optional "builtin <weight>" (procedural trees' share, default 0 once assets are listed) and
// "spacing <x>" (bounding footprints x this may not overlap). Read at startup.
inline constexpr int MAX_TREE_VARIANTS = trees::VARIANTS + 12;

extern int numTreeVariants;

extern std::vector<ModelData> forestAssets;

extern float treeWeight[MAX_TREE_VARIANTS], treeWeightSum, forestSpacing;

extern float treeFoot[MAX_TREE_VARIANTS];

extern float treeC[MAX_TREE_VARIANTS][3], treeR[MAX_TREE_VARIANTS];
extern float treeTop[MAX_TREE_VARIANTS];  // height of each variant's mesh (local m): how far up it sways

extern std::vector<TreeInst> treeList;

extern uint32_t numTrees, numTreeNear, numImpostors;

extern bool placeDirty;

extern Buffer impBuf;

extern Image impAlbedoImg, impNormalImg;

extern void* trunkShape[MAX_TREE_VARIANTS];

extern std::unordered_map<uint64_t, uint32_t> treeBodies;

// small plants (trees::buildPlants): same idea without impostors or colliders, placed within PLANT_RANGE
inline constexpr float PLANT_RANGE = 140.f;

struct PlantInst { float m[12]; uint32_t v; };

extern Model plantModel;

extern float plantC[trees::PLANT_VARIANTS][3], plantR[trees::PLANT_VARIANTS];
extern float plantTop[trees::PLANT_VARIANTS];

extern std::vector<PlantInst> plantList;

extern uint32_t numPlants, numPlantsDrawn;

// CPU: generate the procedural tree and plant meshes (job pool at startup, no GPU)
void generateVegetation();
// GPU: merge the forest assets, upload, trunk colliders, impostor bake (after generateVegetation + loadForest)
void initTrees();

extern bool vegDirty;

extern int vegRegenFrame;

void regenVegetation();

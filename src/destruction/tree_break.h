#pragma once
#include <vector>
#include "vegetation/trees.h"
#include "world/world.h"
#include "core/math.h"
#include "scene/vegetation.h"

// trees / walls broken during the physics update, carried out in the next upload window (processFractures)
struct TreeBreakReq { TreeInst t; V3 impulse, at; float severity; };  // imported trees

extern std::vector<TreeBreakReq> pendingTreeBreaks;

void knockTree(const TreeInst& t, V3 impulse, V3 at, float severity);

void takeTreeMaterials(Model& md, std::vector<trees::Piece>& pcs);

void breakTreeNow(const TreeInst& t, V3 impulse, V3 at, float severity);

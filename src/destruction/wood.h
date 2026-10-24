#pragma once
#include <vector>
#include <cstdint>
#include "vegetation/trees.h"
#include "core/math.h"
#include "destruction/thrown.h"
#include "scene/vegetation.h"

// Procedural trees break by beam mechanics (trees::woodStrike / woodLanding): whatever hits a tree loads the
// limbs from the contact down to the roots, the weakest overloaded joint snaps, leaves near the blow fall. What
// still stands is a static Stump (its Wood in damagedTrees, hit again like a whole tree); what broke off is a
// wood piece on the thrown list carrying its Wood, so it snaps again where its landing loads it hardest.
// A wood's frame: world = R (s x - ctr) + T, x tree-local (unscaled).
struct WoodFrame { float R[3][3]; V3 T; float s; V3 ctr; };

V3 woodRotT(const WoodFrame& F, V3 d);

V3 woodToWorld(const WoodFrame& F, V3 x);

WoodFrame treeFrame(const TreeInst& t);

WoodFrame thrownFrame(const Thrown& t);

uint32_t woodSeed(const TreeInst& t);

// Hit of a procedural tree (standing or a loose piece: body != UINT32_MAX), carried out in the upload window.
struct WoodReq { TreeInst t; uint32_t body; trees::WoodHit hit; V3 at, J; float vmax = 1e30f; };  // vmax: impactor speed

extern std::vector<WoodReq> pendingWood;

bool strikeTree(const TreeInst& t, V3 at, V3 vel, float mass, V3* took = nullptr);

bool strikeTrees(V3 pc, V3 vel, float mass, float reach);

void sweepThinLimbs(Thrown& t, V3 p0, V3 p1, V3 vel, float mass);

void processWood();

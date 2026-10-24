// Leaves grow on shoots. A foliage cluster (growth.h) is a few thin shoots fanning out from its twig. Broadleaf
// shoots carry leaves in a spiral, each angled out from the shoot and turned toward the light, its pointed base on
// the shoot; conifer shoots carry whorls of needles angled forward around them (a bottle brush). Fruit hangs on
// stalks under the twig.
#pragma once
#include "vegetation/trees/growth.h"
#include "vegetation/trees/mesh_builder.h"

namespace trees {
// The cluster's shoots and the first count of its leaves (fewer once some were shaken off: the shoots stay).
void emitFoliage(Builder& b, const Foliage& f, int count, const Plan& p);
// Triangles of the whole cluster, and of its shoots alone (the per-tree budget).
int foliageTris(const Plan& p, const Foliage& f);
int foliageShootTris(const Plan& p, const Foliage& f);
}  // namespace trees

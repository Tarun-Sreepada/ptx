// Impact craters in the ground (shape: world/terrain.h). An impact excavates a volume set by its energy; where it
// lands decides whether it digs a new crater, enlarges the one it fell into, or (fallback debris) does nothing.
#pragma once
#include "core/math.h"

// A body hitting the ground with energy E (J). primary: something thrown / knocked loose; false for fragments of
// what already broke (they only dig if they're fast and land outside every bowl).
void groundImpact(V3 at, float E, bool primary = true);
// Craters added before the terrain existed (command line, saved scenes) get their impact plane here, in order.
void settleCraters();

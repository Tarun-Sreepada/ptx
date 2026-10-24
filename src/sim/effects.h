#pragma once
#include <vector>
#include "core/math.h"

struct Decal { float p[3], r, n[3]; int kind; double t0; };  // kind: 0 wall impact, 1 mud splatter, 2 bark scar

struct Dust { float p[3], r, density, rise; double t0; };

extern std::vector<Decal> decalList;

extern std::vector<Dust> dustList;

void addDecal(V3 p, V3 n, float r, int kind);

void addDust(V3 p, float r, float density, float rise = 0.4f);

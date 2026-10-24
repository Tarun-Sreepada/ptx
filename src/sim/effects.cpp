#include "sim/effects.h"
#include <vector>
#include "app/frame_state.h"
#include "app/settings.h"
#include "core/limits.h"
#include "core/math.h"

std::vector<Decal> decalList;  // ring: the oldest goes when full

std::vector<Dust> dustList;

void addDecal(V3 p, V3 n, float r, int kind) {
    if (!S.decals) return;
    n = normalize(n);
    Decal d{{p.x, p.y, p.z}, r, {n.x, n.y, n.z}, kind, simTime};
    if (decalList.size() >= MAX_DECALS) decalList.erase(decalList.begin());
    decalList.push_back(d);
}

void addDust(V3 p, float r, float density, float rise) {
    if (dustList.size() >= MAX_DUST) dustList.erase(dustList.begin());
    dustList.push_back({{p.x, p.y, p.z}, r, density, rise, simTime});
}

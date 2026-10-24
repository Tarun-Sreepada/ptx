#include "world/terrain_query.h"
#include "world/terrain.h"
#include "world/world.h"

// ---------------------------------------------------------------- scene generation
// terrain height including the currently streamed flatten zones
float heightFn(float x, float z) {
    auto& zs = stream::zones();
    return terrainHeight(x, z, zs.data(), (int)zs.size());
}

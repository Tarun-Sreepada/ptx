// Placement of the procedural scene around the camera: rocks, trees, plants and ground clutter on hashed grids,
// written straight into the TLAS instance buffer and the raster transform lists. Redone only after the camera
// moved a couple of metres or the world / a setting changed (it's tens of thousands of noise + height lookups).
#pragma once
#include <functional>
#include <vector>

#include <vulkan/vulkan.h>

#include "scene/vegetation.h"
#include "world/terrain.h"

// What every placement pass shares this time round.
struct PlacementCtx {
    float RR;          // reach: the terrain's radius (m)
    float pixAngle;    // one output pixel's angle (rad): screen-size culling
    float keepPx;      // rocks are kept while they cover this many pixels
    VkAccelerationStructureInstanceKHR* inst;
    float* xf;         // rock raster transforms (rockXf)
    const std::vector<Zone>& zs;
};
// How well a plant takes at x, z (biome.h vegHabitat: tree / shrub line, slope); y: the ground height there
float habitatAt(float x, float z, bool tree, float& y);
// A cell's result: its transform and, for trees / plants, the instance record.
struct PlacedCell { float m[3][4]; TreeInst t; };
using CellFn = std::function<bool(int ix, int iz, PlacedCell&)>;
// Grid rows around the camera on the job workers, each into its own list (appended in row order afterwards).
std::vector<std::vector<PlacedCell>> placeRows(float cell, float range, const CellFn& fn);

void placeRocks(const PlacementCtx& c);
void placeTrees(const PlacementCtx& c);
void placePlants(const PlacementCtx& c);
// All of it, if the camera / world changed enough since the last time.
void placeScene();

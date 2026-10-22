// World map: hand edits over a fixed 4 km x 4 km region around the origin, on a 2 m grid (2048 x 2048 cells):
//  - height: metres added to the natural terrain (raise / lower / smooth / flatten brushes, carved rivers)
//  - paint: grass and tree density multipliers, a ground texture (palette index + weight)
//  - water: drawn rivers' surface level and flow (the water shader treats them like the main river)
// The CPU copy drives physics, placement and the editor; the GPU copy (worldmap images, bindings 90-92) is updated
// from the dirty rectangle after each edit. Saved to <world>/terrain/map.bin. The GPU-only simulation grid (snow,
// melt water, wetness) shares the layout (main.cpp, shaders/worldsim.comp).
#pragma once
#include <cstdint>
#include <string>
#include <vector>

namespace worldmap {
inline constexpr int N = 2048;              // cells per side
inline constexpr float CELL = 2.f;          // metres per cell
inline constexpr float ORIGIN = -N * CELL * 0.5f;  // world x / z of cell 0's corner
inline constexpr int PALETTE = 8;           // ground textures (index 1..7; 0 = the procedural ground)

struct Paint { uint8_t grass = 128, trees = 128, mat = 0, matW = 0; };  // 128 = 1x density
struct Water { float level = -1000.f, fx = 0.f, fz = 0.f, width = 0.f; };  // level -1000 = none

void init(const std::string& worldDir);
bool save(std::string& err);
bool dirty();  // unsaved edits

// samples (bilinear inside the region, neutral outside)
float height(float x, float z);
float grassMul(float x, float z);
float treeMul(float x, float z);
float waterLevel(float x, float z);  // -1000 = none
const std::string& paletteName(int i);  // matlib name, "" = unused
void setPalette(int i, const std::string& name);

// brushes (world position, radius m, strength per second scaled by dt)
enum Tool { RAISE, LOWER, SMOOTH, FLATTEN, NOISE, PAINT_GRASS, ERASE_GRASS, PAINT_TREES, ERASE_TREES, PAINT_TEXTURE, ERASE_TEXTURE,
            ERASE_WATER, TOOL_COUNT };
void brush(Tool t, float x, float z, float radius, float strength, float dt, float flattenTo, int mat);
void beginStroke();  // snapshot for undo (one per mouse press)
bool undo();

// rivers: a polyline with per-point half-width and depth; carved into the height and written to the water map
struct RiverPt { float x, z, width, depth; };
void addRiver(const std::vector<RiverPt>& pts);

// GPU sync: the changed cell rectangle since the last call (x0 > x1 = nothing)
struct Rect { int x0, z0, x1, z1; };
Rect takeDirty();
const float* heightData();
const Paint* paintData();
const Water* waterData();
uint32_t version();  // bumps on every edit (placement / collision refresh)
}  // namespace worldmap

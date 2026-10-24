// Engine capacities: buffer sizes, list lengths and the TLAS layout. Several are mirrored in the shaders (noted).
#pragma once
#include <cstdint>

// ---- dynamic objects
inline constexpr int MAX_SPRAY = 2048;     // debris particles: droplets, clods, splinters, leaves
inline constexpr int MAX_THROWN = 256;     // thrown rocks and their fragments (physics bodies), also the water shader's list
inline constexpr int MAX_DECALS = 128;     // impact decals (frame.glsl decals[])
inline constexpr int MAX_DUST = 32;        // dust clouds from impacts (frame.glsl dust[])
inline constexpr int MAX_FIRES = 8;        // burning fires (fire.glsl fires[])
inline constexpr int MAX_PUSHERS = 16;     // things bending the grass (grass.comp pushers[])
inline constexpr int MAX_FALLEN = 256;     // stumps / damaged trees left standing
inline constexpr int MAX_CHAR_INST = 128;  // skinned meshes of all characters (character.h)

// ---- terrain mesh: a geometry clipmap (terrain.comp), level k = N x N cells of TERRAIN_SPACING0 * 2^k
inline constexpr uint32_t TERRAIN_GRID = 128;        // cells per level side (terrain.comp N)
inline constexpr uint32_t TERRAIN_LEVELS_MAX = 10;   // 0.25 m .. 128 m vertices: reaches ~8 km
inline constexpr float TERRAIN_SPACING0 = 0.25f;     // finest vertex spacing (m)
inline constexpr float TERRAIN_MORPH_BAND = 16.f;    // cells over which a level's outer border blends into the next
inline constexpr uint32_t TERRAIN_LEVEL_VERTS = (TERRAIN_GRID + 1) * (TERRAIN_GRID + 1);
inline constexpr uint32_t TERRAIN_LEVEL_TRIS = TERRAIN_GRID * TERRAIN_GRID * 2;
// half-width of the area level k covers (m)
constexpr float terrainLevelReach(uint32_t k) { return 0.5f * TERRAIN_GRID * TERRAIN_SPACING0 * float(1u << k); }

// ---- lights
inline constexpr int MAX_LIGHTS = 256;            // orbs + nearest / strongest streamed lights, sampled by ReSTIR
inline constexpr int MAX_ORBS = 64;
inline constexpr uint32_t MAX_TRI_LIGHTS = 8192;  // emissive triangles in the light list (lights.glsl)
inline constexpr int BLUE_NOISE = 64;             // blue noise tile size (lights.glsl bn())

// ---- placed instances
inline constexpr int MAX_ROCKS = 16384;
inline constexpr int MAX_TREES = 65536;  // a 4 km view over forest
inline constexpr int MAX_PLANTS = 16384;

// ---- bindless tables (model.glsl)
inline constexpr uint32_t MAX_MESHES = 16384;
inline constexpr uint32_t MAX_MATERIALS = 65536;
inline constexpr uint32_t MAX_TEXTURES = 4096;
inline constexpr uint32_t MAX_MODEL_INST = 32768;  // TLAS / raster instances of streamed meshes
inline constexpr uint32_t MAX_ZONES = 256;

// ---- TLAS layout, in order: terrain, grass, streamed model instances, rocks, trees, plants, thrown things, debris,
// stumps, character meshes
inline constexpr uint32_t INST_MODELS = 2;  // first model instance (after terrain and grass)
inline constexpr uint32_t TLAS_CAPACITY =
    INST_MODELS + MAX_MODEL_INST + MAX_ROCKS + MAX_TREES + MAX_PLANTS + MAX_THROWN + MAX_SPRAY + MAX_FALLEN + MAX_CHAR_INST;

// ---- sky
inline constexpr uint32_t SKY_W = 256, SKY_H = 128;  // sky-view LUT (atmosphere.glsl)

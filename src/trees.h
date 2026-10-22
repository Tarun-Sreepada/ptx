// Procedural trees: a few mesh variants generated at startup, placed by hashed forest noise around the camera.
// The meshes go through the normal streamed-model path (bindless mesh + materials + BLAS), so the ray tracer
// and the raster G-buffer shade them like any glTF model. Far away the raster pass swaps them for octahedral
// impostors (impostor.glsl); the BVH always holds the real mesh, so shadows and GI stay exact at any distance.
#pragma once
#include <vector>

#include "world.h"

namespace trees {
inline constexpr int VARIANTS = 4;  // 0, 1 broadleaf; 2, 3 conifer

// One ModelData, mesh v = variant v (bark + leaf materials, small procedural textures). Trees stand on the
// origin, +Y up, roughly 7-13 m tall.
void build(ModelData& out);

// Broken tree of variant v in its local space: piece 0 the stump (trunk below cutY, stays standing), then the
// trunk above as logs (more breaks the higher severity >= 1), and branches / crowns that tore off on their own.
// Snapped ends get splintered fresh wood (material MAT_WOOD). mass = rough real mass (kg) at scale 1.
enum { PIECE_STUMP, PIECE_LOG, PIECE_BRANCH, PIECE_LEAVES };
struct Piece { MeshData mesh; int kind = PIECE_LOG; float mass = 0.f; };
inline constexpr uint32_t MAT_WOOD = 1 + VARIANTS;
void breakApart(int v, float cutY, float severity, uint32_t seed, std::vector<Piece>& out);

// Trunk collider for variant v: a closed 8-sided cylinder (pos3, stride 3) for physics::meshShape.
void trunkCollider(const ModelData& m, int v, std::vector<float>& verts, std::vector<uint32_t>& idx);

// Small plants, same mesh-per-variant layout: reed / flower clumps and shrubs, 0.3-1.9 m tall.
// The last four aren't plants: unit-size debris particles kept in the same model (water droplet, dirt clod,
// wood splinter along +y, leaf).
// Then ground clutter scattered densely near the camera: ferns, pebble clusters, fallen twigs, leaf litter.
enum { PLANT_CATTAIL, PLANT_DANDELION, PLANT_ROSE, PLANT_BUSH, PLANT_FLOWERBUSH, PLANT_SPRAY, PLANT_CLOD, PLANT_SPLINTER, PLANT_LEAF,
       PLANT_FERN, PLANT_PEBBLES, PLANT_TWIGS, PLANT_LITTER, PLANT_VARIANTS };
void buildPlants(ModelData& out);

// Bounding sphere of a variant in its local space (impostor frames are fitted to it).
void bounds(const MeshData& m, float center[3], float& radius);
}  // namespace trees

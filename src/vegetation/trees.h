// Procedural trees: a few mesh variants generated at startup, placed by hashed forest noise around the camera.
// The meshes go through the normal streamed-model path (bindless mesh + materials + BLAS), so the ray tracer
// and the raster G-buffer shade them like any glTF model. Far away the raster pass swaps them for octahedral
// impostors (impostor.glsl); the BVH always holds the real mesh, so shadows and GI stay exact at any distance.
#pragma once
#include <vector>

#include "world/world.h"

namespace trees {
inline constexpr int SPECIES = 6, VARIANTS = 2 * SPECIES;  // variant v = species v / 2, two seeds each

// Branching tree generator (trees/growth.h): a straight bole, then a leader that bends as it climbs, branches
// spiralling off it (phyllotaxis) whose length follows the crown envelope, sub-branches recursively down to twigs.
// Each limb is one continuous swept surface with a collar where it leaves its parent (trees/limb_mesh.h); leaves,
// or needles on conifers (leafWidth < 0.2), grow on shoots off the twigs (trees/foliage.h).
enum { CROWN_CONICAL, CROWN_SPHERICAL, CROWN_HEMISPHERICAL, CROWN_CYLINDRICAL, CROWN_TAPERED, CROWN_FLAME, CROWN_INVERSE, CROWN_COUNT };
inline const char* CROWN_NAMES[CROWN_COUNT] = {"Conical", "Spherical", "Hemispherical", "Cylindrical", "Tapered cylinder", "Flame", "Inverse conical"};
struct TreeParams {
    float height = 11.f, trunkRadius = 0.3f, bole = 0.3f, lean = 0.08f, gnarl = 0.25f;  // bole: clear trunk, fraction of height
    float flare = 0.8f, trunkWave = 0.5f;  // root buttresses, lumpy / wandering trunk and limbs
    int levels = 3, branches = 10, children = 5;      // branching levels below the leader, branches on the leader, per branch
    float branchAngle = 55.f, angleVar = 15.f;        // degrees off the parent
    float lengthRatio = 0.55f, childLength = 0.6f, radiusRatio = 0.6f;
    float gravity = 0.2f, upturn = 0.25f;
    int crown = CROWN_SPHERICAL;
    float leafSize = 0.14f, leafWidth = 0.5f, leafDensity = 1.f, leafSpread = 0.5f, leafDroop = 0.3f;
    float leafColor[3] = {0.09f, 0.19f, 0.045f}, hueVar = 0.3f, barkColor[3] = {0.16f, 0.11f, 0.07f};
    float density = 900.f, strength = 50.f;  // green wood: kg/m^3, bending strength (modulus of rupture, MPa)
    int growth = 0;                          // GROWTH_*: how the branches are laid out
    uint32_t seed = 1;
};
// GROWTH_RULES: recursive branching (Weber-Penn style), regular and tidy, right for conifers; GROWTH_COLONIZE: space
// colonization, branches competing for the space in the crown, the irregular look of a grown broadleaf tree
enum { GROWTH_RULES, GROWTH_COLONIZE, GROWTH_COUNT };
inline const char* GROWTH_NAMES[GROWTH_COUNT] = {"Branching rules", "Space colonization"};
inline const char* TREE_PRESETS[] = {"Oak", "Birch", "Maple", "Pine", "Spruce", "Willow", "Poplar", "Fir", "Cypress", "Acacia",
                                     "Cherry", "Beech", "Ash", "Dead tree"};
inline constexpr int TREE_PRESET_COUNT = 14;
void treePreset(int i, TreeParams& p);
inline TreeParams treeParams[SPECIES];  // species s = variants 2s, 2s + 1; set at startup by trees::defaults()
inline constexpr float NEEDLE_MAX_WIDTH = 0.2f;  // leafWidth below this grows needles: a conifer
inline bool conifer(const TreeParams& P) { return P.leafWidth < NEEDLE_MAX_WIDTH; }
// Leaves through the year (trees/season.cpp): build() grows broadleaf species as they are on this day of the year
// (fresh in spring, coloured in autumn, bare in winter); -1 = always as the sliders set them (summer). Conifers keep
// their needles. treeParams aren't changed.
inline float seasonDay = -1.f;
TreeParams seasonal(const TreeParams& P, int species);
// The day a season bucket stands for (the leaves change in steps, each a regrow): -1 for -1.
float seasonBucketDay(float dayOfYear);
inline constexpr int DEFAULT_PRESET[SPECIES] = {0, 3, 1, 4, 11, 2};  // oak, pine, birch, spruce, beech, maple
inline const char* SPECIES_NAMES[SPECIES] = {"Species A", "Species B", "Species C", "Species D", "Species E", "Species F"};

// Bushes from the same generator: several stems out of the ground, short branching, dense leaves and optional
// fruit (berries, rose hips / blossoms) hanging at the twig ends.
struct BushParams {
    float size = 1.f; int stems = 6; float spread = 0.55f, branchAngle = 42.f, gnarl = 0.35f; int levels = 2;
    float leafSize = 0.06f, leafDensity = 1.2f, leafColor[3] = {0.08f, 0.17f, 0.04f};
    float fruitDensity = 1.f, fruitSize = 0.012f, fruitColor[3] = {0.45f, 0.02f, 0.05f};
    uint32_t seed = 1;
};
enum { BUSH_PLAIN, BUSH_BERRY, BUSH_ROSE, BUSH_FLOWER, BUSH_KINDS };
inline BushParams bushParams[BUSH_KINDS];
void defaults();  // species / bush defaults (call once before build)

// One ModelData, mesh v = variant v (bark + leaf materials, small procedural textures). Trees stand on the
// origin, +Y up, roughly 7-13 m tall.
void build(ModelData& out);

// Pieces of a broken tree (imported trees, main.cpp assetBreak): stump stays standing, the rest is thrown.
// mass = rough real mass (kg) at scale 1. colVerts / colIdx (pos3): the piece's static collider (trunk + limbs)
// instead of its convex hull. hullPts / hullCount: a loose piece's collider as convex hulls (one per limb,
// hullCount points each, pos3), for a compound body; empty = the hull of the whole mesh.
enum { PIECE_STUMP, PIECE_LOG, PIECE_BRANCH, PIECE_LEAVES };
struct Piece {
    MeshData mesh; int kind = PIECE_LOG; float mass = 0.f;
    std::vector<float> colVerts; std::vector<uint32_t> colIdx;
    std::vector<float> hullPts; std::vector<uint32_t> hullCount;
};
inline constexpr uint32_t MAT_WOOD = 1 + VARIANTS;

// Wood mechanics. A procedural tree is a skeleton of tapered beams: the trunk in TRUNK_ELEMS pieces, then every
// limb segment, each hanging off its parent element. A Wood is one connected part of that: the standing tree
// (anchored: the trunk base held by the roots) or a piece broken off it (loose). Impacts are solved as beams:
//  - anchored, struck by a body of mass m: the impactor meets the struck limb's effective mass (reduced mass),
//    the deformation energy of the normal velocity component (impact angle) loads the cantilever chain from the
//    contact down to the roots, whose compliance (Euler-Bernoulli, unit-load method, + bending/torsion of every
//    element on the way, + local crushing) gives the peak force. The bending + torsion moment of that force and
//    of the self weight hanging outboard is checked against the wood's strength at every joint on the path;
//    the worst one past 1 snaps.
//  - loose, landing with impulse J: a free body. Every joint carries what it takes to decelerate the part on
//    its far side (d'Alembert: mass * acceleration of the rigid motion, lever to the joint); several can go.
// Leaves come off where the wood they hang on is jerked hard enough (velocity change at the cluster).
inline constexpr int TRUNK_ELEMS = 8;
struct Wood {
    int v = 0;
    bool anchored = true;
    std::vector<uint8_t> has;      // element is part of this wood
    std::vector<uint16_t> leaves;  // leaves left on each foliage cluster
    std::vector<int> cuts;         // elements whose base snapped: splintered end here or a stub on the parent
};
void woodWhole(int v, Wood& w);
// Nearest element of w to local point p (unscaled), distance from its surface.
int woodNearest(const Wood& w, const float p[3], float& dist);
// Element k's axis a..b (unscaled local) and base radius; false if it isn't part of w.
bool woodElem(const Wood& w, int k, float a[3], float b[3], float& r);
// First element of w (unscaled local) that a sphere of radius r moving a -> b touches, among the limbs without
// a physics collider (thin ones a rock would fly through). -1 if none; at = the contact point.
int woodSweep(const Wood& w, const float a[3], const float b[3], float r, float at[3]);
// Mass (kg), centre of mass and inertia about it (kg m^2, row major), scaled local metres: leaves count, spread
// out over the crown (a body built from the wood colliders alone would spin like a stick).
void woodMass(const Wood& w, float s, float& mass, float com[3], float I[9]);
// Point of w furthest along -dir (unscaled local): where a loose piece touches what it landed on.
void woodSupport(const Wood& w, const float dir[3], float out[3]);
// What water acts on (unscaled local): every limb as a tube (a3 b3 ra rb) and every leaf cluster as plates
// (centre3, one-sided leaf area).
void woodHydro(const Wood& w, std::vector<float>& tubes, std::vector<float>& plates);
struct WoodHit {
    std::vector<int> cuts;         // elements whose base fails (loose: several, none nested)
    float ratio = 0.f;             // worst stress / strength
    float impulse[3] = {};         // anchored: momentum the wood took from the impactor (N s, local axes)
    float keep = 1.f;              // anchored: fraction of the impactor's normal speed left after a snap
    std::vector<uint16_t> lose;    // leaves shaken off each cluster
    int lost = 0;
};
// Anchored wood at instance scale s struck at local p (unscaled) by mass m (kg) moving at vel (m/s, local axes).
WoodHit woodStrike(const Wood& w, float s, const float p[3], const float vel[3], float m);
// Loose wood at scale s: impulse J (N s, local axes) at local p, delivered over ~10 ms.
WoodHit woodLanding(const Wood& w, float s, const float p[3], const float J[3]);
// Apply a hit's leaf losses (local positions of the fallen leaves appended to shed) and its cuts: each cut
// subtree moves to a new loose Wood in off.
void woodApply(Wood& w, const WoodHit& h, std::vector<Wood>& off, std::vector<float>* shed);
// Mesh (local, unscaled), mass (kg at scale 1) and colliders: anchored -> colVerts / colIdx, loose -> hulls.
void woodMesh(const Wood& w, uint32_t seed, Piece& out);

// Collider for variant v (pos3, stride 3) for physics::meshShape: the trunk and its major limbs as closed prisms,
// so thrown things hit the branches too.
void trunkCollider(const ModelData& m, int v, std::vector<float>& verts, std::vector<uint32_t>& idx);

// Small plants, same mesh-per-variant layout: reed / flower clumps and bushes (bushParams), 0.3-1.9 m tall.
// The last four aren't plants: unit-size debris particles kept in the same model (water droplet, dirt clod,
// wood splinter along +y, leaf).
// Then ground clutter scattered densely near the camera: ferns, pebble clusters, fallen twigs, leaf litter.
// PLANT_EMBER: a glowing ember (debris of fires). Then small animals (sim/critters.h), unit size, body along +z, wings
// along x (the vertex shader flaps what lies beyond CRITTER_BODY): a firefly, a bird, two butterflies.
enum { PLANT_CATTAIL, PLANT_DANDELION, PLANT_ROSE, PLANT_BUSH, PLANT_FLOWERBUSH, PLANT_BERRYBUSH, PLANT_SPRAY, PLANT_CLOD, PLANT_SPLINTER, PLANT_LEAF,
       PLANT_FERN, PLANT_PEBBLES, PLANT_TWIGS, PLANT_LITTER, PLANT_EMBER, PLANT_FIREFLY, PLANT_BIRD, PLANT_BUTTERFLY, PLANT_BUTTERFLY2,
       PLANT_VARIANTS };
inline constexpr float CRITTER_BODY = 0.06f;  // half width of a critter's body (unit size): wings start here
void buildPlants(ModelData& out);

// Bounding sphere of a variant in its local space (impostor frames are fitted to it).
void bounds(const MeshData& m, float center[3], float& radius);
}  // namespace trees

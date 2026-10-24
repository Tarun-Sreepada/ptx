// Species presets and the startup defaults of the tree / bush generators.
#include "vegetation/trees.h"

namespace trees {
namespace {
// green wood (kg/m^3, modulus of rupture MPa) of each preset; the dead tree is dried out and rotten, light and brittle
const float WOOD[TREE_PRESET_COUNT][2] = {{1000, 57}, {900, 44}, {900, 53}, {800, 35}, {700, 34}, {900, 33}, {800, 36},
                                          {750, 38}, {800, 43}, {1100, 70}, {900, 55}, {1000, 59}, {900, 66}, {450, 18}};
void col(float* c, float r, float g, float b) { c[0] = r; c[1] = g; c[2] = b; }
}  // namespace

void defaults() {
    for (int s = 0; s < SPECIES; s++) { treePreset(DEFAULT_PRESET[s], treeParams[s]); treeParams[s].seed = 1 + s; }
    BushParams& plain = bushParams[BUSH_PLAIN];
    plain = {};
    plain.fruitDensity = 0.f;
    plain.size = 1.1f;
    BushParams& berry = bushParams[BUSH_BERRY];
    berry = {};
    berry.size = 0.9f; berry.stems = 7; col(berry.leafColor, 0.06f, 0.14f, 0.035f);
    berry.fruitDensity = 1.3f; berry.fruitSize = 0.011f;
    col(berry.fruitColor, 0.04f, 0.03f, 0.16f);  // blueberry
    BushParams& rose = bushParams[BUSH_ROSE];
    rose = {};
    rose.size = 0.7f; rose.stems = 5; rose.spread = 0.4f; rose.leafSize = 0.045f; col(rose.leafColor, 0.05f, 0.11f, 0.035f);
    rose.fruitDensity = 0.6f; rose.fruitSize = 0.03f; col(rose.fruitColor, 0.45f, 0.012f, 0.02f);
    BushParams& fl = bushParams[BUSH_FLOWER];
    fl = {};
    fl.size = 0.9f; col(fl.leafColor, 0.12f, 0.2f, 0.06f);
    fl.fruitDensity = 1.f; fl.fruitSize = 0.022f; col(fl.fruitColor, 0.75f, 0.6f, 0.7f);
}

void treePreset(int i, TreeParams& p) {
    p = {};
    switch (i) {
    case 0:  // oak: short bole, wide spreading crown, gnarled
        p.height = 11.f; p.trunkRadius = 0.36f; p.bole = 0.28f; p.gnarl = 0.4f; p.branches = 9; p.children = 5; p.branchAngle = 60.f;
        p.lengthRatio = 0.7f; p.crown = CROWN_HEMISPHERICAL; p.gravity = 0.25f; p.upturn = 0.2f; p.leafSize = 0.13f; p.leafDensity = 1.6f;
        col(p.leafColor, 0.08f, 0.17f, 0.04f);
        break;
    case 1:  // birch: slim, pale bark, light open crown with small leaves
        p.height = 13.f; p.trunkRadius = 0.18f; p.bole = 0.35f; p.gnarl = 0.15f; p.branches = 12; p.children = 4; p.branchAngle = 40.f;
        p.lengthRatio = 0.45f; p.crown = CROWN_TAPERED; p.gravity = 0.35f; p.leafSize = 0.07f; p.leafDensity = 1.2f; p.leafDroop = 0.4f;
        col(p.leafColor, 0.14f, 0.22f, 0.05f); col(p.barkColor, 0.55f, 0.53f, 0.5f);
        break;
    case 2:  // maple: round dense crown
        p.height = 12.f; p.trunkRadius = 0.3f; p.bole = 0.3f; p.branches = 10; p.children = 5; p.branchAngle = 45.f; p.lengthRatio = 0.6f;
        p.crown = CROWN_SPHERICAL; p.leafSize = 0.16f; p.leafWidth = 0.8f; p.leafDensity = 1.1f;
        col(p.leafColor, 0.12f, 0.2f, 0.045f);
        break;
    case 3:  // pine: tall bole, conical crown, long needles in brushes at the branch ends
        p.height = 14.f; p.trunkRadius = 0.28f; p.bole = 0.4f; p.gnarl = 0.12f; p.levels = 2; p.branches = 22; p.children = 6;
        p.branchAngle = 80.f; p.angleVar = 8.f; p.lengthRatio = 0.35f; p.childLength = 0.45f; p.crown = CROWN_CONICAL; p.gravity = 0.15f;
        p.upturn = 0.3f; p.leafSize = 0.14f; p.leafWidth = 0.12f; p.leafDensity = 2.6f; p.leafSpread = 0.35f; p.leafDroop = 0.05f; p.hueVar = 0.15f;
        col(p.leafColor, 0.04f, 0.1f, 0.045f); col(p.barkColor, 0.2f, 0.12f, 0.08f);
        break;
    case 4:  // spruce: narrow, branches all the way down, drooping, short dense needles
        p.height = 15.f; p.trunkRadius = 0.26f; p.bole = 0.08f; p.gnarl = 0.08f; p.levels = 2; p.branches = 34; p.children = 6;
        p.branchAngle = 85.f; p.angleVar = 6.f; p.lengthRatio = 0.3f; p.childLength = 0.4f; p.crown = CROWN_CONICAL; p.gravity = 0.45f;
        p.upturn = 0.15f; p.leafSize = 0.09f; p.leafWidth = 0.18f; p.leafDensity = 2.4f; p.leafSpread = 0.3f; p.leafDroop = 0.05f;
        col(p.leafColor, 0.03f, 0.08f, 0.05f); col(p.barkColor, 0.13f, 0.09f, 0.07f);
        break;
    case 5:  // willow: weeping
        p.height = 10.f; p.trunkRadius = 0.34f; p.bole = 0.25f; p.gnarl = 0.3f; p.branches = 10; p.children = 6; p.branchAngle = 35.f;
        p.lengthRatio = 0.8f; p.childLength = 0.8f; p.crown = CROWN_SPHERICAL; p.gravity = 1.2f; p.upturn = 0.05f; p.leafSize = 0.1f;
        p.leafWidth = 0.25f; p.leafDensity = 1.4f; p.leafDroop = 0.5f;
        col(p.leafColor, 0.12f, 0.2f, 0.06f);
        break;
    case 7:  // fir: dense narrow cone, upturned branch tips
        p.height = 16.f; p.trunkRadius = 0.27f; p.bole = 0.06f; p.gnarl = 0.06f; p.levels = 2; p.branches = 40; p.children = 7;
        p.branchAngle = 75.f; p.angleVar = 5.f; p.lengthRatio = 0.28f; p.childLength = 0.45f; p.crown = CROWN_CONICAL; p.gravity = 0.1f;
        p.upturn = 0.45f; p.leafSize = 0.09f; p.leafWidth = 0.18f; p.leafDensity = 2.6f; p.leafSpread = 0.3f; p.leafDroop = 0.03f;
        col(p.leafColor, 0.025f, 0.075f, 0.045f); col(p.barkColor, 0.17f, 0.15f, 0.14f);
        break;
    case 8:  // cypress: tall flame, branches hugging the trunk
        p.height = 14.f; p.trunkRadius = 0.22f; p.bole = 0.05f; p.gnarl = 0.1f; p.levels = 2; p.branches = 36; p.children = 6;
        p.branchAngle = 18.f; p.angleVar = 6.f; p.lengthRatio = 0.22f; p.childLength = 0.5f; p.crown = CROWN_FLAME; p.gravity = 0.f;
        p.upturn = 0.6f; p.leafSize = 0.08f; p.leafWidth = 0.18f; p.leafDensity = 2.8f; p.leafSpread = 0.3f; p.leafDroop = 0.05f;
        col(p.leafColor, 0.03f, 0.08f, 0.04f); col(p.barkColor, 0.22f, 0.13f, 0.09f);
        break;
    case 9:  // acacia: crooked trunk, flat umbrella crown of tiny leaves
        p.height = 8.f; p.trunkRadius = 0.25f; p.bole = 0.45f; p.lean = 0.25f; p.gnarl = 0.55f; p.branches = 6; p.children = 6;
        p.branchAngle = 55.f; p.lengthRatio = 1.f; p.childLength = 0.7f; p.crown = CROWN_INVERSE; p.gravity = 0.f; p.upturn = 0.05f;
        p.leafSize = 0.05f; p.leafWidth = 0.6f; p.leafDensity = 2.f; p.leafSpread = 0.5f; p.leafDroop = 0.1f;
        col(p.leafColor, 0.1f, 0.16f, 0.05f); col(p.barkColor, 0.2f, 0.17f, 0.13f);
        break;
    case 10:  // cherry: low spreading, pink blossom
        p.height = 7.f; p.trunkRadius = 0.22f; p.bole = 0.25f; p.gnarl = 0.35f; p.branches = 7; p.children = 6; p.branchAngle = 50.f;
        p.lengthRatio = 0.75f; p.crown = CROWN_HEMISPHERICAL; p.gravity = 0.15f; p.leafSize = 0.07f; p.leafWidth = 0.8f; p.leafDensity = 1.8f;
        col(p.leafColor, 0.75f, 0.42f, 0.5f); col(p.barkColor, 0.18f, 0.09f, 0.08f);
        break;
    case 11:  // beech: smooth grey bark, tall dense dome
        p.height = 16.f; p.trunkRadius = 0.35f; p.bole = 0.35f; p.gnarl = 0.15f; p.trunkWave = 0.25f; p.branches = 11; p.children = 6;
        p.branchAngle = 45.f; p.lengthRatio = 0.6f; p.crown = CROWN_SPHERICAL; p.leafSize = 0.1f; p.leafWidth = 0.6f; p.leafDensity = 1.7f;
        col(p.leafColor, 0.1f, 0.2f, 0.04f); col(p.barkColor, 0.35f, 0.34f, 0.32f);
        break;
    case 12:  // ash: open, upswept, compound leaves (narrow)
        p.height = 15.f; p.trunkRadius = 0.3f; p.bole = 0.4f; p.branches = 9; p.children = 4; p.branchAngle = 35.f; p.lengthRatio = 0.55f;
        p.crown = CROWN_TAPERED; p.upturn = 0.45f; p.gravity = 0.1f; p.leafSize = 0.12f; p.leafWidth = 0.3f; p.leafDensity = 1.4f;
        col(p.leafColor, 0.09f, 0.18f, 0.05f); col(p.barkColor, 0.28f, 0.26f, 0.22f);
        break;
    case 13:  // dead tree: bare, twisted, grey
        p.height = 9.f; p.trunkRadius = 0.3f; p.bole = 0.3f; p.gnarl = 0.6f; p.lean = 0.2f; p.levels = 2; p.branches = 7; p.children = 4;
        p.branchAngle = 50.f; p.lengthRatio = 0.6f; p.crown = CROWN_SPHERICAL; p.leafDensity = 0.f; p.trunkWave = 0.9f;
        col(p.barkColor, 0.3f, 0.28f, 0.25f);
        break;
    default:  // poplar: columnar
        p.height = 17.f; p.trunkRadius = 0.28f; p.bole = 0.15f; p.branches = 18; p.children = 4; p.branchAngle = 22.f; p.angleVar = 8.f;
        p.lengthRatio = 0.25f; p.crown = CROWN_CYLINDRICAL; p.gravity = 0.05f; p.upturn = 0.4f; p.leafSize = 0.1f; p.leafWidth = 0.8f;
        col(p.leafColor, 0.1f, 0.19f, 0.05f);
        break;
    }
    if (i >= 0 && i < TREE_PRESET_COUNT) { p.density = WOOD[i][0]; p.strength = WOOD[i][1]; }
    // broadleaf crowns grow by competing for space; conifers, the columnar poplar and the dead tree by rules
    if (i == 0 || i == 2 || i == 9 || i == 10 || i == 11 || i == 12) p.growth = GROWTH_COLONIZE;
    p.seed = 1;
}
}  // namespace trees

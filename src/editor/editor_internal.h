// Editor internals shared by its parts (editor.cpp is the public API): the cells being edited and the selection,
// the asset library and its background import worker, the scatter brush and the world map tools.
#pragma once
#include "editor/editor.h"

#include <imgui.h>
#include <stb_image.h>
#include <stb_image_write.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <thread>
#include <cstdarg>
#include <chrono>
#include <cfloat>
#include <cmath>
#include <cstring>
#include <filesystem>
#include <map>
#include <meshoptimizer.h>
#include "world/matlib.h"
#include "world/worldmap.h"
#include "physics/physics.h"
#include "vegetation/trees.h"

namespace editor {

inline constexpr float DEG = 3.14159265f / 180.f;
enum Kind { K_PRIM, K_MESH, K_ASSET, K_LIGHT, K_VOLUME, K_ZONE, K_TREE, K_ROCK, K_COUNT };
inline const char* KIND_NAME[K_COUNT] = {"Primitive", "Mesh", "Asset", "Light", "Bake volume", "Flatten zone", "Tree", "Rock"};
enum Shape { BOX, CYLINDER };
inline const char* TREE_NAME[trees::VARIANTS] = {"Species A, seed 1", "Species A, seed 2", "Species B, seed 1", "Species B, seed 2",
                                          "Species C, seed 1", "Species C, seed 2", "Species D, seed 1", "Species D, seed 2",
                                          "Species E, seed 1", "Species E, seed 2", "Species F, seed 1", "Species F, seed 2"};
// Everything in a cell as one editable list. Transforms are position + Euler angles (degrees, applied
// yaw * pitch * roll) + scale; primitives keep their size in the mesh (scale 1).
struct Item {
    Kind kind = K_PRIM;
    std::string name;
    float pos[3] = {}, rot[3] = {}, scale[3] = {1, 1, 1};  // rot: pitch (x), yaw (y), roll (z)
    // primitive
    int shape = BOX, sides = 24;
    float size[3] = {1, 1, 1};  // box x, y, z; cylinder diameter x, height y, diameter z
    std::string mat = "brick";
    float uvScale = 1.f, tint[3] = {1, 1, 1};
    float strength = 1.f;
    bool collide = true;
    // mesh / asset
    int mesh = -1;
    std::string asset, extras;
    // light
    float color[3] = {1.f, 0.85f, 0.65f}, intensity = 400.f;
    // zone
    float falloff = 8.f;
    std::string zoneMat;  // ground texture, empty = none
    float zoneTile = 2.f;
    // tree
    int variant = 0;
    // where it went in the live model (in-place moves)
    int slot = -1;
};
struct ECell {
    int cx = 0, cz = 0;
    std::vector<Item> items;
    std::vector<MeshData> meshes;  // the cell's own (imported) meshes, for K_MESH items
    std::vector<MaterialData> mats;
    std::vector<TextureData> texs;
    bool modified = false, dirty = false;  // differs from what was loaded / from disk
    bool rebuild = false, refresh = false;
    std::shared_ptr<Model> live;
    std::vector<std::vector<Item>> undo;
};
inline Host host;
inline bool on = false;
inline std::map<std::pair<int, int>, ECell> cells;
struct Sel { int cx = 0, cz = 0, idx = -1; };
inline Sel sel;
inline View view;
inline float grid = 0.25f, rotStep = 15.f;
inline std::vector<std::pair<int, int>> reverts;  // cells to reload from disk in apply()
inline bool snap = true, showVolumes = true, showAll = false;
inline bool groundSnap = true;  // placed / dragged assets and trees / rocks sit on the terrain...
inline float sinkDepth = 0.f;   // ...this far into it (m): trunks, foundations, rocks bedded in
inline std::string newMat = "brick";
inline std::string status;
inline std::vector<std::string> assetNames;
inline std::vector<std::string> assetDirs;  // folders under assets/, relative
inline char importPath[512] = "";
inline double lastRebuild = 0;
// dragging
inline bool dragging = false, dragVertical = false, dragMoved = false;
inline float dragPlaneY = 0, dragOff[3] = {}, dragN[3] = {}, dragP0[3] = {};
inline bool snapInvert = false;  // Shift held: snapping temporarily the other way
inline bool moveOnly = false;    // the pending refresh only moved meshes / assets / lights
struct Hit { int cx, cz, idx; float t; };
// ---------------------------------------------------------------- asset library: import, thumbnails, browser
// Imports and thumbnails run on one background thread; finished thumbnails are uploaded in ui().
inline constexpr int THUMB = 128;
struct Job { bool import; std::string path; float keep = 1.f; std::string as; bool materials = false; };  // import a source file (keep = triangle fraction, as = asset name), or (re)make the thumbnail of asset <path>
inline std::mutex jobMu;
inline std::condition_variable jobCv;
inline std::deque<Job> jobQ;
inline bool workerUp = false;
inline std::atomic<int> jobsLeft{0};
struct Done { std::string name; std::vector<uint8_t> rgba; std::string msg; bool rescan = false; float foot = 0; };
inline std::vector<Done> jobDone;  // under jobMu
inline std::vector<std::string> pickedPaths;  // from the file picker, under jobMu
struct Thumb { uint64_t tex = 0; bool queued = false; float foot = 0; };  // foot: footprint radius (m) about the origin
inline std::map<std::string, Thumb> thumbs;
inline std::string dragAsset;  // asset dragged out of the browser, placed where it is released over the world
inline bool showBrowser = true;
inline float importKeep = 1.f;  // fraction of triangles kept on import (meshoptimizer simplify)
// Native file picker (KDE kdialog, else zenity) on its own thread; the chosen paths are imported.
inline std::atomic<bool> pickerOpen{false};
// ---- scatter: a weighted mix of assets dropped over a disc, dart throwing with footprint spacing
struct MixEntry { std::string name; float weight = 1.f; };
inline std::vector<MixEntry> scatterMix;
inline float scatterRadius = 20.f, scatterDensity = 2.f, scatterSpacing = 0.8f, scatterScale[2] = {0.8f, 1.2f};
inline bool showScatter = false;
// ---- library management: folders, rename / move (leaves an alias so placed copies keep working), delete
enum { POP_NONE, POP_NEWFOLDER, POP_RENAME, POP_RENAMEDIR, POP_DELETE };
inline int popup = POP_NONE, popupOpen = POP_NONE;
inline char popupName[128] = "";
inline std::string popupTarget, curDir;
// ---------------------------------------------------------------- world map editor (M)
// Terrain sculpting and painting on the world map grid (worldmap.h), drawn rivers, and the snow brush. While it's
// on, the left mouse paints with the current tool instead of picking items.
inline bool mapMode = false, mapOpen = true;
inline int mapTool = worldmap::RAISE, mapTexSlot = 1;
inline float mapRadius = 12.f, mapStrength = 2.f, mapFlatten = 0.f;
inline bool stroking = false;
enum { MT_RIVER = 100, MT_SNOW_ADD, MT_SNOW_REMOVE };
inline std::vector<worldmap::RiverPt> riverPts;
inline float riverW = 4.f, riverD = 1.2f;  // the pen: each point takes the width / depth it was drawn with (wheel / Shift+wheel)
inline float snowRate = 2.f;

// procedural buildings (building_gen.cpp)
struct BuildingParams {
    float width = 9.f, depth = 7.f, storeyH = 3.f, wall = 0.25f;
    int storeys = 2, style = 0;
    float windowW = 1.2f, windowH = 1.3f, windowSill = 0.9f, windowSpacing = 2.4f;
    bool gable = true;
};
void generateBuilding(const BuildingParams& P, const float at[3], float yawDeg, std::vector<Item>& out);
const char* buildingStyleName(int i);
int buildingStyleCount();
void buildingPanel();

// ---------------------------------------------------------------- functions (one file per part)
double now();
Xf compose(const float p[3], const float r[3], const float s[3]);
void decompose(const Xf& x, float p[3], float r[3], float s[3]);
Xf inverse(const Xf& a);
Xf itemXf(const Item& it);
float snapv(float v, float g);
int cellOf(float x);
std::string cellPath(int cx, int cz);
void genPrim(const Item& it, uint32_t mat, MeshData& m);
std::string fmt(const char* f, ...);
std::string primExtras(const Item& it);
std::string setKey(std::string json, const char* key, float v);
ECell importCell(int cx, int cz, const std::shared_ptr<Model>& live);
ECell* open(int cx, int cz);
void build(ECell& c, ModelData& d, bool shapes);
void freeModel(std::shared_ptr<Model> m);
bool pushLive(ECell& c);
bool moveInPlace(ECell& c, const Item& it);
Item* selected();
ECell* selCell();
void pushUndo(ECell& c);
void changed(ECell& c, bool geometry);
void rehome(ECell& c, int idx);
void localBounds(const ECell& c, const Item& it, float mn[3], float mx[3]);
Xf boundsXf(const Item& it);
void snapToGround(const ECell& c, Item& it);
bool groundable(const Item& it);
bool pickable(const Item& it);
float rayItem(const ECell& c, const Item& it, const float o[3], const float d[3]);
Hit pick(const float o[3], const float d[3]);
float rayTerrain(const float o[3], const float d[3]);
void mouseRay(float d[3]);
bool project(const float p[3], ImVec2& s);
void placePoint(float out[3]);
float camYawSnapped();
void addItem(Item it, bool atCursor = true);
Item prim(int shape, float sx, float sy, float sz, const std::string& mat, const char* name);
void simplify(ModelData& m, float keep);
size_t triCount(const ModelData& m);
std::string lower(std::string s);
std::string assetName(const std::filesystem::path& p);
std::vector<uint8_t> renderThumb(const ModelData& m);
std::string thumbPath(const std::string& name);
float footprint(const ModelData& m);
std::vector<uint8_t> makeThumb(const std::string& name, const ModelData* loaded, float& foot);
int extractMaterials(const ModelData& m, const std::string& name, std::string& names);
Done runJob(const Job& j);
void queueJob(Job j);
void scanAssets();
void importPaths(const std::vector<std::string>& paths);
void pollJobs();
bool mousePoint(float out[3]);
void addAsset(const std::string& name, const float* at);
void openPicker(bool folders);
void openFolder(const std::string& dir);
float footOf(const std::string& name);
void scatter(const float at[3]);
void unscatter(const float at[3]);
void scatterPanel();
std::string dirOf(const std::string& n);
std::string baseOf(const std::string& n);
bool isLoose(const std::string& n);
std::string glbName(const std::string& dir, const std::string& base);
void renameThumb(const std::string& from, const std::string& to);
bool moveAsset(const std::string& name, const std::string& dir, const std::string& base);
void deleteAsset(const std::string& name);
void renameFolder(const std::string& dir, const std::string& newBase);
void deleteFolder(const std::string& dir);
void assetBrowser();
void assetDrop();
void deleteSelected();
void duplicateSelected();
void drawBox(ImDrawList* dl, const ECell& c, const Item& it, ImU32 col, float thick);
bool matCombo(const char* label, std::string& mat, bool allowNone = false);
void inspector();
bool terrainUnderMouse(float out[3]);
void groundCircle(const float c[3], float r, ImU32 col);
void commitRiver();
void mapTools();
}  // namespace editor

// In-engine world editor (Tab): place, move, rotate and delete breakable primitives (boxes, cylinders with a
// material, size and strength), imported .glb assets, lights, bake volumes, flatten zones and hand-placed trees and
// rocks; saves each edited 64 m cell back to <world>/cells/<x>_<z>.glb (the streamer's own format, so Blender-made
// and editor-made cells mix). Edits go live at once: an edited cell's model is rebuilt, uploaded and swapped into
// the stream (stream::setCellModel), pure moves just re-place it (stream::refreshCell).
#pragma once
#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "world.h"

namespace editor {
struct Host {
    std::string worldDir;
    std::function<bool(Model&)> upload;                  // record the model's GPU upload (upload window only)
    std::function<void(std::shared_ptr<Model>)> retire;  // free a model's GPU side and shapes after this frame
    std::function<float(float, float)> height;           // terrain height
    std::function<uint64_t(const uint8_t*, int, int)> thumbnail;  // RGBA8 -> ImTextureID for the asset browser
    std::function<void(float, float, float, float)> snowBrush;    // world map snow brush this frame: x, z, radius, m/s (0 radius = off)
};
// camera for picking / overlay: unit vectors, tan of half the vertical fov
struct View { float pos[3], fwd[3], right[3], up[3]; float tanHalfFov; };

void init(const Host& h);
bool active();
void setActive(bool on);
bool wantsMouse();         // left mouse is the editor's (no rock throwing)
// True (once) when the last change only moved meshes / lights: the renderer can skip re-placing the terrain,
// vegetation and bake volumes.
bool takeMoveOnly();
void ui(const View& v);    // inside the ImGui frame: panels, picking, dragging, overlay
void apply();              // inside the upload window: push edited cells into the stream
void demoHouse();          // build the example house at the placement point (also --edit-demo)
bool saveAll(std::string& err);
// Files / folders dropped on the window: .gltf / .glb (found recursively in folders) are converted into
// <world>/assets/<name>.glb, PNG / JPG go to <world>/textures/. Runs in the background; opens the editor.
void dropFiles(const std::vector<std::string>& paths);
}  // namespace editor

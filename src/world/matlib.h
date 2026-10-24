// Building materials for the editor's primitives: a built-in procedural set (brick, concrete, ...) and any PNG / JPG
// in <world>/textures/. A texture covers 1 m x 1 m at uv scale 1. Strength scales how much energy it takes to
// breach the material (main.cpp breakStatic; brick = 1).
#pragma once
#include <string>
#include <vector>

#include "world/world.h"

namespace matlib {
struct Entry {
    std::string name;      // procedural name, or "file:<name>" for <world>/textures/<name>
    float strength;        // default breach strength
    float swatch[3];       // average colour, for the UI
};
void init(const std::string& worldDir);
void rescan();  // pick up new files in <world>/textures/
const std::vector<Entry>& list();
const Entry* find(const std::string& name);
// RGBA8 sRGB, cached. Unknown names give a grey checker.
const TextureData& texture(const std::string& name);
// PBR companions, cached: normal (rgb, tangent space) + height (a), ORM (r occlusion, g roughness, b metalness).
// heightScale = parallax depth in metres (0 for photos).
struct Maps { TextureData normal, orm; float heightScale = 0.f; };
const Maps& maps(const std::string& name);
}  // namespace matlib

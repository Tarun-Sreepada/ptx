// World map window (M): biomes / water / relief from above, markers, teleport.
#pragma once

namespace worldmap_ui {
void window();                  // every frame (M toggles it)
bool& isOpen();
void invalidate();              // the terrain changed: redraw
void teleport(float x, float z);
}

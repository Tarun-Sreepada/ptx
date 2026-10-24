// The light list each frame: the drifting orbs (faded at the edge of their tile), then the strongest streamed lights
// near the camera (switched-off buildings' lights after the active ones, for the lights-on bake), the triangle light
// list + light tree, and the light / bake counts in the frame uniforms.
#pragma once

void uploadLights();

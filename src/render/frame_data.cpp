#include "render/frame_data.h"

#include <cstring>

#include "app/frame_state.h"
#include "app/settings.h"
#include "core/prof.h"
#include "render/frame_ubo.h"
#include "render/light_upload.h"
#include "render/ubo_fill.h"
#include "scene/grass.h"
#include "scene/placement/placement.h"
#include "scene/scene_gpu.h"

float jitter[2] = {0, 0}, prevJitter[2] = {0, 0};

// ---------------------------------------------------------------- per-frame
uint32_t flags() {
    return (S.restir ? 1u : 0u) | (S.restirTemporal ? 2u : 0u) | (S.restirSpatial ? 4u : 0u) | (S.accumulate ? 8u : 0u) |
           (S.restirGI ? 16u : 0u) | (S.giTemporal ? 32u : 0u) | (S.giSpatial ? 64u : 0u) | (S.giEnabled ? 0u : 128u) |
           (S.giCache ? 256u : 0u) | (S.giChecker ? 512u : 0u) | (S.giProbes ? 1024u : 0u) | (S.microShadow ? 2048u : 0u);
}

// The frame's uniforms (camera, sky, materials, world state, weather, post), the light list, then placement of the
// rocks / trees / plants around the camera when it moved far enough.
void updateFrameData() {
    PROF_FUNC();
    UBO u{};
    fillCameraSky(u);
    fillSurfaceParams(u);
    fillGrassParams(u);
    fillWorldState(u);
    fillWeather(u);
    fillPost(u);
    memcpy(uboBuf.map, &u, sizeof u);
    uploadLights();
    placeScene();
}

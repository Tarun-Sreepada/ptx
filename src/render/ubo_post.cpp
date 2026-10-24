#include "render/ubo_fill.h"

#include <vector>
#include <string>
#include <functional>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <cstddef>
#include <vulkan/vulkan.h>
#include "core/prof.h"
#include "core/jobs.h"
#include "core/simd.h"
#include "vegetation/trees.h"
#include "world/worldmap.h"
#include "world/terrain.h"
#include "world/world.h"
#include "app/frame_state.h"
#include "app/settings.h"
#include "core/limits.h"
#include "core/math.h"
#include "destruction/rocks.h"
#include "destruction/thrown.h"
#include "destruction/tree_damage.h"
#include "gpu/accel.h"
#include "gpu/swapchain.h"
#include "render/bake.h"
#include "render/crater_bins.h"
#include "render/frame_ubo.h"
#include "render/ground_textures.h"
#include "render/ground_vt.h"
#include "render/light_build.h"
#include "render/lights.h"
#include "render/targets.h"
#include "scene/camera.h"
#include "scene/grass.h"
#include "scene/ground_mask.h"
#include "scene/model_instances.h"
#include "scene/models.h"
#include "scene/rocks.h"
#include "scene/scene_gpu.h"
#include "scene/stream_sync.h"
#include "scene/vegetation.h"
#include "sim/effects.h"
#include "sim/water.h"
#include "sim/water_splash.h"
#include "sim/world_sim.h"
#include "sky/celestial.h"
#include "sky/lightning.h"
#include "world/terrain_query.h"

void fillPost(UBO& u) {
        static double lastPostT = simTime;
        float pdt = std::clamp((float)(simTime - lastPostT), 0.f, 0.25f);
        lastPostT = simTime;
        u.post0[0] = S.evComp; u.post0[1] = S.autoExposure ? 1.f : 0.f; u.post0[2] = S.adaptUp; u.post0[3] = S.adaptDown;
        float ap = S.physCam ? std::clamp(4.f / S.fStop, 0.3f, 1.5f) : 1.f;  // wide open vignettes more
        u.post1[0] = S.bloom; u.post1[1] = 0; u.post1[2] = S.vignette * ap; u.post1[3] = S.chromAb;
        u.post2[0] = S.grain * (S.physCam ? sqrtf(S.iso / 100.f) : 1.f); u.post2[1] = S.sharpen; u.post2[2] = (float)S.tonemapper; u.post2[3] = pdt;
        float focal = 0.012f / tanf(cam.fovDeg * 0.5f * 3.14159265f / 180.f);  // 24 mm tall sensor
        u.post3[0] = S.focusDist; u.post3[1] = focal / std::max(S.fStop, 0.5f);
        u.post3[2] = !S.motionBlur ? 0.f : S.physCam ? std::clamp((1.f / S.shutterDen) / std::max(frameDt, 1e-3f), 0.f, 2.f) : S.mbShutter;
        u.post3[3] = S.dof ? 1.f : 0.f;
        if (S.physCam) {  // EV100 = log2(N^2 / t) - log2(ISO / 100); exposure = 1 / (1.2 * 2^EV100) in cd/m^2
            float ev100 = log2f(S.fStop * S.fStop * S.shutterDen) - log2f(S.iso / 100.f);
            u.params[0] = S.sceneLum / (1.2f * exp2f(ev100));
            u.post0[1] = 0.f;
        }
        u.post4[0] = (float)hdrActive; u.post4[1] = S.paperWhite; u.post4[2] = S.peakNits; u.post4[3] = (float)OW / OH;
}

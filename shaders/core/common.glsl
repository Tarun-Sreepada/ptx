// Everything a compute pass may need, in dependency order (shaders/README in the folder layout).
#ifndef CORE_COMMON
#define CORE_COMMON

#extension GL_EXT_ray_query : require
#define PI 3.14159265359

#define FLAG_RESTIR          1u
#define FLAG_RESTIR_TEMPORAL 2u
#define FLAG_RESTIR_SPATIAL  4u
#define FLAG_ACCUM           8u
#define FLAG_GI_RESTIR      16u
#define FLAG_GI_TEMPORAL    32u
#define FLAG_GI_SPATIAL     64u
#define FLAG_NO_GI         128u   // direct light only (A/B view of what GI adds)
#define FLAG_MICRO_SHADOW 2048u   // material occlusion shadows direct light / occludes glossy reflections (brdf.glsl)
#include "core/frame.glsl"
#include "core/bindings.glsl"
#include "core/util.glsl"
#include "materials/brdf.glsl"
#include "terrain/terrain.glsl"
#include "materials/model.glsl"
#include "terrain/ground.glsl"
#include "scene/wind.glsl"
#include "core/camera.glsl"
#include "sky/sky.glsl"
#include "sky/sky_with_sun.glsl"
#include "lighting/trace.glsl"
#include "lighting/sun.glsl"
#include "lighting/lightning_light.glsl"
#include "core/gbuffer.glsl"
#include "lighting/lights.glsl"
#include "lighting/fog.glsl"
#include "core/restir.glsl"
#endif

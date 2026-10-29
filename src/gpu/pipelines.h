// The one descriptor set every pass binds (bindings documented in createPipelineLayout), the compute pipelines
// (one per shaders/*.comp, indexed by P_*) and the raster G-buffer pipelines. Pipelines compile in parallel on the
// job pool; the driver's pipeline cache is kept in the build folder, tagged with a hash of the SPIR-V so a stale
// cache is dropped instead of growing forever.
#pragma once
#include <vulkan/vulkan.h>

extern VkDescriptorPool descPool, imguiPool;
extern VkDescriptorSet descSet;
extern VkPipelineLayout pipeLayout;
extern VkPipelineCache pipeCache;

enum { P_SKY, P_TERRAIN, P_GRASS, P_GRASS_TAIL, P_PRIMARY, P_RESTIR_INIT, P_RESTIR_SPATIAL, P_GI, P_GI_SPATIAL, P_TEMPORAL, P_ATROUS, P_COMPOSITE,
       P_DLSS_PREP, P_DLSS_TONEMAP, P_IMPOSTOR_BAKE, P_PROBE_BAKE, P_PROBE_FILTER, P_WIND, P_EXPOSURE, P_BLOOM, P_POST, P_PRESENT_HDR, P_VOLFOG,
       P_SURFACE_FX, P_WORLDSIM, P_RADCACHE, P_SKIN, P_VTGEN, P_OCCLUSION, P_CLOUD, P_VEGCULL, P_ROCKCULL, P_COUNT };

extern VkPipeline pipes[P_COUNT];
extern VkPipeline rasterTerrain, rasterGrass, rasterRock, rasterModel, rasterImpostor;
extern VkPipeline rasterModelMesh;  // trees and plants as meshlets (null without mesh shaders)

// Descriptor set layout, pool and set, pipeline layout and the pipeline cache (main thread, fast).
void createPipelineLayout();
// Every pipeline, compiled in parallel (any thread; blocks until all are built).
void createPipelines();
void savePipelineCache();  // at exit

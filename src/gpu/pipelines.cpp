#include "gpu/pipelines.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/jobs.h"
#include "core/limits.h"
#include "core/paths.h"
#include "core/prof.h"
#include "gpu/context.h"
#include "gpu/vk.h"
#include "render/targets.h"

VkDescriptorPool descPool, imguiPool;
VkDescriptorSet descSet;
VkPipelineLayout pipeLayout;
VkPipelineCache pipeCache;
VkPipeline pipes[P_COUNT];
VkPipeline rasterTerrain, rasterGrass, rasterRock, rasterModel, rasterImpostor;

namespace {
VkDescriptorSetLayout setLayout;

const char* PIPE_NAMES[P_COUNT] = {"sky", "terrain", "grass", "grass_tail", "primary", "restir_initial", "restir_spatial", "gi",
                                   "restir_gi_spatial", "temporal", "atrous", "composite", "dlss_prep", "dlss_tonemap", "impostor_bake",
                                   "probe_bake", "probe_filter", "wind", "exposure", "bloom", "post", "present_hdr", "volfog", "surface_fx",
                                   "worldsim", "radcache_resolve", "skin", "vtgen", "occlusion", "cloud"};
constexpr uint32_t PUSH_CONSTANT_BYTES = 16;     // 4 x uint (every compute pass)
constexpr uint32_t CACHE_MAGIC = 0x50544350;     // "PCTP": header in front of the driver's blob
const char* CACHE_FILE = "pipeline_cache.bin";   // in the build folder, next to shaders/

std::vector<char> readFile(const std::string& path) {
    std::ifstream f(path, std::ios::binary | std::ios::ate);
    if (!f) { fprintf(stderr, "cannot open %s\n", path.c_str()); abort(); }
    std::vector<char> d((size_t)f.tellg());
    f.seekg(0);
    f.read(d.data(), d.size());
    return d;
}

std::string cachePath() { return (std::filesystem::path(SHADER_DIR) / ".." / CACHE_FILE).lexically_normal().string(); }

// FNV-1a over every SPIR-V file: a shader change invalidates the whole cache (the driver's own disk cache still
// remembers unchanged shaders, so nothing recompiles that didn't change)
uint64_t shaderHash() {
    uint64_t h = 1469598103934665603ull;
    std::error_code ec;
    std::vector<std::string> files;
    for (auto& e : std::filesystem::directory_iterator(SHADER_DIR, ec))
        if (e.path().extension() == ".spv") files.push_back(e.path().string());
    std::sort(files.begin(), files.end());
    for (auto& f : files) {
        std::vector<char> d = readFile(f);
        for (char c : f) h = (h ^ (uint8_t)c) * 1099511628211ull;
        for (char c : d) h = (h ^ (uint8_t)c) * 1099511628211ull;
    }
    return h;
}
uint64_t currentHash = 0;

void loadPipelineCache() {
    currentHash = shaderHash();
    std::vector<char> blob;
    if (std::ifstream f{cachePath(), std::ios::binary | std::ios::ate}) {
        size_t n = (size_t)f.tellg();
        uint32_t magic = 0;
        uint64_t hash = 0;
        f.seekg(0);
        if (n > 12 && f.read((char*)&magic, 4) && f.read((char*)&hash, 8) && magic == CACHE_MAGIC && hash == currentHash) {
            blob.resize(n - 12);
            f.read(blob.data(), blob.size());
        }
    }
    VkPipelineCacheCreateInfo pci{VK_STRUCTURE_TYPE_PIPELINE_CACHE_CREATE_INFO};
    pci.initialDataSize = blob.size();  // the driver validates its own header and ignores a foreign blob
    pci.pInitialData = blob.data();
    if (vkCreatePipelineCache(dev, &pci, nullptr, &pipeCache) != VK_SUCCESS) {
        pci.initialDataSize = 0;
        VK_CHECK(vkCreatePipelineCache(dev, &pci, nullptr, &pipeCache));
    }
}

VkShaderModule loadModule(const char* name) {
    auto code = readFile(std::string(SHADER_DIR) + name);
    VkShaderModuleCreateInfo smi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    smi.codeSize = code.size();
    smi.pCode = (const uint32_t*)code.data();
    VkShaderModule mod;
    VK_CHECK(vkCreateShaderModule(dev, &smi, nullptr, &mod));
    return mod;
}

VkPipeline mkComputePipeline(const char* name) {
    VkShaderModule mod = loadModule((std::string(name) + ".comp.spv").c_str());
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, mod, "main", nullptr};
    cpi.layout = pipeLayout;
    VkPipeline p;
    VK_CHECK(vkCreateComputePipelines(dev, pipeCache, 1, &cpi, nullptr, &p));
    vkDestroyShaderModule(dev, mod, nullptr);
    return p;
}

// G-buffer raster pipeline: GBUF_TARGETS colour targets (gDepth, gNorm, gAlbedo, gEmis, gMat, gMat2, gTan, gVel) +
// reverse-Z depth. stride 0 = no vertex input (vertex pulling)
VkPipeline mkRasterPipeline(const char* vert, uint32_t stride, bool hasNormal, const char* frag) {
    VkShaderModule vm = loadModule(vert), fm = loadModule(frag);
    VkPipelineShaderStageCreateInfo st[2] = {
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_VERTEX_BIT, vm, "main", nullptr},
        {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_FRAGMENT_BIT, fm, "main", nullptr}};
    VkVertexInputBindingDescription vb{0, stride, VK_VERTEX_INPUT_RATE_VERTEX};
    VkVertexInputAttributeDescription va[2] = {{0, 0, VK_FORMAT_R32G32B32_SFLOAT, 0}, {1, 0, VK_FORMAT_R32G32B32_SFLOAT, 12}};
    VkPipelineVertexInputStateCreateInfo vi{VK_STRUCTURE_TYPE_PIPELINE_VERTEX_INPUT_STATE_CREATE_INFO};
    vi.vertexBindingDescriptionCount = stride ? 1 : 0;
    vi.pVertexBindingDescriptions = &vb;
    vi.vertexAttributeDescriptionCount = stride ? (hasNormal ? 2 : 1) : 0;
    vi.pVertexAttributeDescriptions = va;
    VkPipelineInputAssemblyStateCreateInfo ia{VK_STRUCTURE_TYPE_PIPELINE_INPUT_ASSEMBLY_STATE_CREATE_INFO};
    ia.topology = VK_PRIMITIVE_TOPOLOGY_TRIANGLE_LIST;
    VkPipelineViewportStateCreateInfo vp{VK_STRUCTURE_TYPE_PIPELINE_VIEWPORT_STATE_CREATE_INFO};
    vp.viewportCount = vp.scissorCount = 1;
    VkPipelineRasterizationStateCreateInfo rs{VK_STRUCTURE_TYPE_PIPELINE_RASTERIZATION_STATE_CREATE_INFO};
    rs.polygonMode = VK_POLYGON_MODE_FILL;
    rs.cullMode = VK_CULL_MODE_NONE;
    rs.lineWidth = 1.f;
    VkPipelineMultisampleStateCreateInfo ms{VK_STRUCTURE_TYPE_PIPELINE_MULTISAMPLE_STATE_CREATE_INFO};
    ms.rasterizationSamples = VK_SAMPLE_COUNT_1_BIT;
    VkPipelineDepthStencilStateCreateInfo ds{VK_STRUCTURE_TYPE_PIPELINE_DEPTH_STENCIL_STATE_CREATE_INFO};
    ds.depthTestEnable = ds.depthWriteEnable = VK_TRUE;
    ds.depthCompareOp = VK_COMPARE_OP_GREATER;  // reverse Z
    VkPipelineColorBlendAttachmentState cba[GBUF_TARGETS]{};
    for (auto& a : cba) a.colorWriteMask = 0xF;
    VkPipelineColorBlendStateCreateInfo cb{VK_STRUCTURE_TYPE_PIPELINE_COLOR_BLEND_STATE_CREATE_INFO};
    cb.attachmentCount = GBUF_TARGETS;
    cb.pAttachments = cba;
    VkDynamicState dyn[2] = {VK_DYNAMIC_STATE_VIEWPORT, VK_DYNAMIC_STATE_SCISSOR};
    VkPipelineDynamicStateCreateInfo dy{VK_STRUCTURE_TYPE_PIPELINE_DYNAMIC_STATE_CREATE_INFO};
    dy.dynamicStateCount = 2;
    dy.pDynamicStates = dyn;
    VkPipelineRenderingCreateInfo pr{VK_STRUCTURE_TYPE_PIPELINE_RENDERING_CREATE_INFO};
    VkFormat fmts[GBUF_TARGETS];
    for (uint32_t i = 0; i < GBUF_TARGETS; i++) fmts[i] = gbufFormat(i);
    pr.colorAttachmentCount = GBUF_TARGETS;
    pr.pColorAttachmentFormats = fmts;
    pr.depthAttachmentFormat = depthFmt;
    VkGraphicsPipelineCreateInfo gp{VK_STRUCTURE_TYPE_GRAPHICS_PIPELINE_CREATE_INFO};
    gp.pNext = &pr;
    gp.stageCount = 2;
    gp.pStages = st;
    gp.pVertexInputState = &vi;
    gp.pInputAssemblyState = &ia;
    gp.pViewportState = &vp;
    gp.pRasterizationState = &rs;
    gp.pMultisampleState = &ms;
    gp.pDepthStencilState = &ds;
    gp.pColorBlendState = &cb;
    gp.pDynamicState = &dy;
    gp.layout = pipeLayout;
    VkPipeline p;
    VK_CHECK(vkCreateGraphicsPipelines(dev, pipeCache, 1, &gp, nullptr, &p));
    vkDestroyShaderModule(dev, vm, nullptr);
    vkDestroyShaderModule(dev, fm, nullptr);
    return p;
}

struct RasterDef { VkPipeline* out; const char* vert; uint32_t stride; bool normal; const char* frag; };
const RasterDef RASTER[] = {
    {&rasterTerrain, "raster_terrain.vert.spv", 24, true, "raster.frag.spv"},
    {&rasterGrass, "raster_grass.vert.spv", 12, false, "raster.frag.spv"},
    {&rasterRock, "raster_rock.vert.spv", 24, true, "raster.frag.spv"},
    {&rasterModel, "raster_model.vert.spv", 0, false, "raster_model.frag.spv"},
    {&rasterImpostor, "raster_impostor.vert.spv", 0, false, "raster_impostor.frag.spv"},
};
constexpr uint32_t RASTER_COUNT = sizeof(RASTER) / sizeof(RASTER[0]);
}  // namespace

void createPipelineLayout() {
    std::vector<VkDescriptorSetLayoutBinding> b;
    auto add = [&](uint32_t i, VkDescriptorType t, uint32_t n = 1) { b.push_back({i, t, n, VK_SHADER_STAGE_ALL, nullptr}); };
    add(0, VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR);
    add(1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER);
    for (uint32_t i = 2; i <= 9; i++) add(i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    for (uint32_t i = 10; i < 10 + I_COUNT; i++) add(i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    add(25, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    add(26, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    add(27, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    add(28, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    for (uint32_t i = 29; i <= 31; i++) add(i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    add(32, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, MAX_TEXTURES);  // model textures (model.glsl)
    for (uint32_t i = 33; i <= 36; i++) add(i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);  // meshes, materials, raster xf, zones
    for (uint32_t i = 37; i < 37 + RR_COUNT; i++) add(i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);  // DLSS-RR inputs + output
    add(44, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);                                               // impostor instances
    add(45, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);                                       // impostor atlas (albedo, normal+depth)
    add(46, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);
    add(47, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);                                                // same, written by the bake
    add(48, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    add(49, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);                                               // bake volumes, probes
    add(50, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);
    add(51, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);                                               // probe filter scratch
    add(52, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);                                                // wind map ping-pong
    add(53, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);
    for (uint32_t i = 0; i < X_COUNT; i++) add(X_BINDING + i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);  // more screen images
    for (uint32_t i = 80; i <= 82; i++) add(i, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);  // triangle lights, light tree, blue noise
    add(83, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);                                      // post.glsl: exposure
    add(87, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);                                      // light alias table
    add(84, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);                              // HDR scene
    add(85, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);                              // bloom chain
    add(86, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 8);                                    // bloom mips
    add(88, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);                              // UI (HDR present)
    add(89, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);                                       // composed HDR frame
    for (uint32_t i = 90; i <= 92; i++) add(i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);  // world map: height, paint, water
    for (uint32_t i = 93; i <= 96; i++) add(i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);           // world sim: state, water flux, snow flux, ground
    add(97, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);                                               // crater bins
    add(98, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);                                               // radiance cache
    for (uint32_t i = 99; i <= 101; i++) add(i, VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER);  // ground virtual texture
    add(102, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);                                              // its tile requests
    for (uint32_t i = 103; i <= 105; i++) add(i, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE);          // vtgen.comp outputs
    add(106, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER);                                              // occlusion grid
    VkDescriptorSetLayoutCreateInfo lci{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    lci.bindingCount = (uint32_t)b.size();
    lci.pBindings = b.data();
    VK_CHECK(vkCreateDescriptorSetLayout(dev, &lci, nullptr, &setLayout));

    std::vector<VkDescriptorPoolSize> ps;  // exactly what the layout holds
    for (auto& x : b) {
        auto it = std::find_if(ps.begin(), ps.end(), [&](auto& p) { return p.type == x.descriptorType; });
        if (it == ps.end()) ps.push_back({x.descriptorType, x.descriptorCount});
        else it->descriptorCount += x.descriptorCount;
    }
    VkDescriptorPoolCreateInfo pci{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    pci.maxSets = 1;
    pci.poolSizeCount = (uint32_t)ps.size();
    pci.pPoolSizes = ps.data();
    VK_CHECK(vkCreateDescriptorPool(dev, &pci, nullptr, &descPool));
    VkDescriptorSetAllocateInfo dai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dai.descriptorPool = descPool;
    dai.descriptorSetCount = 1;
    dai.pSetLayouts = &setLayout;
    VK_CHECK(vkAllocateDescriptorSets(dev, &dai, &descSet));

    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, PUSH_CONSTANT_BYTES};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1;
    pli.pSetLayouts = &setLayout;
    pli.pushConstantRangeCount = 1;
    pli.pPushConstantRanges = &pcr;
    VK_CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &pipeLayout));
    loadPipelineCache();
}

void createPipelines() {
    PROF_FUNC();
    // one pipeline per job: the driver compiles them side by side (cache hits are near free either way)
    jobs::parallelFor(P_COUNT + RASTER_COUNT, 1, [](uint32_t a, uint32_t b) {
        for (uint32_t i = a; i < b; i++) {
            if (i < P_COUNT) pipes[i] = mkComputePipeline(PIPE_NAMES[i]);
            else {
                const RasterDef& r = RASTER[i - P_COUNT];
                *r.out = mkRasterPipeline(r.vert, r.stride, r.normal, r.frag);
            }
        }
    });
}

void savePipelineCache() {
    size_t n = 0;
    vkGetPipelineCacheData(dev, pipeCache, &n, nullptr);
    std::vector<char> blob(n);
    if (!n || vkGetPipelineCacheData(dev, pipeCache, &n, blob.data()) != VK_SUCCESS) return;
    std::ofstream f(cachePath(), std::ios::binary);
    f.write((const char*)&CACHE_MAGIC, 4);
    f.write((const char*)&currentHash, 8);
    f.write(blob.data(), (std::streamsize)n);
}

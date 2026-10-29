#include "render/ground_vt.h"
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include "app/settings.h"
#include "gpu/memory.h"
#include "gpu/offload.h"
#include "gpu/vk.h"
#include "render/frame_ubo.h"
#include "scene/camera.h"
#include "scene/scene_gpu.h"

Image vtImgA, vtImgB, vtImgH;

Buffer vtTileBuf;

// tx, tz, ver: what the slot holds; busy: a tile for it is being generated on the integrated GPU
static struct VtSlot { int tx = INT32_MIN, tz = INT32_MIN; uint32_t ver = 0; bool busy = false; } vtSlots[VT_L][VT_T * VT_T];

uint32_t vtVersion = 1, vtQueued = 0;

// ---- generation on the integrated GPU (gpu/offload.h): the same vtgen.comp writes into the iGPU's own copy of the
// clipmap, the finished tiles are copied to host memory, and the main GPU only copies them into its images. A job
// takes a frame or more; until it lands its tiles count as missing (the coarser level shows), as before.
Buffer vtUpload;                       // main GPU staging: the last finished job's tiles (A, B, H per tile)
std::vector<VtTileReq> vtUploadTiles;  // copied into the clipmap by the "ground VT" pass this frame
static bool vtOff = false;
static igpu::Pipe vtPipe;
static igpu::Img vtGA, vtGB, vtGH;
static igpu::Buf vtGUbo, vtGTiles, vtGOut;
static std::vector<VtTileReq> vtJob;
static std::vector<uint32_t> vtJobVer;
constexpr uint32_t VT_JOB_MAX = 256;
constexpr size_t VT_TILE_BYTES = VT_TILE * VT_TILE * (4 + 4 + 2);

static void vtOffloadInit() {
    vtOff = true;
    vtPipe = igpu::mkPipe("vtgen.comp", {{1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER}, {102, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER},
                                    {103, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE}, {104, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE},
                                    {105, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE}});
    const VkImageUsageFlags use = VK_IMAGE_USAGE_STORAGE_BIT | VK_IMAGE_USAGE_TRANSFER_SRC_BIT;
    vtGA = igpu::mkImageArray(VT_N, VT_N, VT_L, VK_FORMAT_R8G8B8A8_UNORM, use);
    vtGB = igpu::mkImageArray(VT_N, VT_N, VT_L, VK_FORMAT_R8G8B8A8_UNORM, use);
    vtGH = igpu::mkImageArray(VT_N, VT_N, VT_L, VK_FORMAT_R16_UNORM, use);
    vtGUbo = igpu::mkBuf(sizeof(UBO), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT);
    vtGTiles = igpu::mkBuf(VT_JOB_MAX * sizeof(VtTileReq), VK_BUFFER_USAGE_STORAGE_BUFFER_BIT);
    vtGOut = igpu::mkBuf(VT_JOB_MAX * VT_TILE_BYTES, VK_BUFFER_USAGE_TRANSFER_DST_BIT);
    igpu::writeBuf(vtPipe, 1, VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER, vtGUbo);
    igpu::writeBuf(vtPipe, 102, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, vtGTiles);
    igpu::writeImg(vtPipe, 103, vtGA);
    igpu::writeImg(vtPipe, 104, vtGB);
    igpu::writeImg(vtPipe, 105, vtGH);
    vtUpload = mkBuf(VT_JOB_MAX * VT_TILE_BYTES, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
                     VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
    VkCommandBuffer cb = igpu::begin();  // the images live in GENERAL, like the main GPU's
    VkImageMemoryBarrier b[3];
    const igpu::Img* ims[3] = {&vtGA, &vtGB, &vtGH};
    for (int i = 0; i < 3; i++) {
        b[i] = {VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER};
        b[i].oldLayout = VK_IMAGE_LAYOUT_UNDEFINED;
        b[i].newLayout = VK_IMAGE_LAYOUT_GENERAL;
        b[i].srcQueueFamilyIndex = b[i].dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED;
        b[i].image = ims[i]->img;
        b[i].subresourceRange = {VK_IMAGE_ASPECT_COLOR_BIT, 0, 1, 0, (uint32_t)VT_L};
        b[i].dstAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    }
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TOP_OF_PIPE_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, 0, 0, nullptr, 0, nullptr, 3, b);
    igpu::submit();
}

// the tile's texels in a job's output: A (rgba8), B (rgba8), H (r16), each tightly packed
static size_t vtTileOffset(uint32_t i, int which) {
    return i * VT_TILE_BYTES + (which == 0 ? 0 : which == 1 ? VT_TILE * VT_TILE * 4 : VT_TILE * VT_TILE * 8);
}

static void vtSubmitJob() {
    memcpy(vtGTiles.map, vtJob.data(), vtJob.size() * sizeof(VtTileReq));
    memcpy(vtGUbo.map, uboBuf.map, sizeof(UBO));  // last frame's: the generator inputs barely change frame to frame
    VkCommandBuffer cb = igpu::begin();
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, vtPipe.p);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, vtPipe.layout, 0, 1, &vtPipe.set, 0, nullptr);
    uint32_t pc[4] = {(uint32_t)vtJob.size(), vtVersion, 0, 0};
    vkCmdPushConstants(cb, vtPipe.layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, 16, pc);
    vkCmdDispatch(cb, (VT_TILE / 16) * (VT_TILE / 16), (uint32_t)vtJob.size(), 1);
    VkMemoryBarrier m{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_TRANSFER_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT, 0, 1, &m, 0, nullptr, 0, nullptr);
    std::vector<VkBufferImageCopy> c[3];
    for (uint32_t i = 0; i < vtJob.size(); i++) {
        const VtTileReq& t = vtJob[i];
        for (int k = 0; k < 3; k++) {
            VkBufferImageCopy r{};
            r.bufferOffset = vtTileOffset(i, k);
            r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, t.level, 1};
            r.imageOffset = {(int32_t)(t.slot & 0xff) * VT_TILE, (int32_t)(t.slot >> 8) * VT_TILE, 0};
            r.imageExtent = {VT_TILE, VT_TILE, 1};
            c[k].push_back(r);
        }
    }
    const igpu::Img* ims[3] = {&vtGA, &vtGB, &vtGH};
    for (int k = 0; k < 3; k++)
        vkCmdCopyImageToBuffer(cb, ims[k]->img, VK_IMAGE_LAYOUT_GENERAL, vtGOut.b, (uint32_t)c[k].size(), c[k].data());
    VkMemoryBarrier h{VK_STRUCTURE_TYPE_MEMORY_BARRIER, nullptr, VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT};
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_HOST_BIT, 0, 1, &h, 0, nullptr, 0, nullptr);
    igpu::submit();
}

void recordGroundVTUpload(VkCommandBuffer cb) {
    std::vector<VkBufferImageCopy> c[3];
    for (uint32_t i = 0; i < vtUploadTiles.size(); i++) {
        const VtTileReq& t = vtUploadTiles[i];
        for (int k = 0; k < 3; k++) {
            VkBufferImageCopy r{};
            r.bufferOffset = vtTileOffset(i, k);
            r.imageSubresource = {VK_IMAGE_ASPECT_COLOR_BIT, 0, t.level, 1};
            r.imageOffset = {(int32_t)(t.slot & 0xff) * VT_TILE, (int32_t)(t.slot >> 8) * VT_TILE, 0};
            r.imageExtent = {VT_TILE, VT_TILE, 1};
            c[k].push_back(r);
        }
    }
    Image* ims[3] = {&vtImgA, &vtImgB, &vtImgH};
    for (int k = 0; k < 3; k++)
        if (!c[k].empty()) vkCmdCopyBufferToImage(cb, vtUpload.b, ims[k]->img, VK_IMAGE_LAYOUT_GENERAL, (uint32_t)c[k].size(), c[k].data());
}

static float vtGenKey[2] = {-1, -1};  // generator inputs: pebbles, warp (changing them regenerates everything)

// Ground virtual texture: queue the tiles each level's window needs (coarse levels and nearby tiles first), and
// work out per level the rectangle that is fully resident for the shaders. Runs every frame before the UBO fill;
// the tiles queued here are generated by the "ground VT" pass ahead of all shading in the same frame.
void updateGroundVT(float (*rect)[4]) {
    if (S.fractalPebbles != vtGenKey[0] || S.groundWarp != vtGenKey[1]) {
        vtGenKey[0] = S.fractalPebbles; vtGenKey[1] = S.groundWarp;
        vtVersion++;  // out of date: still shown (and resident) until regenerated
    }
    if (!vtOff && igpu::ready()) vtOffloadInit();
    vtUploadTiles.clear();
    bool canQueue = true;
    if (vtOff) {
        canQueue = !igpu::busy();
        if (canQueue && !vtJob.empty()) {  // the last job is done: hand its tiles to the main GPU, now resident
            memcpy(vtUpload.map, vtGOut.map, vtJob.size() * VT_TILE_BYTES);
            for (size_t i = 0; i < vtJob.size(); i++) {
                const VtTileReq& t = vtJob[i];
                VtSlot& sl = vtSlots[t.level][(t.slot >> 8) * VT_T + (t.slot & 0xff)];
                sl = {t.tx, t.tz, vtJobVer[i], false};
            }
            vtUploadTiles.swap(vtJob);
            vtJob.clear();
            vtJobVer.clear();
        }
        canQueue = canQueue && ((const UBO*)uboBuf.map)->vtP[0] > 0.f;  // a frame's UBO exists to copy
    }
    VtTileReq* out = (VtTileReq*)vtTileBuf.map;
    uint32_t n = 0, budget = (uint32_t)std::clamp(S.vtBudget, 1, vtOff ? (int)VT_JOB_MAX : 4096);
    bool generate = S.groundMode == 3;
    for (int pass = 0; pass < 2; pass++)  // missing tiles, then out-of-date ones
        for (int l = VT_L - 1; l >= 0; l--) {
            double tw = VT_TEXEL0 * (double)(1 << l) * VT_TILE;  // tile width (m)
            int cx = (int)floor(cam.pos.x / tw + 0.5) - VT_T / 2, cz = (int)floor(cam.pos.z / tw + 0.5) - VT_T / 2;
            // nearest first
            int order[VT_T * VT_T];
            for (int i = 0; i < VT_T * VT_T; i++) order[i] = i;
            auto d2 = [&](int i) { float dx = (i % VT_T) + 0.5f - VT_T / 2.f, dz = (i / VT_T) + 0.5f - VT_T / 2.f; return dx * dx + dz * dz; };
            std::sort(order, order + VT_T * VT_T, [&](int a, int b) { return d2(a) < d2(b); });
            for (int oi = 0; oi < VT_T * VT_T && generate && canQueue && n < budget; oi++) {
                int tx = cx + order[oi] % VT_T, tz = cz + order[oi] / VT_T;
                uint32_t sx = (uint32_t)(tx & (VT_T - 1)), sz = (uint32_t)(tz & (VT_T - 1));
                VtSlot& sl = vtSlots[l][sz * VT_T + sx];
                bool missing = sl.tx != tx || sl.tz != tz;
                if (sl.busy || (pass == 0 ? !missing : (missing || sl.ver == vtVersion))) continue;
                VtTileReq r{(uint32_t)l, sx | sz << 8, tx, tz};
                n++;
                if (vtOff) {
                    vtJob.push_back(r);
                    vtJobVer.push_back(vtVersion);
                    sl.busy = true;
                } else {
                    out[n - 1] = r;
                    sl = {tx, tz, vtVersion};
                }
            }
            // resident rectangle: start with the window, cut sides until no missing tile is left inside
            int x0 = cx, x1 = cx + VT_T, z0 = cz, z1 = cz + VT_T;  // [x0, x1) tiles
            for (bool again = true; again && x0 < x1 && z0 < z1;) {
                again = false;
                for (int tz = z0; tz < z1 && !again; tz++)
                    for (int tx = x0; tx < x1 && !again; tx++) {
                        const VtSlot& sl = vtSlots[l][(tz & (VT_T - 1)) * VT_T + (tx & (VT_T - 1))];
                        if (sl.tx == tx && sl.tz == tz) continue;
                        // cut the side nearest this tile
                        int dl = tx - x0, dr = x1 - 1 - tx, db = tz - z0, dt = z1 - 1 - tz, m = std::min({dl, dr, db, dt});
                        if (m == dl) x0 = tx + 1; else if (m == dr) x1 = tx; else if (m == db) z0 = tz + 1; else z1 = tz;
                        again = true;
                    }
            }
            float tex = (float)(VT_TEXEL0 * (double)(1 << l));
            if (x0 < x1 && z0 < z1) {
                rect[l][0] = (float)(x0 * tw) + tex; rect[l][1] = (float)(z0 * tw) + tex;
                rect[l][2] = (float)(x1 * tw) - tex; rect[l][3] = (float)(z1 * tw) - tex;
            } else rect[l][0] = rect[l][1] = 1e9f, rect[l][2] = rect[l][3] = -1e9f;
        }
    if (vtOff) {
        if (!vtJob.empty() && canQueue) vtSubmitJob();
        n = 0;
    }
    vtQueued = n;
}

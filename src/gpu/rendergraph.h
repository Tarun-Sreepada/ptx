// Minimal frame graph. Passes are declared in a valid execution order with the resources they read and
// write (logical resources: a name stands for any group of images / buffers used together). From that the
// graph derives
//   - one merged sync2 memory barrier in front of each pass (RAW / WAW / WAR on the same queue)
//   - queue batches + timeline-semaphore waits when a pass consumes something another queue produced,
//     so independent work (e.g. BVH build on async compute vs. raster G-buffer) overlaps
//   - GPU timestamps around every pass (timings for the next frame's UI, any queue)
// All screen images stay in VK_IMAGE_LAYOUT_GENERAL, so global memory barriers are sufficient; passes that
// need layout transitions (swapchain) do them inside their record function.
#pragma once
#include <vulkan/vulkan.h>

#include <cstdint>
#include <functional>
#include <string>
#include <vector>

namespace rg {
enum Queue : uint8_t { GFX, COMPUTE, QUEUE_COUNT };

struct QueueInfo { VkQueue queue = VK_NULL_HANDLE; uint32_t family = 0; };

// VK_EXT_debug_utils labels around every pass (null = off)
void setDebugLabels(PFN_vkCmdBeginDebugUtilsLabelEXT begin, PFN_vkCmdEndDebugUtilsLabelEXT end);

struct PassTiming { std::string name; Queue queue; float startMs, ms; };

class Graph {
public:
    // phys: for the Tracy GPU contexts (one per queue; only with -DPTX_PROFILE=ON)
    void init(VkPhysicalDevice phys, VkDevice dev, const QueueInfo queues[QUEUE_COUNT], float timestampPeriodNs);
    void destroy();

    // ---- per frame
    void begin();  // after the frame fence: resets command buffers and queries, reads last frame's timings
    uint32_t resource(const char* name);  // stable id per name
    struct PassBuilder {
        Graph* g; uint32_t idx;
        PassBuilder& read(uint32_t res, VkPipelineStageFlags2 stage, VkAccessFlags2 access);
        PassBuilder& write(uint32_t res, VkPipelineStageFlags2 stage, VkAccessFlags2 access);
    };
    PassBuilder add(const char* name, Queue q, std::function<void(VkCommandBuffer)> record);
    // Builds batches, records and submits. firstCbs run at the start of the first GFX batch (e.g. uploads).
    // acquire is waited on (at waitStage) by the batch containing the pass named presentPass.
    void execute(const std::vector<VkCommandBuffer>& firstCbs, VkSemaphore acquire, VkPipelineStageFlags2 waitStage, const char* presentPass,
                 VkSemaphore renderDone, VkFence fence);

    const std::vector<PassTiming>& timings() const { return lastTimings; }
    float frameMs() const { return lastFrameMs; }
    float passMs(const char* prefix) const;  // sum of passes whose name starts with prefix
    bool asyncCompute() const { return queues[COMPUTE].queue != queues[GFX].queue; }

private:
    struct Use { uint32_t res; VkPipelineStageFlags2 stage; VkAccessFlags2 access; bool write; };
    struct Pass { std::string name; Queue q; std::vector<Use> uses; std::function<void(VkCommandBuffer)> record; };
    struct Batch {
        Queue q; VkCommandBuffer cb;
        std::vector<VkSemaphoreSubmitInfo> waits;
        uint64_t signal;
        bool present = false;
    };
    // per resource: last writer and readers since, each as (queue, batch timeline value, stages, access)
    struct Access { Queue q; uint64_t value; VkPipelineStageFlags2 stage; VkAccessFlags2 access; };
    struct ResState { bool written = false; Access w{}; std::vector<Access> readers; };

    VkCommandBuffer nextCb(Queue q);

    VkDevice dev = VK_NULL_HANDLE;
    QueueInfo queues[QUEUE_COUNT];
    VkCommandPool pools[QUEUE_COUNT] = {};
    std::vector<VkCommandBuffer> cbs[QUEUE_COUNT];
    uint32_t cbUsed[QUEUE_COUNT] = {};
    VkSemaphore timeline[QUEUE_COUNT] = {};
    uint64_t timelineValue[QUEUE_COUNT] = {};
    VkQueryPool queries = VK_NULL_HANDLE;
    static constexpr uint32_t MAX_PASSES = 64;
    float tsPeriod = 1.f;
    void* tracyCtx[QUEUE_COUNT] = {};  // tracy::VkCtx*, null without Tracy

    std::vector<std::string> resNames;
    std::vector<Pass> passes;
    std::vector<std::pair<std::string, Queue>> lastPasses;  // what the queries in flight belong to
    std::vector<PassTiming> lastTimings;
    float lastFrameMs = 0;
};
}  // namespace rg

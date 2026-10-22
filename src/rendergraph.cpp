#include "rendergraph.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#define RG_CHECK(x)                                                                    \
    do {                                                                               \
        VkResult r_ = (x);                                                             \
        if (r_ != VK_SUCCESS) {                                                        \
            fprintf(stderr, "Vulkan error %d at %s:%d\n", (int)r_, __FILE__, __LINE__); \
            abort();                                                                   \
        }                                                                              \
    } while (0)

namespace rg {
static PFN_vkCmdBeginDebugUtilsLabelEXT labelBegin = nullptr;
static PFN_vkCmdEndDebugUtilsLabelEXT labelEnd = nullptr;
void setDebugLabels(PFN_vkCmdBeginDebugUtilsLabelEXT b, PFN_vkCmdEndDebugUtilsLabelEXT e) { labelBegin = b; labelEnd = e; }

namespace {
constexpr VkAccessFlags2 WRITE_ACCESS = VK_ACCESS_2_SHADER_WRITE_BIT | VK_ACCESS_2_SHADER_STORAGE_WRITE_BIT | VK_ACCESS_2_COLOR_ATTACHMENT_WRITE_BIT |
                                        VK_ACCESS_2_DEPTH_STENCIL_ATTACHMENT_WRITE_BIT | VK_ACCESS_2_TRANSFER_WRITE_BIT |
                                        VK_ACCESS_2_HOST_WRITE_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT |
                                        VK_ACCESS_2_ACCELERATION_STRUCTURE_WRITE_BIT_KHR;
}

void Graph::init(VkDevice d, const QueueInfo q[QUEUE_COUNT], float period) {
    dev = d;
    tsPeriod = period;
    for (int i = 0; i < QUEUE_COUNT; i++) {
        queues[i] = q[i];
        VkCommandPoolCreateInfo ci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        ci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        ci.queueFamilyIndex = q[i].family;
        RG_CHECK(vkCreateCommandPool(dev, &ci, nullptr, &pools[i]));
        VkSemaphoreTypeCreateInfo ti{VK_STRUCTURE_TYPE_SEMAPHORE_TYPE_CREATE_INFO};
        ti.semaphoreType = VK_SEMAPHORE_TYPE_TIMELINE;
        VkSemaphoreCreateInfo si{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        si.pNext = &ti;
        RG_CHECK(vkCreateSemaphore(dev, &si, nullptr, &timeline[i]));
    }
    VkQueryPoolCreateInfo qi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qi.queryType = VK_QUERY_TYPE_TIMESTAMP;
    qi.queryCount = MAX_PASSES * 2;
    RG_CHECK(vkCreateQueryPool(dev, &qi, nullptr, &queries));
    vkResetQueryPool(dev, queries, 0, MAX_PASSES * 2);
}

void Graph::destroy() {
    for (int i = 0; i < QUEUE_COUNT; i++) {
        vkDestroyCommandPool(dev, pools[i], nullptr);
        vkDestroySemaphore(dev, timeline[i], nullptr);
    }
    vkDestroyQueryPool(dev, queries, nullptr);
}

void Graph::begin() {
    // last frame's timings (the caller waited on its fence, which covers every queue)
    if (!lastPasses.empty()) {
        std::vector<uint64_t> t(lastPasses.size() * 2);
        if (vkGetQueryPoolResults(dev, queries, 0, (uint32_t)t.size(), t.size() * 8, t.data(), 8, VK_QUERY_RESULT_64_BIT) == VK_SUCCESS) {
            uint64_t t0 = UINT64_MAX, t1 = 0;
            for (size_t i = 0; i < lastPasses.size(); i++) { t0 = std::min(t0, t[2 * i]); t1 = std::max(t1, t[2 * i + 1]); }
            float ms = tsPeriod * 1e-6f;
            lastTimings.clear();
            for (size_t i = 0; i < lastPasses.size(); i++)
                lastTimings.push_back({lastPasses[i].first, lastPasses[i].second, (t[2 * i] - t0) * ms, (t[2 * i + 1] - t[2 * i]) * ms});
            lastFrameMs = (t1 - t0) * ms;
        }
        vkResetQueryPool(dev, queries, 0, (uint32_t)lastPasses.size() * 2);
    }
    for (auto& c : cbUsed) c = 0;
    passes.clear();
}

uint32_t Graph::resource(const char* name) {
    for (uint32_t i = 0; i < resNames.size(); i++)
        if (resNames[i] == name) return i;
    resNames.push_back(name);
    return (uint32_t)resNames.size() - 1;
}

Graph::PassBuilder Graph::add(const char* name, Queue q, std::function<void(VkCommandBuffer)> record) {
    if (!asyncCompute()) q = GFX;
    passes.push_back({name, q, {}, std::move(record)});
    return {this, (uint32_t)passes.size() - 1};
}

Graph::PassBuilder& Graph::PassBuilder::read(uint32_t res, VkPipelineStageFlags2 stage, VkAccessFlags2 access) {
    g->passes[idx].uses.push_back({res, stage, access, false});
    return *this;
}
Graph::PassBuilder& Graph::PassBuilder::write(uint32_t res, VkPipelineStageFlags2 stage, VkAccessFlags2 access) {
    g->passes[idx].uses.push_back({res, stage, access, true});
    return *this;
}

VkCommandBuffer Graph::nextCb(Queue q) {
    if (cbUsed[q] == cbs[q].size()) {
        VkCommandBufferAllocateInfo ai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        ai.commandPool = pools[q];
        ai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        ai.commandBufferCount = 1;
        VkCommandBuffer cb;
        RG_CHECK(vkAllocateCommandBuffers(dev, &ai, &cb));
        cbs[q].push_back(cb);
    }
    VkCommandBuffer cb = cbs[q][cbUsed[q]++];
    vkResetCommandBuffer(cb, 0);
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    vkBeginCommandBuffer(cb, &bi);
    return cb;
}

void Graph::execute(const std::vector<VkCommandBuffer>& firstCbs, VkSemaphore acquire, VkPipelineStageFlags2 waitStage, const char* presentPass,
                    VkSemaphore renderDone, VkFence fence) {
    std::vector<ResState> state(resNames.size());
    std::vector<Batch> batches;
    int open[QUEUE_COUNT] = {-1, -1};  // index of the batch currently recording on each queue
    auto closeBatch = [&](Queue q) {
        if (open[q] < 0) return;
        vkEndCommandBuffer(batches[open[q]].cb);
        open[q] = -1;
    };
    auto openBatch = [&](Queue q) {
        Batch b;
        b.q = q;
        b.cb = nextCb(q);
        b.signal = ++timelineValue[q];
        batches.push_back(b);
        open[q] = (int)batches.size() - 1;
    };

    lastPasses.clear();
    uint32_t qi = 0;
    for (auto& p : passes) {
        // cross-queue producers this pass must wait for; a wait splits the batch so earlier work isn't held back
        std::vector<VkSemaphoreSubmitInfo> waits;
        auto needWait = [&](const Access& a, VkPipelineStageFlags2 stage) {
            if (a.q == p.q) return;
            for (auto& w : waits)
                if (w.semaphore == timeline[a.q]) { w.value = std::max(w.value, a.value); w.stageMask |= stage; return; }
            VkSemaphoreSubmitInfo w{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
            w.semaphore = timeline[a.q];
            w.value = a.value;
            w.stageMask = stage;
            waits.push_back(w);
        };
        VkMemoryBarrier2 mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER_2};
        for (auto& u : p.uses) {
            ResState& s = state[u.res];
            if (s.written) {  // RAW / WAW
                if (s.w.q == p.q) {
                    mb.srcStageMask |= s.w.stage;
                    mb.srcAccessMask |= s.w.access & WRITE_ACCESS;
                    mb.dstStageMask |= u.stage;
                    mb.dstAccessMask |= u.access;
                } else needWait(s.w, u.stage);
            }
            if (u.write)  // WAR: execution dependency on everyone who read the old contents
                for (auto& r : s.readers) {
                    if (r.q == p.q) { mb.srcStageMask |= r.stage; mb.dstStageMask |= u.stage; }
                    else needWait(r, u.stage);
                }
        }
        // a wait on a value the open batch of the other queue hasn't submitted yet is still fine (timeline
        // wait-before-signal), but that batch must end first so its value gets signalled
        for (auto& w : waits)
            for (int oq = 0; oq < QUEUE_COUNT; oq++)
                if (w.semaphore == timeline[oq] && open[oq] >= 0 && batches[open[oq]].signal <= w.value) closeBatch((Queue)oq);
        if (open[p.q] < 0 || !waits.empty()) {
            closeBatch(p.q);
            openBatch(p.q);
            batches[open[p.q]].waits = waits;
        }
        Batch& b = batches[open[p.q]];
        if (presentPass && p.name == presentPass) b.present = true;
        VkCommandBuffer cb = b.cb;
        if (mb.srcStageMask || mb.dstStageMask) {
            if (!mb.srcStageMask) mb.srcStageMask = VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT;
            VkDependencyInfo di{VK_STRUCTURE_TYPE_DEPENDENCY_INFO};
            di.memoryBarrierCount = 1;
            di.pMemoryBarriers = &mb;
            vkCmdPipelineBarrier2(cb, &di);
        }
        bool timed = qi + 2 <= MAX_PASSES * 2;
        if (timed) vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_TOP_OF_PIPE_BIT, queries, qi);
        if (labelBegin) {
            VkDebugUtilsLabelEXT l{VK_STRUCTURE_TYPE_DEBUG_UTILS_LABEL_EXT};
            l.pLabelName = p.name.c_str();
            labelBegin(cb, &l);
        }
        p.record(cb);
        if (labelEnd) labelEnd(cb);
        if (timed) {
            vkCmdWriteTimestamp2(cb, VK_PIPELINE_STAGE_2_BOTTOM_OF_PIPE_BIT, queries, qi + 1);
            lastPasses.push_back({p.name, p.q});
            qi += 2;
        }
        // update resource state
        for (auto& u : p.uses) {
            ResState& s = state[u.res];
            Access a{p.q, b.signal, u.stage, u.access};
            if (u.write) { s.written = true; s.w = a; s.readers.clear(); }
            else s.readers.push_back(a);
        }
    }
    closeBatch(GFX);
    closeBatch(COMPUTE);

    // the last GFX batch carries the fence: make it wait for every other queue's final work
    int last = -1;
    for (int i = 0; i < (int)batches.size(); i++) if (batches[i].q == GFX) last = i;
    if (last < 0) { fprintf(stderr, "rendergraph: no graphics work\n"); abort(); }
    for (int i = 0; i < (int)batches.size(); i++)
        if (batches[i].q != GFX) {
            VkSemaphoreSubmitInfo w{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
            w.semaphore = timeline[batches[i].q];
            w.value = batches[i].signal;
            w.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            batches[last].waits.push_back(w);
        }

    bool first = true;
    for (int i = 0; i < (int)batches.size(); i++) {
        Batch& b = batches[i];
        std::vector<VkCommandBufferSubmitInfo> cbi;
        if (b.q == GFX && first) {
            for (auto c : firstCbs) cbi.push_back({VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, nullptr, c, 0});
            first = false;
        }
        cbi.push_back({VK_STRUCTURE_TYPE_COMMAND_BUFFER_SUBMIT_INFO, nullptr, b.cb, 0});
        std::vector<VkSemaphoreSubmitInfo> waits = b.waits, signals;
        if (b.present && acquire) {
            VkSemaphoreSubmitInfo w{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
            w.semaphore = acquire;
            w.stageMask = waitStage;
            waits.push_back(w);
        }
        VkSemaphoreSubmitInfo s{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
        s.semaphore = timeline[b.q];
        s.value = b.signal;
        s.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
        signals.push_back(s);
        if (i == last && renderDone) {
            VkSemaphoreSubmitInfo r{VK_STRUCTURE_TYPE_SEMAPHORE_SUBMIT_INFO};
            r.semaphore = renderDone;
            r.stageMask = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT;
            signals.push_back(r);
        }
        VkSubmitInfo2 si{VK_STRUCTURE_TYPE_SUBMIT_INFO_2};
        si.waitSemaphoreInfoCount = (uint32_t)waits.size();
        si.pWaitSemaphoreInfos = waits.data();
        si.commandBufferInfoCount = (uint32_t)cbi.size();
        si.pCommandBufferInfos = cbi.data();
        si.signalSemaphoreInfoCount = (uint32_t)signals.size();
        si.pSignalSemaphoreInfos = signals.data();
        RG_CHECK(vkQueueSubmit2(queues[b.q].queue, 1, &si, i == last ? fence : VK_NULL_HANDLE));
    }
}

float Graph::passMs(const char* prefix) const {
    float s = 0;
    size_t n = strlen(prefix);
    for (auto& t : lastTimings)
        if (!t.name.compare(0, n, prefix)) s += t.ms;
    return s;
}
}  // namespace rg

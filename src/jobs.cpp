#include "jobs.h"

#include <TaskScheduler.h>
#include "prof.h"

#include <memory>

namespace jobs {
namespace {
std::unique_ptr<enki::TaskScheduler> ts;
}

void init(uint32_t threads) {
    ts = std::make_unique<enki::TaskScheduler>();
    enki::TaskSchedulerConfig cfg;
    cfg.numTaskThreadsToCreate = threads > 1 ? threads - 1 : 0;
    cfg.profilerCallbacks.threadStart = [](uint32_t i) {
        static char names[64][16];
        snprintf(names[i % 64], 16, "job %u", i);
        prof::setThreadName(names[i % 64]);
    };
    ts->Initialize(cfg);
}

void shutdown() {
    if (ts) ts->WaitforAllAndShutdown();
    ts.reset();
}

enki::TaskScheduler& scheduler() { return *ts; }
uint32_t threadCount() { return ts->GetNumTaskThreads(); }

void parallelFor(uint32_t n, uint32_t minRange, const std::function<void(uint32_t, uint32_t)>& fn) {
    if (n == 0) return;
    if (n <= minRange) { fn(0, n); return; }
    enki::TaskSet task(n, [&](enki::TaskSetPartition r, uint32_t) {
        PROF_ZONE("parallelFor");
        fn(r.start, r.end);
    });
    task.m_MinRange = minRange;
    ts->AddTaskSetToPipe(&task);
    ts->WaitforTask(&task);
}
}  // namespace jobs

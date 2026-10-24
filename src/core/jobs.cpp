#include "core/jobs.h"

#include <TaskScheduler.h>
#include "core/prof.h"

#include <cstdio>
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

struct Async::Task : enki::ITaskSet {
    std::function<void()> fn;
    bool waited = false;
    void ExecuteRange(enki::TaskSetPartition, uint32_t) override {
        PROF_ZONE("async task");
        fn();
    }
};

Async::Async() = default;
Async::Async(std::function<void()> fn) : task(std::make_unique<Task>()) {
    task->fn = std::move(fn);
    ts->AddTaskSetToPipe(task.get());
}
Async::Async(Async&&) noexcept = default;
Async& Async::operator=(Async&& o) noexcept {
    wait();
    task = std::move(o.task);
    return *this;
}
Async::~Async() { wait(); }
void Async::wait() {
    if (!task || task->waited) return;
    ts->WaitforTask(task.get());
    task->waited = true;
}
bool Async::done() const { return !task || task->GetIsComplete(); }
}  // namespace jobs

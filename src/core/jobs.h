// One enkiTS scheduler for the whole process: Jolt's jobs, culling and other parallel CPU work share
// the same worker threads instead of each subsystem spawning its own pool and oversubscribing cores.
#pragma once
#include <cstdint>
#include <functional>
#include <memory>

namespace enki { class TaskScheduler; }

namespace jobs {
void init(uint32_t threads);  // total threads including the main thread
void shutdown();
enki::TaskScheduler& scheduler();
uint32_t threadCount();
// Runs fn(begin, end) over [0, n) in chunks of at least minRange, main thread participates; blocks until done.
void parallelFor(uint32_t n, uint32_t minRange, const std::function<void(uint32_t, uint32_t)>& fn);

// A function running on the pool while the caller does other work (startup: loading, generating, compiling).
// wait() blocks until it finished, running other pool tasks meanwhile. Destroying an Async waits too.
class Async {
public:
    Async();
    explicit Async(std::function<void()> fn);
    Async(Async&&) noexcept;
    Async& operator=(Async&&) noexcept;
    ~Async();
    void wait();
    bool done() const;

private:
    struct Task;
    std::unique_ptr<Task> task;
};
inline Async async(std::function<void()> fn) { return Async(std::move(fn)); }
}  // namespace jobs

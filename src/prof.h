// Profiling: one zone macro feeds three consumers.
//  - Tracy (connect the Tracy profiler to a running ptx, on demand)
//  - NVTX ranges for Nsight Systems (nsys profile --trace=nvtx,vulkan ./ptx); the GPU passes carry Vulkan debug
//    labels (rendergraph.cpp) so Nsight shows them by name on the GPU rows
//  - the in-engine capture (F9 / Profiler window): records N frames of CPU zones on every thread plus the GPU pass
//    timings, shows them as a flame chart and exports Chrome trace JSON (chrome://tracing, ui.perfetto.dev)
#pragma once
#include <atomic>
#include <cstdint>
#include <string>
#include <vector>

#include <tracy/Tracy.hpp>
#if PTX_NVTX
#include <nvtx3/nvToolsExt.h>
#endif

namespace prof {
extern std::atomic<bool> capturing;
uint64_t nowNs();
void setThreadName(const char* name);  // this thread's lane name (capture, Tracy, NVTX)

struct Scope {
    const char* name;
    uint64_t t0;
    bool rec;
    explicit Scope(const char* n);
    ~Scope();
};

struct GpuPass { std::string name; bool async; float startMs, ms; };
void frameBegin();                                  // main thread, top of the frame
void gpuFrame(const std::vector<GpuPass>& passes);  // last frame's GPU passes (they finish a frame late)
void ui(bool* open);                                // the Profiler window
float slowFps();                                    // > 0: the main loop sleeps to this frame rate (slow motion)
void toggleCapture();                               // F9
void setCaptureFrames(int n);
bool hasCapture();
std::string exportTrace();                          // Chrome trace JSON of the last capture -> its path
}  // namespace prof

#define PROF_CAT2(a, b) a##b
#define PROF_CAT(a, b) PROF_CAT2(a, b)
#define PROF_ZONE(name) ZoneScopedN(name); prof::Scope PROF_CAT(prof_scope_, __LINE__)(name)
#define PROF_FUNC() ZoneScoped; prof::Scope PROF_CAT(prof_scope_, __LINE__)(__func__)

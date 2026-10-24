#include "tools/bench.h"
#include "scene/grass.h"
#include "render/targets.h"
#include "gpu/dlss.h"
#include "gpu/context.h"
#include <cstdio>
#include <algorithm>
#include <cmath>
#include <cstddef>
#include "app/frame_state.h"
#include "app/settings.h"
#include "core/math.h"
#include "render/frame_graph.h"
#include "scene/camera.h"

Bench bench;

static int submittedTier = 0, submittedFrame = 0;

int targetFps = 0;          // 0 = uncapped/manual

static int scaleDown = 0, scaleUp = 0;

// ---------------------------------------------------------------- stats / benchmark / scaler
int raysPerPixel(int n, int b) {
    // [primary] + ReSTIR DI visibility + final light shadow + sun shadow, then per path vertex: bounce + sun + light shadow rays,
    // + [ReSTIR GI reconnection visibility]
    return (S.rasterPrimary ? 0 : 1) + 3 + n * b * 3 + (S.restirGI && S.giSpatial ? 1 : 0);
}

void readTimestamps() {
    gpuTotal = graph.frameMs();
    if (gpuTotal <= 0) return;
    float gi = graph.passMs("GI");
    gpuEmaTotal = gpuEmaTotal == 0 ? gpuTotal : gpuEmaTotal * 0.92f + gpuTotal * 0.08f;
    gpuEmaGI = gpuEmaGI == 0 ? gi : gpuEmaGI * 0.92f + gi * 0.08f;
    if (submittedTier > 0 && submittedFrame >= Bench::WARM) bench.samples[submittedTier].push_back(gpuTotal);
}

void startBenchmark() {
    bench.running = true;
    bench.done = false;
    bench.tier = 1;
    bench.frame = 0;
    bench.vt = 0;
    for (auto& s : bench.samples) s.clear();
    bench.saved = S;
    bench.savedCam = cam;
    resetHistory = true;
}

void benchStep() {
    if (!bench.running) { submittedTier = 0; return; }
    if (bench.frame >= Bench::WARM + Bench::MEAS) {
        auto& v = bench.samples[bench.tier];
        if (!v.empty()) {
            float sum = 0;
            for (float x : v) sum += x;
            bench.avg[bench.tier] = sum / v.size();
            std::sort(v.begin(), v.end());
            bench.p95[bench.tier] = v[std::min(v.size() - 1, (size_t)(v.size() * 0.95f))];
        }
        bench.tier++;
        bench.frame = 0;
        resetHistory = true;
        if (bench.tier > 4) {
            bench.running = false;
            bench.done = true;
            S = bench.saved;
            cam = prevCam = bench.savedCam;
            submittedTier = 0;
            return;
        }
    }
    // fixed, repeatable workload: default quality settings, scripted orbit, simulated 60 Hz clock
    Settings keep = S;
    S = Settings{};
    S.exposure = keep.exposure;
    S.rasterPrimary = keep.rasterPrimary;
    S.restirGI = keep.restirGI;
    S.raysFirst = bench.tier;
    double a = bench.vt * 0.25;
    cam.pos = {(float)sin(a) * 16.f, 2.0f + (float)sin(a * 0.7) * 0.8f, (float)cos(a) * 16.f};
    V3 d = normalize(V3{0, 0.5f, 0} - cam.pos);
    cam.yaw = atan2f(d.x, d.z);
    cam.pitch = asinf(d.y);
    simTime = bench.vt;
    bench.vt += 1.0 / 60.0;
    submittedTier = bench.tier;
    submittedFrame = bench.frame++;
}

// best tier whose p95 GPU time fits the frame budget
int benchBestTier(int fps) {
    float budget = 1000.f / fps;
    int best = 0;
    for (int t = 1; t <= 4; t++)
        if (bench.p95[t] > 0 && bench.p95[t] <= budget) best = t;
    return best;
}

void autoScale() {
    if (targetFps <= 0 || bench.running || gpuEmaTotal <= 0) return;
    float budget = 1000.f / targetFps * 0.92f;
    float perRay = gpuEmaGI / std::max(1, S.raysFirst);
    float fixed = gpuEmaTotal - gpuEmaGI;
    int fit = perRay > 0 ? std::clamp((int)floorf((budget - fixed) / perRay), 1, 4) : S.raysFirst;
    if (fit < S.raysFirst) { if (++scaleDown > 10) { S.raysFirst = fit; scaleDown = scaleUp = 0; } } else scaleDown = 0;
    if (fit > S.raysFirst) { if (++scaleUp > 90) { S.raysFirst++; scaleDown = scaleUp = 0; } } else scaleUp = 0;
}

void printBenchmark() {
    printf("%s @ %ux%u -> %ux%u (%s), ~%u blades/frame\n", gpuName, W, H, OW, OH, rrActive ? dlss::modeName(upscaler) : "native", lastBladeCount);
    for (int t = 1; t <= 4; t++)
        printf("  %d ray(s)/bounce (%2d rpp): avg %.2f ms  p95 %.2f ms  -> %.0f fps\n", t, raysPerPixel(t, 2), bench.avg[t], bench.p95[t], 1000.f / bench.p95[t]);
    for (int fps : {60, 40, 30, 25}) printf("  target %d fps -> %d ray(s)/bounce\n", fps, benchBestTier(fps));
}

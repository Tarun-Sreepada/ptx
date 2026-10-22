#include "audio.h"

#define MINIAUDIO_IMPLEMENTATION
#define MA_NO_ENCODING
#define MA_NO_GENERATION
#include <miniaudio.h>
#include <meshoptimizer.h>
#include <phonon.h>
#include "prof.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstring>
#include <functional>
#include <mutex>
#include <random>
#include <thread>
#include <vector>

namespace audio {
namespace {
constexpr int RATE = 48000, FRAME = 1024;  // Steam Audio's convolution is tuned for 1024-sample frames
constexpr int MAX_EMITTERS = 64;
constexpr size_t MAX_SOUNDS = 1024;
constexpr int AMBI_ORDER = 1, AMBI_CH = (AMBI_ORDER + 1) * (AMBI_ORDER + 1);
constexpr float IR_SECONDS = 1.0f;
constexpr int IR_SIZE = (int)(IR_SECONDS * RATE);

// absorption (low/mid/high), scattering, transmission (low/mid/high) — values from Steam Audio's presets
const IPLMaterial kMaterials[MAT_COUNT] = {
    {{0.10f, 0.20f, 0.30f}, 0.05f, {0.100f, 0.050f, 0.030f}},  // generic
    {{0.05f, 0.07f, 0.08f}, 0.05f, {0.015f, 0.002f, 0.001f}},  // concrete
    {{0.11f, 0.07f, 0.06f}, 0.05f, {0.070f, 0.014f, 0.005f}},  // wood
    {{0.06f, 0.03f, 0.02f}, 0.05f, {0.060f, 0.044f, 0.011f}},  // glass
    {{0.20f, 0.07f, 0.06f}, 0.05f, {0.200f, 0.025f, 0.010f}},  // metal
    {{0.60f, 0.70f, 0.80f}, 0.40f, {0.000f, 0.000f, 0.000f}},  // ground / grass / soil
};

struct Sound { std::vector<float> s; };  // mono, RATE; immutable once registered

enum SlotState : int { FREE, PENDING, LIVE, DEAD };
struct Emitter {
    std::atomic<int> state{FREE};
    std::atomic<float> px{0}, py{0}, pz{0}, gain{1};
    std::atomic<bool> stop{false};
    const Sound* sound = nullptr;
    bool loop = false;
    float pitch = 1.f;
    double cursor = 0;  // audio thread
    IPLSource src = nullptr;
    IPLDirectEffect direct = nullptr;
    IPLBinauralEffect binaural = nullptr;
    IPLReflectionEffect reflect = nullptr;
};

struct SubMesh { IPLScene scene = nullptr; IPLStaticMesh mesh = nullptr; uint32_t tris = 0; };

IPLContext ctx;
IPLHRTF hrtf;
IPLEmbreeDevice embree;
IPLScene scene;
IPLSimulator sim;
IPLReflectionMixer mixer;
IPLAmbisonicsDecodeEffect ambiDecode;
IPLSceneType sceneType = IPL_SCENETYPE_DEFAULT;
IPLAudioSettings as{RATE, FRAME};
ma_device device;
bool deviceOk = false, ok = false;

Settings S;
std::vector<Sound> sounds;  // index = sound id; only appended from the main thread before use
Emitter emitters[MAX_EMITTERS];

// listener snapshot shared with the sim and audio threads
std::mutex lisMtx;
IPLCoordinateSpace3 listener{{1, 0, 0}, {0, 1, 0}, {0, 0, -1}, {0, 0, 0}};
IPLCoordinateSpace3 getListener() { std::lock_guard l(lisMtx); return listener; }

// scene edits are queued by the main thread and applied by the simulation thread between runs
std::mutex cmdMtx;
std::vector<std::function<void()>> cmds;
void post(std::function<void()> f) { std::lock_guard l(cmdMtx); cmds.push_back(std::move(f)); }

// sim-thread owned
std::vector<SubMesh> meshes;
struct Inst { IPLInstancedMesh m = nullptr; uint32_t tris = 0; };
std::vector<Inst> instances;
SubMesh terrain;
uint32_t sceneTris = 0;
// main-thread handle allocation
uint32_t nextMesh = 0, nextInst = 0;
std::vector<uint32_t> freeMesh, freeInst;
uint32_t liveMeshes = 0, liveInsts = 0;

std::thread simThread;
std::atomic<bool> quit{false};
std::atomic<float> simMs{0}, dspMs{0}, windGain{0};

// ---------------------------------------------------------------- device thread
float fifo[FRAME * 2 * 2];  // interleaved stereo, at most one block + leftover
int fifoCount = 0;          // frames
IPLAudioBuffer inBuf, dirBuf, binBuf, reflBuf, ambiBuf, mixBuf, decBuf;

void renderBlock(float* outInterleaved) {
    PROF_ZONE("audio block");
    auto t0 = std::chrono::steady_clock::now();
    IPLCoordinateSpace3 L = getListener();
    for (int c = 0; c < 2; c++) std::fill_n(mixBuf.data[c], FRAME, 0.f);
    bool anyRefl = false;
    for (auto& e : emitters) {
        if (e.state.load(std::memory_order_acquire) != LIVE) continue;
        if (e.stop.load()) { e.state.store(DEAD, std::memory_order_release); continue; }
        // source samples (pitch via linear interpolation)
        const auto& s = e.sound->s;
        float g = e.gain.load(), *in = inBuf.data[0];
        bool ended = false;
        for (int i = 0; i < FRAME; i++) {
            if (e.cursor >= s.size() - 1) {
                if (e.loop) e.cursor = fmod(e.cursor, (double)(s.size() - 1));
                else { std::fill(in + i, in + FRAME, 0.f); ended = true; break; }
            }
            size_t k = (size_t)e.cursor;
            float f = (float)(e.cursor - k);
            in[i] = (s[k] + (s[k + 1] - s[k]) * f) * g;
            e.cursor += e.pitch;
        }
        IPLSimulationOutputs out{};
        iplSourceGetOutputs(e.src, (IPLSimulationFlags)(IPL_SIMULATIONFLAGS_DIRECT | IPL_SIMULATIONFLAGS_REFLECTIONS), &out);
        // direct path
        IPLDirectEffectParams& dp = out.direct;
        dp.flags = (IPLDirectEffectFlags)(IPL_DIRECTEFFECTFLAGS_APPLYDISTANCEATTENUATION | IPL_DIRECTEFFECTFLAGS_APPLYAIRABSORPTION |
                                          (S.occlusion ? IPL_DIRECTEFFECTFLAGS_APPLYOCCLUSION : 0) |
                                          (S.transmission ? IPL_DIRECTEFFECTFLAGS_APPLYTRANSMISSION : 0));
        dp.transmissionType = IPL_TRANSMISSIONTYPE_FREQDEPENDENT;
        iplDirectEffectApply(e.direct, &dp, &inBuf, &dirBuf);
        IPLVector3 sp{e.px.load(), e.py.load(), e.pz.load()};
        IPLBinauralEffectParams bp{};
        bp.direction = iplCalculateRelativeDirection(ctx, sp, L.origin, L.ahead, L.up);
        bp.interpolation = IPL_HRTFINTERPOLATION_BILINEAR;
        bp.spatialBlend = 1.f;
        bp.hrtf = hrtf;
        iplBinauralEffectApply(e.binaural, &bp, &dirBuf, &binBuf);
        iplAudioBufferMix(ctx, &binBuf, &mixBuf);
        // ray-traced reflections, accumulated into the shared mixer (one inverse FFT for all sources)
        if (S.reflections && out.reflections.ir) {
            IPLReflectionEffectParams& rp = out.reflections;
            rp.type = IPL_REFLECTIONEFFECTTYPE_CONVOLUTION;
            rp.numChannels = AMBI_CH;
            rp.irSize = IR_SIZE;
            iplReflectionEffectApply(e.reflect, &rp, &inBuf, &reflBuf, mixer);
            anyRefl = true;
        }
        if (ended && !e.loop) e.state.store(DEAD, std::memory_order_release);
    }
    if (anyRefl) {
        IPLReflectionEffectParams rp{};
        rp.type = IPL_REFLECTIONEFFECTTYPE_CONVOLUTION;
        rp.numChannels = AMBI_CH;
        rp.irSize = IR_SIZE;
        iplReflectionMixerApply(mixer, &rp, &ambiBuf);
        IPLAmbisonicsDecodeEffectParams ap{};
        ap.order = AMBI_ORDER;
        ap.hrtf = hrtf;
        ap.orientation = L;
        ap.binaural = IPL_TRUE;
        iplAmbisonicsDecodeEffectApply(ambiDecode, &ap, &ambiBuf, &decBuf);
        for (int c = 0; c < 2; c++)
            for (int i = 0; i < FRAME; i++) mixBuf.data[c][i] += decBuf.data[c][i] * S.reverbGain;
    }
    // wind bed: the same loop read at two offsets so left/right decorrelate
    float wg = windGain.load();
    if (wg > 0.f && sounds.size() > SND_WIND) {
        static double wc = 0;
        const auto& w = sounds[SND_WIND].s;
        size_t n = w.size();
        for (int i = 0; i < FRAME; i++) {
            size_t k = (size_t)wc % n;
            mixBuf.data[0][i] += w[k] * wg;
            mixBuf.data[1][i] += w[(k + n / 3) % n] * wg;
            wc += 1;
        }
    }
    float mg = S.masterGain;
    for (int i = 0; i < FRAME; i++) {
        // soft clip keeps transient pile-ups (many footsteps + reverb) from wrapping
        outInterleaved[i * 2] = tanhf(mixBuf.data[0][i] * mg);
        outInterleaved[i * 2 + 1] = tanhf(mixBuf.data[1][i] * mg);
    }
    float ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
    dspMs.store(dspMs.load() * 0.95f + ms * 0.05f);
}

void dataCallback(ma_device*, void* output, const void*, ma_uint32 frames) {
    float* out = (float*)output;
    while (frames > 0) {
        if (fifoCount == 0) { renderBlock(fifo); fifoCount = FRAME; }
        int take = std::min<int>(frames, fifoCount);
        int off = FRAME - fifoCount;
        memcpy(out, fifo + off * 2, take * 2 * sizeof(float));
        out += take * 2;
        frames -= take;
        fifoCount -= take;
    }
}

// ---------------------------------------------------------------- simulation thread
void releaseEmitter(Emitter& e) {
    iplSourceRemove(e.src, sim);
    iplSourceRelease(&e.src);
    iplDirectEffectRelease(&e.direct);
    iplBinauralEffectRelease(&e.binaural);
    iplReflectionEffectRelease(&e.reflect);
    e.state.store(FREE, std::memory_order_release);
}

void simMain() {
    prof::setThreadName("audio sim");
    while (!quit.load()) {
        auto t0 = std::chrono::steady_clock::now();
        bool sceneChanged = false, sourcesChanged = false;
        {
            std::vector<std::function<void()>> todo;
            { std::lock_guard l(cmdMtx); todo.swap(cmds); }
            for (auto& f : todo) f();
            sceneChanged = !todo.empty();
        }
        // emitter lifetimes: create pending sources, release finished ones (audio thread no longer touches DEAD)
        for (auto& e : emitters) {
            int st = e.state.load(std::memory_order_acquire);
            if (st == PENDING) {
                IPLSourceSettings ss{(IPLSimulationFlags)(IPL_SIMULATIONFLAGS_DIRECT | IPL_SIMULATIONFLAGS_REFLECTIONS)};
                iplSourceCreate(sim, &ss, &e.src);
                iplSourceAdd(e.src, sim);
                IPLDirectEffectSettings ds{1};
                iplDirectEffectCreate(ctx, &as, &ds, &e.direct);
                IPLBinauralEffectSettings bs{hrtf};
                iplBinauralEffectCreate(ctx, &as, &bs, &e.binaural);
                IPLReflectionEffectSettings rs{IPL_REFLECTIONEFFECTTYPE_CONVOLUTION, IR_SIZE, AMBI_CH};
                iplReflectionEffectCreate(ctx, &as, &rs, &e.reflect);
                sourcesChanged = true;
            } else if (st == DEAD) {
                releaseEmitter(e);
                sourcesChanged = true;
            }
        }
        if (sceneChanged) iplSceneCommit(scene);
        if (sceneChanged || sourcesChanged) iplSimulatorCommit(sim);
        // the commit above makes new sources visible to the simulator; only then hand them to the audio thread
        for (auto& e : emitters)
            if (e.state.load() == PENDING && e.src) e.state.store(LIVE, std::memory_order_release);

        IPLCoordinateSpace3 L = getListener();
        IPLSimulationFlags flags = (IPLSimulationFlags)(IPL_SIMULATIONFLAGS_DIRECT | (S.reflections ? IPL_SIMULATIONFLAGS_REFLECTIONS : 0));
        IPLSimulationSharedInputs shared{};
        shared.listener = L;
        shared.numRays = std::clamp(S.reflectionRays, 128, 4096);
        shared.numBounces = std::clamp(S.bounces, 1, 16);
        shared.duration = IR_SECONDS;
        shared.order = AMBI_ORDER;
        shared.irradianceMinDistance = 1.f;
        iplSimulatorSetSharedInputs(sim, flags, &shared);
        int active = 0;
        for (auto& e : emitters) {
            if (e.state.load() != LIVE) continue;
            IPLSimulationInputs in{};
            in.flags = flags;
            in.directFlags = (IPLDirectSimulationFlags)(IPL_DIRECTSIMULATIONFLAGS_DISTANCEATTENUATION | IPL_DIRECTSIMULATIONFLAGS_AIRABSORPTION |
                                                        IPL_DIRECTSIMULATIONFLAGS_OCCLUSION | IPL_DIRECTSIMULATIONFLAGS_TRANSMISSION);
            in.source = {{1, 0, 0}, {0, 1, 0}, {0, 0, -1}, {e.px.load(), e.py.load(), e.pz.load()}};
            in.distanceAttenuationModel.type = IPL_DISTANCEATTENUATIONTYPE_DEFAULT;
            in.airAbsorptionModel.type = IPL_AIRABSORPTIONTYPE_DEFAULT;
            in.occlusionType = IPL_OCCLUSIONTYPE_VOLUMETRIC;
            in.occlusionRadius = 0.4f;
            in.numOcclusionSamples = 16;
            in.numTransmissionRays = 3;
            in.reverbScale[0] = in.reverbScale[1] = in.reverbScale[2] = 1.f;
            iplSourceSetInputs(e.src, flags, &in);
            active++;
        }
        if (active) {
            PROF_ZONE("audio sim");
            iplSimulatorRunDirect(sim);
            if (S.reflections) iplSimulatorRunReflections(sim);
        }
        float ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
        simMs.store(simMs.load() * 0.9f + ms * 0.1f);
        // direct updates ~60 Hz is plenty; reflections are the long pole and set the real rate
        if (ms < 16.f) std::this_thread::sleep_for(std::chrono::microseconds((int)((16.f - ms) * 1000)));
    }
}

// ---------------------------------------------------------------- builtin sounds
std::vector<float> synthFootstep(uint32_t seed, float cutoff, float thump, float len) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<float> s((size_t)(len * RATE));
    float lp = 0, a = 1.f - expf(-2.f * 3.14159265f * cutoff / RATE);
    for (size_t i = 0; i < s.size(); i++) {
        float t = (float)i / RATE;
        lp += (U(rng) - lp) * a;
        float crunch = lp * expf(-t / 0.035f) * (1.f + 0.6f * U(rng) * expf(-t / 0.01f));
        float body = sinf(2.f * 3.14159265f * (70.f + 40.f * expf(-t / 0.02f)) * t) * expf(-t / 0.05f) * thump;
        s[i] = (crunch * 1.6f + body) * 0.5f;
    }
    return s;
}

std::vector<float> synthHum() {
    std::vector<float> s(2 * RATE);  // 2 s, every partial completes whole cycles so the loop is seamless
    for (size_t i = 0; i < s.size(); i++) {
        float t = (float)i / RATE, w = 2.f * 3.14159265f;
        float trem = 0.8f + 0.2f * sinf(w * 1.5f * t);
        s[i] = trem * (0.5f * sinf(w * 110 * t) + 0.3f * sinf(w * 220 * t) + 0.12f * sinf(w * 330.5f * t) + 0.05f * sinf(w * 661 * t)) * 0.4f;
    }
    return s;
}

std::vector<float> synthWind() {
    const size_t n = 8 * RATE;
    std::mt19937 rng(7);
    std::uniform_real_distribution<float> U(-1.f, 1.f);
    std::vector<float> s(n);
    float lp1 = 0, lp2 = 0;
    for (size_t i = 0; i < n; i++) {
        float t = (float)i / RATE;
        float gust = 0.55f + 0.45f * sinf(2.f * 3.14159265f * t / 8.f) * sinf(2.f * 3.14159265f * t * 3.f / 8.f);
        float fc = 250.f + 500.f * gust, a = 1.f - expf(-2.f * 3.14159265f * fc / RATE);
        lp1 += (U(rng) - lp1) * a;
        lp2 += (lp1 - lp2) * a;
        s[i] = lp2 * gust * 1.5f;
    }
    const size_t xf = RATE / 2;  // crossfade the ends so the loop has no click
    for (size_t i = 0; i < xf; i++) {
        float k = (float)i / xf;
        s[i] = s[i] * k + s[n - xf + i] * (1.f - k);
    }
    s.resize(n - xf);
    return s;
}

void allocBuf(int ch, IPLAudioBuffer& b) { iplAudioBufferAllocate(ctx, ch, FRAME, &b); }
}  // namespace

// ---------------------------------------------------------------- API
bool init() {
    IPLContextSettings cs{};
    cs.version = STEAMAUDIO_VERSION;
    cs.simdLevel = IPL_SIMDLEVEL_AVX2;
    if (iplContextCreate(&cs, &ctx) != IPL_STATUS_SUCCESS) { fprintf(stderr, "audio: Steam Audio context failed\n"); return false; }
    IPLHRTFSettings hs{};
    hs.type = IPL_HRTFTYPE_DEFAULT;
    hs.volume = 1.f;
    hs.normType = IPL_HRTFNORMTYPE_NONE;
    iplHRTFCreate(ctx, &as, &hs, &hrtf);

    IPLEmbreeDeviceSettings es{};
    if (iplEmbreeDeviceCreate(ctx, &es, &embree) == IPL_STATUS_SUCCESS) sceneType = IPL_SCENETYPE_EMBREE;
    IPLSceneSettings scs{};
    scs.type = sceneType;
    scs.embreeDevice = embree;
    iplSceneCreate(ctx, &scs, &scene);

    IPLSimulationSettings ss{};
    ss.flags = (IPLSimulationFlags)(IPL_SIMULATIONFLAGS_DIRECT | IPL_SIMULATIONFLAGS_REFLECTIONS);
    ss.sceneType = sceneType;
    ss.reflectionType = IPL_REFLECTIONEFFECTTYPE_CONVOLUTION;
    ss.maxNumOcclusionSamples = 16;
    ss.maxNumRays = 4096;
    ss.numDiffuseSamples = 32;
    ss.maxDuration = IR_SECONDS;
    ss.maxOrder = AMBI_ORDER;
    ss.maxNumSources = MAX_EMITTERS;
    ss.numThreads = 2;
    ss.samplingRate = RATE;
    ss.frameSize = FRAME;
    if (iplSimulatorCreate(ctx, &ss, &sim) != IPL_STATUS_SUCCESS) { fprintf(stderr, "audio: simulator failed\n"); return false; }
    iplSimulatorSetScene(sim, scene);
    iplSimulatorCommit(sim);

    IPLReflectionEffectSettings rs{IPL_REFLECTIONEFFECTTYPE_CONVOLUTION, IR_SIZE, AMBI_CH};
    iplReflectionMixerCreate(ctx, &as, &rs, &mixer);
    IPLAmbisonicsDecodeEffectSettings ads{};
    ads.speakerLayout.type = IPL_SPEAKERLAYOUTTYPE_STEREO;
    ads.hrtf = hrtf;
    ads.maxOrder = AMBI_ORDER;
    iplAmbisonicsDecodeEffectCreate(ctx, &as, &ads, &ambiDecode);
    allocBuf(1, inBuf); allocBuf(1, dirBuf); allocBuf(2, binBuf); allocBuf(AMBI_CH, reflBuf);
    allocBuf(AMBI_CH, ambiBuf); allocBuf(2, mixBuf); allocBuf(2, decBuf);

    sounds.reserve(MAX_SOUNDS);  // never reallocates: emitters on the audio thread hold pointers into it
    sounds.resize(SND_BUILTIN_COUNT);
    for (uint32_t i = 0; i < 4; i++) sounds[SND_FOOTSTEP0 + i].s = synthFootstep(11 + i * 7, 1800.f + 500.f * i, 0.5f, 0.2f);
    sounds[SND_LAND].s = synthFootstep(99, 1200.f, 1.2f, 0.35f);
    sounds[SND_HUM].s = synthHum();
    sounds[SND_WIND].s = synthWind();

    simThread = std::thread(simMain);

    ma_device_config dc = ma_device_config_init(ma_device_type_playback);
    dc.playback.format = ma_format_f32;
    dc.playback.channels = 2;
    dc.sampleRate = RATE;
    dc.periodSizeInFrames = FRAME / 2;
    dc.dataCallback = dataCallback;
    deviceOk = ma_device_init(nullptr, &dc, &device) == MA_SUCCESS && ma_device_start(&device) == MA_SUCCESS;
    if (!deviceOk) fprintf(stderr, "audio: no output device, running silent\n");
    printf("audio: Steam Audio %s scene, %d Hz, %d-sample frames, %s\n", sceneType == IPL_SCENETYPE_EMBREE ? "Embree" : "built-in", RATE,
           FRAME, deviceOk ? device.playback.name : "no device");
    ok = true;
    return true;
}

void shutdown() {
    if (!ok) return;
    if (deviceOk) ma_device_uninit(&device);
    quit = true;
    simThread.join();
    for (auto& e : emitters)
        if (e.state.load() != FREE && e.src) releaseEmitter(e);
    {
        std::lock_guard l(cmdMtx);
        cmds.clear();
    }
    for (auto& i : instances) if (i.m) iplInstancedMeshRelease(&i.m);
    for (auto& m : meshes) { if (m.mesh) iplStaticMeshRelease(&m.mesh); if (m.scene) iplSceneRelease(&m.scene); }
    if (terrain.mesh) iplStaticMeshRelease(&terrain.mesh);
    for (auto* b : {&inBuf, &dirBuf, &binBuf, &reflBuf, &ambiBuf, &mixBuf, &decBuf}) iplAudioBufferFree(ctx, b);
    iplAmbisonicsDecodeEffectRelease(&ambiDecode);
    iplReflectionMixerRelease(&mixer);
    iplSimulatorRelease(&sim);
    iplSceneRelease(&scene);
    if (embree) iplEmbreeDeviceRelease(&embree);
    iplHRTFRelease(&hrtf);
    iplContextRelease(&ctx);
    ok = false;
}

Settings& settings() { return S; }

uint32_t meshCreate(const float* verts, uint32_t stride, uint32_t nverts, const uint32_t* idx, uint32_t ntris, Material mat) {
    if (!ok) return UINT32_MAX;
    uint32_t id;
    if (!freeMesh.empty()) { id = freeMesh.back(); freeMesh.pop_back(); }
    else id = nextMesh++;
    liveMeshes++;
    // copy positions now (the caller's data may go away), simplify + build on the sim thread
    std::vector<float> pos(nverts * 3);
    for (uint32_t i = 0; i < nverts; i++) memcpy(&pos[i * 3], verts + (size_t)i * stride, 12);
    std::vector<uint32_t> ind(idx, idx + ntris * 3);
    post([id, pos = std::move(pos), ind = std::move(ind), mat]() mutable {
        if (meshes.size() <= id) meshes.resize(id + 1);
        // acoustics don't need visual detail: ~25% of the triangles at 2% error keeps occluders and room shapes intact
        size_t target = std::max<size_t>(ind.size() / 4 / 3 * 3, 36);
        if (ind.size() > 300) {
            std::vector<uint32_t> s(ind.size());
            s.resize(meshopt_simplify(s.data(), ind.data(), ind.size(), pos.data(), pos.size() / 3, 12, target, 0.02f, 0, nullptr));
            if (s.size() >= 3) ind.swap(s);
        }
        // compact to the used vertices
        std::vector<uint32_t> remap(pos.size() / 3, UINT32_MAX);
        std::vector<IPLVector3> v;
        std::vector<IPLTriangle> t(ind.size() / 3);
        for (size_t i = 0; i < ind.size(); i++) {
            uint32_t& r = remap[ind[i]];
            if (r == UINT32_MAX) { r = (uint32_t)v.size(); v.push_back({pos[ind[i] * 3], pos[ind[i] * 3 + 1], pos[ind[i] * 3 + 2]}); }
            t[i / 3].indices[i % 3] = (IPLint32)r;
        }
        std::vector<IPLint32> mi(t.size(), 0);
        IPLMaterial m = kMaterials[mat];
        SubMesh sm;
        IPLSceneSettings scs{};
        scs.type = sceneType;
        scs.embreeDevice = embree;
        iplSceneCreate(ctx, &scs, &sm.scene);
        IPLStaticMeshSettings ms{(IPLint32)v.size(), (IPLint32)t.size(), 1, v.data(), t.data(), mi.data(), &m};
        iplStaticMeshCreate(sm.scene, &ms, &sm.mesh);
        iplStaticMeshAdd(sm.mesh, sm.scene);
        iplSceneCommit(sm.scene);
        sm.tris = (uint32_t)t.size();
        meshes[id] = sm;
    });
    return id;
}

void meshRelease(uint32_t id) {
    if (!ok || id == UINT32_MAX) return;
    freeMesh.push_back(id);
    liveMeshes--;
    post([id] {
        SubMesh& m = meshes[id];
        if (m.mesh) iplStaticMeshRelease(&m.mesh);
        if (m.scene) iplSceneRelease(&m.scene);  // instances hold their own reference to the sub-scene
        m = {};
    });
}

uint32_t instanceAdd(uint32_t mesh, const Xf& xf) {
    if (!ok || mesh == UINT32_MAX) return UINT32_MAX;
    uint32_t id;
    if (!freeInst.empty()) { id = freeInst.back(); freeInst.pop_back(); }
    else id = nextInst++;
    liveInsts++;
    post([id, mesh, xf] {
        if (instances.size() <= id) instances.resize(id + 1);
        if (mesh >= meshes.size() || !meshes[mesh].scene) return;
        IPLInstancedMeshSettings is{};
        is.subScene = meshes[mesh].scene;
        for (int r = 0; r < 3; r++)
            for (int c = 0; c < 4; c++) is.transform.elements[r][c] = xf.m[r][c];
        is.transform.elements[3][3] = 1.f;
        iplInstancedMeshCreate(scene, &is, &instances[id].m);
        iplInstancedMeshAdd(instances[id].m, scene);
        instances[id].tris = meshes[mesh].tris;
        sceneTris += meshes[mesh].tris;
    });
    return id;
}

void instanceRemove(uint32_t id) {
    if (!ok || id == UINT32_MAX) return;
    freeInst.push_back(id);
    liveInsts--;
    post([id] {
        if (id >= instances.size() || !instances[id].m) return;
        iplInstancedMeshRemove(instances[id].m, scene);
        iplInstancedMeshRelease(&instances[id].m);
        sceneTris -= instances[id].tris;
        instances[id] = {};
    });
}

void setTerrain(const float* verts, uint32_t nverts, const uint32_t* idx, uint32_t ntris) {
    if (!ok) return;
    std::vector<IPLVector3> v(nverts);
    for (uint32_t i = 0; i < nverts; i++) v[i] = {verts[i * 3], verts[i * 3 + 1], verts[i * 3 + 2]};
    std::vector<IPLTriangle> t(ntris);
    for (uint32_t i = 0; i < ntris; i++) t[i] = {{(IPLint32)idx[i * 3], (IPLint32)idx[i * 3 + 1], (IPLint32)idx[i * 3 + 2]}};
    post([v = std::move(v), t = std::move(t)]() mutable {
        if (terrain.mesh) { iplStaticMeshRemove(terrain.mesh, scene); iplStaticMeshRelease(&terrain.mesh); }
        std::vector<IPLint32> mi(t.size(), 0);
        IPLMaterial m = kMaterials[MAT_GROUND];
        IPLStaticMeshSettings ms{(IPLint32)v.size(), (IPLint32)t.size(), 1, v.data(), t.data(), mi.data(), &m};
        iplStaticMeshCreate(scene, &ms, &terrain.mesh);
        iplStaticMeshAdd(terrain.mesh, scene);
        terrain.tris = (uint32_t)t.size();
    });
}

void setListener(const float p[3], const float f[3], const float u[3]) {
    IPLVector3 ahead{f[0], f[1], f[2]}, up{u[0], u[1], u[2]};
    IPLVector3 right{ahead.y * up.z - ahead.z * up.y, ahead.z * up.x - ahead.x * up.z, ahead.x * up.y - ahead.y * up.x};
    std::lock_guard l(lisMtx);
    listener = {right, up, ahead, {p[0], p[1], p[2]}};
}

uint32_t soundLoad(const char* path) {
    ma_decoder_config dc = ma_decoder_config_init(ma_format_f32, 1, RATE);
    float* frames = nullptr;
    ma_uint64 n = 0;
    if (ma_decode_file(path, &dc, &n, (void**)&frames) != MA_SUCCESS) { fprintf(stderr, "audio: cannot decode %s\n", path); return UINT32_MAX; }
    Sound s;
    s.s.assign(frames, frames + n);
    ma_free(frames, nullptr);
    if (s.s.size() < 2 || sounds.size() >= MAX_SOUNDS) return UINT32_MAX;
    sounds.push_back(std::move(s));
    return (uint32_t)sounds.size() - 1;
}

uint32_t emitterPlay(uint32_t sound, const float pos[3], float gain, bool loop, float pitch) {
    if (!ok || sound >= sounds.size()) return UINT32_MAX;
    for (uint32_t i = 0; i < MAX_EMITTERS; i++) {
        Emitter& e = emitters[i];
        if (e.state.load(std::memory_order_acquire) != FREE) continue;
        e.sound = &sounds[sound];
        e.loop = loop;
        e.pitch = pitch;
        e.cursor = 0;
        e.stop = false;
        e.px = pos[0]; e.py = pos[1]; e.pz = pos[2];
        e.gain = gain;
        e.src = nullptr;
        e.state.store(PENDING, std::memory_order_release);
        return i;
    }
    return UINT32_MAX;
}

void emitterSetPos(uint32_t i, const float p[3]) {
    if (i >= MAX_EMITTERS) return;
    emitters[i].px = p[0]; emitters[i].py = p[1]; emitters[i].pz = p[2];
}
void emitterSetGain(uint32_t i, float g) { if (i < MAX_EMITTERS) emitters[i].gain = g; }
void emitterStop(uint32_t i) { if (i < MAX_EMITTERS) emitters[i].stop = true; }
bool emitterAlive(uint32_t i) {
    if (i >= MAX_EMITTERS) return false;
    int s = emitters[i].state.load();
    return (s == PENDING || s == LIVE) && !emitters[i].stop.load();
}
void setAmbience(float g) { windGain = g; }

Stats stats() {
    Stats s{};
    for (auto& e : emitters) s.emitters += e.state.load() == LIVE;
    s.meshes = liveMeshes;
    s.instances = liveInsts;
    s.sceneTris = sceneTris + terrain.tris;
    s.simMs = simMs.load();
    s.dspMs = dspMs.load();
    s.running = ok && deviceOk;
    return s;
}
}  // namespace audio

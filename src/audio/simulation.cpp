// The simulation thread: Steam Audio rays for direct paths (occlusion, transmission) and reflection impulse responses.
#include "audio/audio_internal.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>

#include "core/prof.h"

namespace audio {
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
            // reverb changes slowly as the listener moves: a few updates a second sound the same as 60
            static auto lastRefl = t0 - std::chrono::seconds(1);
            if (S.reflections && std::chrono::duration<float>(t0 - lastRefl).count() >= 1.f / std::max(S.reflectionHz, 1.f)) {
                iplSimulatorRunReflections(sim);
                lastRefl = t0;
            }
        }
        float ms = std::chrono::duration<float, std::milli>(std::chrono::steady_clock::now() - t0).count();
        simMs.store(simMs.load() * 0.9f + ms * 0.1f);
        // direct updates ~60 Hz is plenty; reflections are the long pole and set the real rate
        if (ms < 16.f) std::this_thread::sleep_for(std::chrono::microseconds((int)((16.f - ms) * 1000)));
    }
}

}  // namespace audio

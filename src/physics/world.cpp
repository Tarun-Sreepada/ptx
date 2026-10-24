// Jolt setup: the physics system with its jobs on the shared enkiTS scheduler, terrain contact materials, the step.
#include "physics/jolt_internal.h"

#include "core/constants.h"
#include "core/jobs.h"

namespace physics {
std::unique_ptr<PhysicsSystem> sys;
bool addedSinceOptimize = false;
std::unordered_map<uint32_t, Ref<Ragdoll>> ragdolls;
uint32_t nextRagdoll = 1;
std::mutex hitMu;
std::vector<RagdollHit> hits;
std::unordered_map<uint32_t, std::pair<uint32_t, int>> ragdollOfBody;
Ref<CharacterVirtual> player;

std::unique_ptr<TempAllocatorImpl> temp;

namespace {
// Jolt jobs run on the shared enkiTS scheduler (jobs.h) rather than Jolt's own thread pool.
class EnkiJobSystem final : public JobSystemWithBarrier {
public:
    EnkiJobSystem(uint maxJobs, uint maxBarriers) : JobSystemWithBarrier(maxBarriers), tasks(maxJobs) {
        pool.Init(maxJobs, maxJobs);
    }
    ~EnkiJobSystem() override {
        for (auto& t : tasks) jobs::scheduler().WaitforTask(&t);
    }
    int GetMaxConcurrency() const override { return (int)jobs::threadCount(); }
    JobHandle CreateJob(const char* name, ColorArg color, const JobFunction& fn, uint32 numDeps = 0) override {
        uint32 index;
        while ((index = pool.ConstructObject(name, color, this, fn, numDeps)) == decltype(pool)::cInvalidObjectIndex)
            std::this_thread::yield();  // out of job slots: wait for workers to free some
        Job* job = &pool.Get(index);
        JobHandle h(job);
        if (numDeps == 0) QueueJob(job);
        return h;
    }

protected:
    void QueueJob(Job* job) override {
        job->AddRef();
        // grab a task object that has finished running (ring, so normally the first probe succeeds)
        for (;;) {
            Task& t = tasks[next.fetch_add(1, std::memory_order_relaxed) % tasks.size()];
            if (!t.GetIsComplete()) continue;
            t.job = job;
            jobs::scheduler().AddTaskSetToPipe(&t);
            return;
        }
    }
    void QueueJobs(Job** js, uint n) override {
        for (uint i = 0; i < n; i++) QueueJob(js[i]);
    }
    void FreeJob(Job* job) override { pool.DestructObject(job); }

private:
    struct Task : enki::ITaskSet {
        Job* job = nullptr;
        void ExecuteRange(enki::TaskSetPartition, uint32_t) override {
            job->Execute();
            job->Release();
        }
    };
    FixedSizeFreeList<Job> pool;
    std::vector<Task> tasks;
    std::atomic<uint32_t> next{0};
};
std::unique_ptr<EnkiJobSystem> jobSys;
std::unique_ptr<ObjectLayerPairFilterTable> pairFilter;
std::unique_ptr<BroadPhaseLayerInterfaceTable> bpInterface;
std::unique_ptr<ObjectVsBroadPhaseLayerFilterTable> bpFilter;
physics::SurfaceFn surfaceFn = nullptr;
// terrain contacts take the ground's friction / bounce at the contact point
class SurfaceListener : public ContactListener {
public:
    void apply(const Body& a, const Body& b, const ContactManifold& m, ContactSettings& s) {
        if (!surfaceFn) return;
        const Body* ground = a.GetShape()->GetSubType() == EShapeSubType::HeightField ? &a
                           : b.GetShape()->GetSubType() == EShapeSubType::HeightField ? &b : nullptr;
        if (!ground) return;
        const Body& other = ground == &a ? b : a;
        RVec3 p = m.GetWorldSpaceContactPointOn1(0);
        float pp[3] = {(float)p.GetX(), (float)p.GetY(), (float)p.GetZ()}, f = 0.7f, e = 0.15f;
        surfaceFn(pp, f, e);
        // combine like Jolt's defaults would with the body's own: geometric friction, the larger bounce
        s.mCombinedFriction = sqrtf(f * other.GetFriction());
        s.mCombinedRestitution = std::min(std::max(e, 0.f), other.GetRestitution() > e ? 0.5f * (e + other.GetRestitution()) : e);
    }
    void OnContactAdded(const Body& a, const Body& b, const ContactManifold& m, ContactSettings& s) override {
        apply(a, b, m, s);
        ragdollHit(a, b, m);
    }
    void ragdollHit(const Body& a, const Body& b, const ContactManifold& m);
    void OnContactPersisted(const Body& a, const Body& b, const ContactManifold& m, ContactSettings& s) override { apply(a, b, m, s); }
};
void SurfaceListener::ragdollHit(const Body& a, const Body& b, const ContactManifold& m) {
    const Body* rd = (a.GetUserData() & RAGDOLL_TAG) ? &a : (b.GetUserData() & RAGDOLL_TAG) ? &b : nullptr;
    if (!rd || !rd->IsKinematic()) return;
    const Body& o = rd == &a ? b : a;
    if (!o.IsDynamic() || (o.GetUserData() & RAGDOLL_TAG)) return;
    Vec3 v = o.GetLinearVelocity() - rd->GetLinearVelocity();
    if (v.LengthSq() < 4.f) return;  // 2 m/s: a nudge, not a hit
    auto it = ragdollOfBody.find(rd->GetID().GetIndexAndSequenceNumber());
    if (it == ragdollOfBody.end()) return;
    float mass = 1.f / std::max(o.GetMotionPropertiesUnchecked()->GetInverseMass(), 1e-6f);
    RVec3 p = m.GetWorldSpaceContactPointOn1(0);
    physics::RagdollHit h{it->second.first, it->second.second, {(float)p.GetX(), (float)p.GetY(), (float)p.GetZ()},
                          {v.GetX() * mass, v.GetY() * mass, v.GetZ() * mass}, mass};
    std::lock_guard lk(hitMu);
    hits.push_back(h);
}
SurfaceListener surfaceListener;
}  // namespace

void setSurfaceFn(SurfaceFn fn) { surfaceFn = fn; }

void init() {
    RegisterDefaultAllocator();
    Factory::sInstance = new Factory();
    RegisterTypes();
    temp = std::make_unique<TempAllocatorImpl>(32 * 1024 * 1024);
    jobSys = std::make_unique<EnkiJobSystem>(cMaxPhysicsJobs, cMaxPhysicsBarriers);

    pairFilter = std::make_unique<ObjectLayerPairFilterTable>(Layers::COUNT);
    pairFilter->EnableCollision(Layers::MOVING, Layers::STATIC);
    pairFilter->EnableCollision(Layers::MOVING, Layers::MOVING);
    bpInterface = std::make_unique<BroadPhaseLayerInterfaceTable>(Layers::COUNT, BPLayers::COUNT);
    bpInterface->MapObjectToBroadPhaseLayer(Layers::STATIC, BPLayers::STATIC);
    bpInterface->MapObjectToBroadPhaseLayer(Layers::MOVING, BPLayers::MOVING);
    bpFilter = std::make_unique<ObjectVsBroadPhaseLayerFilterTable>(*bpInterface, BPLayers::COUNT, *pairFilter, Layers::COUNT);

    sys = std::make_unique<PhysicsSystem>();
    sys->Init(262144, 0, 65536, 32768, *bpInterface, *bpFilter, *pairFilter);
    sys->SetContactListener(&surfaceListener);
    sys->SetGravity(Vec3(0, -phys::GRAVITY, 0));
}

void shutdown() {
    player = nullptr;
    for (auto& [h, r] : ragdolls) r->RemoveFromPhysicsSystem();
    ragdolls.clear();
    sys.reset();
    bpFilter.reset();
    bpInterface.reset();
    pairFilter.reset();
    jobSys.reset();
    temp.reset();
    UnregisterTypes();
    delete Factory::sInstance;
    Factory::sInstance = nullptr;
}

void step(float dt) {
    if (addedSinceOptimize) { sys->OptimizeBroadPhase(); addedSinceOptimize = false; }
    sys->Update(dt, 1, temp.get(), jobSys.get());
}

uint32_t bodyCount() { return sys->GetNumBodies(); }
}  // namespace physics

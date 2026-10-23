// Jolt physics: static world (streamed meshes + terrain heightfields) and a CharacterVirtual player.
#include <Jolt/Jolt.h>

#include <Jolt/Core/Factory.h>
#include <Jolt/Core/FixedSizeFreeList.h>
#include <Jolt/Core/JobSystemWithBarrier.h>
#include <Jolt/Core/TempAllocator.h>
#include <TaskScheduler.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <thread>
#include <Jolt/Physics/Body/BodyCreationSettings.h>
#include <Jolt/Physics/Character/CharacterVirtual.h>
#include <Jolt/Physics/Collision/BroadPhase/BroadPhaseLayerInterfaceTable.h>
#include <Jolt/Physics/Collision/BroadPhase/ObjectVsBroadPhaseLayerFilterTable.h>
#include <Jolt/Physics/Collision/ObjectLayerPairFilterTable.h>
#include <Jolt/Physics/Collision/Shape/CapsuleShape.h>
#include <Jolt/Physics/Collision/Shape/ConvexHullShape.h>
#include <Jolt/Physics/Collision/Shape/HeightFieldShape.h>
#include <Jolt/Physics/Collision/Shape/MeshShape.h>
#include <Jolt/Physics/Collision/Shape/RotatedTranslatedShape.h>
#include <Jolt/Physics/Collision/Shape/ScaledShape.h>
#include <Jolt/Physics/Collision/Shape/StaticCompoundShape.h>
#include <Jolt/Physics/Collision/ContactListener.h>
#include <Jolt/Physics/Constraints/SwingTwistConstraint.h>
#include <Jolt/Physics/PhysicsSystem.h>
#include <Jolt/Physics/Ragdoll/Ragdoll.h>
#include <Jolt/Skeleton/Skeleton.h>
#include <Jolt/RegisterTypes.h>

#include <cstdio>
#include <memory>
#include <mutex>
#include <unordered_map>

#include "physics.h"
#include "jobs.h"

using namespace JPH;

namespace {
namespace Layers { constexpr ObjectLayer STATIC = 0, MOVING = 1, COUNT = 2; }
namespace BPLayers { constexpr BroadPhaseLayer STATIC(0), MOVING(1); constexpr uint COUNT = 2; }

std::unique_ptr<TempAllocatorImpl> temp;
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
std::unique_ptr<PhysicsSystem> sys;
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
// ragdolls: body user data = RAGDOLL_TAG | handle
constexpr uint64 RAGDOLL_TAG = 1ull << 63;
std::unordered_map<uint32_t, Ref<Ragdoll>> ragdolls;
uint32_t nextRagdoll = 1;
std::mutex hitMu;
std::vector<physics::RagdollHit> hits;
std::unordered_map<uint32_t, std::pair<uint32_t, int>> ragdollOfBody;  // body id -> (ragdoll, part), main thread writes only between steps
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
Ref<CharacterVirtual> player;
bool addedSinceOptimize = false;
}  // namespace

namespace physics {

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
    sys->SetGravity(Vec3(0, -9.8f, 0));
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

void* meshShape(const float* verts, uint32_t stride, uint32_t nverts, const uint32_t* idx, uint32_t ntris) {
    VertexList vl(nverts);
    for (uint32_t i = 0; i < nverts; i++) vl[i] = Float3(verts[i * stride], verts[i * stride + 1], verts[i * stride + 2]);
    IndexedTriangleList tl(ntris);
    for (uint32_t i = 0; i < ntris; i++) tl[i] = IndexedTriangle(idx[i * 3], idx[i * 3 + 1], idx[i * 3 + 2]);
    MeshShapeSettings s(std::move(vl), std::move(tl));
    s.SetEmbedded();
    auto r = s.Create();
    if (r.HasError()) { fprintf(stderr, "mesh shape: %s\n", r.GetError().c_str()); return nullptr; }
    Shape* sh = const_cast<Shape*>(r.Get().GetPtr());
    sh->AddRef();
    return sh;
}

void* heightShape(const float* h, uint32_t n, float x0, float z0, float spacing) {
    HeightFieldShapeSettings s(h, Vec3(x0, 0, z0), Vec3(spacing, 1, spacing), n);
    s.SetEmbedded();
    auto r = s.Create();
    if (r.HasError()) { fprintf(stderr, "heightfield: %s\n", r.GetError().c_str()); return nullptr; }
    Shape* sh = const_cast<Shape*>(r.Get().GetPtr());
    sh->AddRef();
    return sh;
}

void releaseShape(void* s) {
    if (s) ((Shape*)s)->Release();
}

void addStatic(void* const* shapes, const Xf* xfs, size_t n, uint32_t* out) {
    BodyInterface& bi = sys->GetBodyInterfaceNoLock();
    std::vector<BodyID> ids;
    ids.reserve(n);
    for (size_t i = 0; i < n; i++) {
        const Xf& x = xfs[i];
        // 3x4 -> scale * rotation + translation (no shear); a mirrored transform gets a negative x scale
        Vec3 c0(x.m[0][0], x.m[1][0], x.m[2][0]), c1(x.m[0][1], x.m[1][1], x.m[2][1]), c2(x.m[0][2], x.m[1][2], x.m[2][2]);
        Vec3 sc(c0.Length(), c1.Length(), c2.Length());
        if (c0.Cross(c1).Dot(c2) < 0) sc.SetX(-sc.GetX());
        Mat44 rot(Vec4(c0 / sc.GetX(), 0), Vec4(c1 / sc.GetY(), 0), Vec4(c2 / sc.GetZ(), 0), Vec4(0, 0, 0, 1));
        Quat q = rot.GetQuaternion().Normalized();
        RefConst<Shape> shape = (const Shape*)shapes[i];
        if (!sc.IsClose(Vec3::sOne(), 1e-6f)) shape = new ScaledShape(shape, sc);
        BodyCreationSettings bcs(shape, RVec3(x.m[0][3], x.m[1][3], x.m[2][3]), q, EMotionType::Static, Layers::STATIC);
        Body* b = bi.CreateBody(bcs);
        if (!b) { out[i] = BodyID::cInvalidBodyID; continue; }
        out[i] = b->GetID().GetIndexAndSequenceNumber();
        ids.push_back(b->GetID());
    }
    if (ids.empty()) return;
    auto st = bi.AddBodiesPrepare(ids.data(), (int)ids.size());
    bi.AddBodiesFinalize(ids.data(), (int)ids.size(), st, EActivation::DontActivate);
    addedSinceOptimize = true;
}

void removeBodies(const uint32_t* raw, size_t n) {
    std::vector<BodyID> ids;
    for (size_t i = 0; i < n; i++)
        if (raw[i] != BodyID::cInvalidBodyID) ids.emplace_back(raw[i]);
    if (ids.empty()) return;
    BodyInterface& bi = sys->GetBodyInterfaceNoLock();
    bi.RemoveBodies(ids.data(), (int)ids.size());
    bi.DestroyBodies(ids.data(), (int)ids.size());
}

void step(float dt) {
    if (addedSinceOptimize) { sys->OptimizeBroadPhase(); addedSinceOptimize = false; }
    sys->Update(dt, 1, temp.get(), jobSys.get());
}

uint32_t bodyCount() { return sys->GetNumBodies(); }

void* convexShape(const float* verts, uint32_t stride, uint32_t n) {
    Array<Vec3> pts(n);
    for (uint32_t i = 0; i < n; i++) pts[i] = Vec3(verts[i * stride], verts[i * stride + 1], verts[i * stride + 2]);
    ConvexHullShapeSettings s(pts);
    s.SetEmbedded();
    auto r = s.Create();
    if (r.HasError()) { fprintf(stderr, "convex hull: %s\n", r.GetError().c_str()); return nullptr; }
    Shape* sh = const_cast<Shape*>(r.Get().GetPtr());
    sh->AddRef();
    return sh;
}

void* compoundShape(const float* pts, const uint32_t* counts, uint32_t nHulls) {
    StaticCompoundShapeSettings cs;
    for (uint32_t h = 0; h < nHulls; pts += counts[h++] * 3) {
        Array<Vec3> p(counts[h]);
        for (uint32_t i = 0; i < counts[h]; i++) p[i] = Vec3(pts[i * 3], pts[i * 3 + 1], pts[i * 3 + 2]);
        auto r = ConvexHullShapeSettings(p, 0.005f).Create();
        if (!r.HasError()) cs.AddShape(Vec3::sZero(), Quat::sIdentity(), r.Get());
    }
    if (cs.mSubShapes.empty()) return nullptr;
    cs.SetEmbedded();
    auto r = cs.Create();
    if (r.HasError()) { fprintf(stderr, "compound: %s\n", r.GetError().c_str()); return nullptr; }
    Shape* sh = const_cast<Shape*>(r.Get().GetPtr());
    sh->AddRef();
    return sh;
}

float shapeVolume(void* shape) { return ((const Shape*)shape)->GetVolume(); }

uint32_t addDynamic(void* shape, float scale, const float p[3], const float v[3], const float w[3], float density, float yaw) {
    BodyInterface& bi = sys->GetBodyInterface();
    RefConst<Shape> sh = (const Shape*)shape;
    if (fabsf(scale - 1.f) > 1e-6f) sh = new ScaledShape(sh, Vec3::sReplicate(scale));
    BodyCreationSettings bcs(sh, RVec3(p[0], p[1], p[2]), Quat::sRotation(Vec3::sAxisY(), yaw), EMotionType::Dynamic, Layers::MOVING);
    bcs.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
    bcs.mMassPropertiesOverride.mMass = density * sh->GetVolume();
    bcs.mLinearVelocity = Vec3(v[0], v[1], v[2]);
    bcs.mAngularVelocity = Vec3(w[0], w[1], w[2]);
    bcs.mFriction = 0.7f;
    bcs.mRestitution = 0.15f;
    bcs.mMotionQuality = EMotionQuality::LinearCast;  // fast throws don't tunnel through the heightfield
    Body* b = bi.CreateBody(bcs);
    if (!b) return BodyID::cInvalidBodyID;
    bi.AddBody(b->GetID(), EActivation::Activate);
    return b->GetID().GetIndexAndSequenceNumber();
}

uint32_t addDynamicPoseInertia(void* shape, const Xf& x, const float v[3], const float w[3], float mass, const float com[3],
                               const float I[9]) {
    BodyInterface& bi = sys->GetBodyInterface();
    Mat44 rot(Vec4(x.m[0][0], x.m[1][0], x.m[2][0], 0), Vec4(x.m[0][1], x.m[1][1], x.m[2][1], 0), Vec4(x.m[0][2], x.m[1][2], x.m[2][2], 0),
              Vec4(0, 0, 0, 1));
    const Shape* sh = (const Shape*)shape;
    BodyCreationSettings bcs(sh, RVec3(x.m[0][3], x.m[1][3], x.m[2][3]), rot.GetQuaternion().Normalized(), EMotionType::Dynamic, Layers::MOVING);
    Vec3 d = sh->GetCenterOfMass() - Vec3(com[0], com[1], com[2]);  // parallel axis
    float dv[3] = {d.GetX(), d.GetY(), d.GetZ()}, dd = d.LengthSq();
    Mat44 in = Mat44::sIdentity();
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 3; c++) in(r, c) = I[r * 3 + c] + mass * ((r == c ? dd : 0.f) - dv[r] * dv[c]);
    bcs.mOverrideMassProperties = EOverrideMassProperties::MassAndInertiaProvided;
    bcs.mMassPropertiesOverride.mMass = mass;
    bcs.mMassPropertiesOverride.mInertia = in;
    bcs.mLinearVelocity = Vec3(v[0], v[1], v[2]);
    bcs.mAngularVelocity = Vec3(w[0], w[1], w[2]);
    bcs.mFriction = 0.7f;
    bcs.mRestitution = 0.1f;  // green wood and leaves: dead, no bounce
    bcs.mMotionQuality = EMotionQuality::LinearCast;
    Body* b = bi.CreateBody(bcs);
    if (!b) return BodyID::cInvalidBodyID;
    bi.AddBody(b->GetID(), EActivation::Activate);
    return b->GetID().GetIndexAndSequenceNumber();
}

uint32_t addDynamicPose(void* shape, const Xf& x, const float v[3], const float w[3], float density) {
    BodyInterface& bi = sys->GetBodyInterface();
    Mat44 rot(Vec4(x.m[0][0], x.m[1][0], x.m[2][0], 0), Vec4(x.m[0][1], x.m[1][1], x.m[2][1], 0), Vec4(x.m[0][2], x.m[1][2], x.m[2][2], 0),
              Vec4(0, 0, 0, 1));
    const Shape* sh = (const Shape*)shape;
    BodyCreationSettings bcs(sh, RVec3(x.m[0][3], x.m[1][3], x.m[2][3]), rot.GetQuaternion().Normalized(), EMotionType::Dynamic, Layers::MOVING);
    bcs.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
    bcs.mMassPropertiesOverride.mMass = density * sh->GetVolume();
    bcs.mLinearVelocity = Vec3(v[0], v[1], v[2]);
    bcs.mAngularVelocity = Vec3(w[0], w[1], w[2]);
    bcs.mFriction = 0.7f;
    bcs.mRestitution = 0.15f;
    bcs.mMotionQuality = EMotionQuality::LinearCast;
    Body* b = bi.CreateBody(bcs);
    if (!b) return BodyID::cInvalidBodyID;
    bi.AddBody(b->GetID(), EActivation::Activate);
    return b->GetID().GetIndexAndSequenceNumber();
}

bool bodyState(uint32_t id, float scale, Xf& xf, float vel[3], float angVel[3], float& mass) {
    BodyLockRead lock(sys->GetBodyLockInterface(), BodyID(id));
    if (!lock.Succeeded()) return false;
    const Body& b = lock.GetBody();
    Mat44 m = b.GetWorldTransform();
    for (int r = 0; r < 3; r++) {
        for (int c = 0; c < 3; c++) xf.m[r][c] = m(r, c) * scale;
        xf.m[r][3] = m(r, 3);
    }
    Vec3 v = b.GetLinearVelocity(), w = b.GetAngularVelocity();
    for (int k = 0; k < 3; k++) { vel[k] = v[k]; angVel[k] = w[k]; }
    mass = 1.f / std::max(b.GetMotionProperties()->GetInverseMass(), 1e-9f);
    return true;
}

void addImpulse(uint32_t id, const float j[3], const float at[3]) {
    sys->GetBodyInterface().AddImpulse(BodyID(id), Vec3(j[0], j[1], j[2]), RVec3(at[0], at[1], at[2]));
}

void setVelocity(uint32_t id, const float v[3]) {
    sys->GetBodyInterface().SetLinearVelocity(BodyID(id), Vec3(v[0], v[1], v[2]));
}

void setAngularVelocity(uint32_t id, const float w[3]) {
    sys->GetBodyInterface().SetAngularVelocity(BodyID(id), Vec3(w[0], w[1], w[2]));
}

void setDamping(uint32_t id, float linear, float angular) {
    BodyLockWrite lock(sys->GetBodyLockInterface(), BodyID(id));
    if (!lock.Succeeded() || !lock.GetBody().GetMotionProperties()) return;
    lock.GetBody().GetMotionProperties()->SetLinearDamping(linear);
    lock.GetBody().GetMotionProperties()->SetAngularDamping(angular);
}

void addForce(uint32_t id, const float f[3], const float t[3]) {
    BodyInterface& bi = sys->GetBodyInterface();
    bi.AddForceAndTorque(BodyID(id), Vec3(f[0], f[1], f[2]), Vec3(t[0], t[1], t[2]), EActivation::Activate);
}

// ---------------------------------------------------------------- ragdolls
static Mat44 toMat(const Xf& x) {
    return Mat44(Vec4(x.m[0][0], x.m[1][0], x.m[2][0], 0), Vec4(x.m[0][1], x.m[1][1], x.m[2][1], 0), Vec4(x.m[0][2], x.m[1][2], x.m[2][2], 0),
                 Vec4(x.m[0][3], x.m[1][3], x.m[2][3], 1));
}
static Xf fromMat(const Mat44& m) {
    Xf x;
    for (int r = 0; r < 3; r++)
        for (int c = 0; c < 4; c++) x.m[r][c] = m(r, c);
    return x;
}
static Ragdoll* ragdoll(uint32_t h) { auto it = ragdolls.find(h); return it == ragdolls.end() ? nullptr : it->second.GetPtr(); }

uint32_t ragdollCreate(const RagdollPart* parts, size_t n, float totalMass) {
    Ref<RagdollSettings> rs = new RagdollSettings;
    rs->mSkeleton = new Skeleton;
    rs->mParts.resize(n);
    float vol = 0;
    for (size_t i = 0; i < n; i++) {
        const RagdollPart& p = parts[i];
        Vec3 a(p.a[0], p.a[1], p.a[2]), b(p.b[0], p.b[1], p.b[2]);
        float hl = std::max(0.5f * (b - a).Length(), 1e-3f);
        vol += 3.14159f * p.radius * p.radius * (2.f * hl + 4.f / 3.f * p.radius);
    }
    float density = totalMass / std::max(vol, 1e-6f);
    for (size_t i = 0; i < n; i++) {
        const RagdollPart& p = parts[i];
        rs->mSkeleton->AddJoint("b" + std::to_string(i), p.parent);
        Vec3 a(p.a[0], p.a[1], p.a[2]), b(p.b[0], p.b[1], p.b[2]), d = b - a;
        float len = d.Length();
        // Jolt capsules run along y: rotate y onto the segment, centre it
        Quat q = len > 1e-5f ? Quat::sFromTo(Vec3::sAxisY(), d / len) : Quat::sIdentity();
        Ref<Shape> cap = new CapsuleShape(std::max(0.5f * len, 1e-3f), p.radius);
        Ref<Shape> sh = RotatedTranslatedShapeSettings(0.5f * (a + b), q, cap).Create().Get();
        Mat44 m = toMat(p.xf);
        RagdollSettings::Part& part = rs->mParts[i];
        part.SetShape(sh);
        part.mPosition = RVec3(m.GetTranslation());
        part.mRotation = m.GetQuaternion().Normalized();
        part.mMotionType = EMotionType::Kinematic;
        part.mAllowDynamicOrKinematic = true;
        part.mObjectLayer = Layers::MOVING;
        part.mOverrideMassProperties = EOverrideMassProperties::CalculateInertia;
        part.mMassPropertiesOverride.mMass = std::max(density * 3.14159f * p.radius * p.radius * (len + 4.f / 3.f * p.radius), 0.2f);
        part.mLinearDamping = 0.1f;
        part.mAngularDamping = 0.6f;  // limbs settle instead of flailing
        part.mFriction = 0.8f;
        if (p.parent >= 0) {
            // the joint sits at this bone's origin; twist about the bone's own direction
            Ref<SwingTwistConstraintSettings> c = new SwingTwistConstraintSettings;
            Vec3 at = m * Vec3::sZero();
            Vec3 twist = len > 1e-5f ? m.Multiply3x3(d / len).Normalized() : m.GetAxisY();
            c->mPosition1 = c->mPosition2 = RVec3(at);
            c->mTwistAxis1 = c->mTwistAxis2 = twist;
            c->mPlaneAxis1 = c->mPlaneAxis2 = twist.GetNormalizedPerpendicular();
            c->mNormalHalfConeAngle = c->mPlaneHalfConeAngle = p.swing;
            c->mTwistMinAngle = -p.twist;
            c->mTwistMaxAngle = p.twist;
            c->mMaxFrictionTorque = 2.f;  // a little joint friction: no rag-flapping
            part.mToParent = c;
        }
    }
    rs->mSkeleton->CalculateParentJointIndices();
    rs->Stabilize();
    {   // also bodies that overlap in the creation pose (thighs, upper arms against the chest): they'd push apart
        std::vector<Mat44> pose(n);
        for (size_t i = 0; i < n; i++) pose[i] = toMat(parts[i].xf);
        rs->DisableParentChildCollisions(pose.data(), 0.02f);
    }
    rs->CalculateBodyIndexToConstraintIndex();
    rs->CalculateConstraintIndexToBodyIdxPair();
    uint32_t h = nextRagdoll++;
    Ragdoll* r = rs->CreateRagdoll((CollisionGroup::GroupID)h, RAGDOLL_TAG | h, sys.get());
    if (!r) return 0;
    r->AddToPhysicsSystem(EActivation::Activate);
    ragdolls.emplace(h, r);
    for (size_t i = 0; i < r->GetBodyCount(); i++) ragdollOfBody[r->GetBodyID((int)i).GetIndexAndSequenceNumber()] = {h, (int)i};
    addedSinceOptimize = true;
    return h;
}

void ragdollDestroy(uint32_t h) {
    Ragdoll* r = ragdoll(h);
    if (!r) return;
    for (size_t i = 0; i < r->GetBodyCount(); i++) ragdollOfBody.erase(r->GetBodyID((int)i).GetIndexAndSequenceNumber());
    r->RemoveFromPhysicsSystem();
    ragdolls.erase(h);
}

static void toMats(const Ragdoll* r, const Xf* xfs, std::vector<Mat44>& m) {
    m.resize(r->GetBodyCount());
    for (size_t i = 0; i < m.size(); i++) m[i] = toMat(xfs[i]);
}
void ragdollDrive(uint32_t h, const Xf* xfs, float dt) {
    Ragdoll* r = ragdoll(h);
    if (!r || dt <= 0.f) return;
    std::vector<Mat44> m;
    toMats(r, xfs, m);
    r->DriveToPoseUsingKinematics(RVec3::sZero(), m.data(), dt);
}
void ragdollTeleport(uint32_t h, const Xf* xfs) {
    Ragdoll* r = ragdoll(h);
    if (!r) return;
    std::vector<Mat44> m;
    toMats(r, xfs, m);
    r->SetPose(RVec3::sZero(), m.data());
    r->SetLinearAndAngularVelocity(Vec3::sZero(), Vec3::sZero());
}
void ragdollSetLimp(uint32_t h, bool limp) {
    Ragdoll* r = ragdoll(h);
    if (!r) return;
    BodyInterface& bi = sys->GetBodyInterface();
    for (size_t i = 0; i < r->GetBodyCount(); i++) {
        BodyID id = r->GetBodyID((int)i);
        Vec3 v = bi.GetLinearVelocity(id), w = bi.GetAngularVelocity(id);
        bi.SetMotionType(id, limp ? EMotionType::Dynamic : EMotionType::Kinematic, EActivation::Activate);
        if (limp) bi.SetLinearAndAngularVelocity(id, v, w);  // keep the animation's motion into the fall
    }
}
void ragdollGetPose(uint32_t h, Xf* xfs) {
    Ragdoll* r = ragdoll(h);
    if (!r) return;
    RVec3 root;
    std::vector<Mat44> m(r->GetBodyCount());
    r->GetPose(root, m.data());
    for (size_t i = 0; i < m.size(); i++) {
        m[i].SetTranslation(m[i].GetTranslation() + Vec3(root));
        xfs[i] = fromMat(m[i]);
    }
}
void ragdollImpulse(uint32_t h, int part, const float j[3], const float at[3]) {
    Ragdoll* r = ragdoll(h);
    if (!r || part < 0 || part >= (int)r->GetBodyCount()) return;
    sys->GetBodyInterface().AddImpulse(r->GetBodyID(part), Vec3(j[0], j[1], j[2]), RVec3(at[0], at[1], at[2]));
}
void ragdollAddVelocity(uint32_t h, const float v[3]) {
    if (Ragdoll* r = ragdoll(h)) r->AddLinearVelocity(Vec3(v[0], v[1], v[2]));
}
bool ragdollResting(uint32_t h) {
    Ragdoll* r = ragdoll(h);
    return !r || !r->IsActive();
}
void takeRagdollHits(std::vector<RagdollHit>& out) {
    std::lock_guard lk(hitMu);
    out.swap(hits);
    hits.clear();
}

void spawnPlayer(const float feet[3]) {
    const float radius = 0.3f, cylinder = 1.2f;  // 1.8 m tall
    Ref<CharacterVirtualSettings> s = new CharacterVirtualSettings();
    s->mShape = RotatedTranslatedShapeSettings(Vec3(0, 0.5f * cylinder + radius, 0), Quat::sIdentity(), new CapsuleShape(0.5f * cylinder, radius))
                    .Create().Get();
    s->mMaxSlopeAngle = DegreesToRadians(50.f);
    s->mSupportingVolume = Plane(Vec3::sAxisY(), -radius);
    player = new CharacterVirtual(s, RVec3(feet[0], feet[1], feet[2]), Quat::sIdentity(), 0, sys.get());
}

void movePlayer(float dt, const float wish[2], bool jump, float outFeet[3], bool& grounded) {
    if (!player) return;
    Vec3 g = sys->GetGravity();
    player->UpdateGroundVelocity();
    Vec3 vel = player->GetLinearVelocity(), ground = player->GetGroundVelocity();
    grounded = player->GetGroundState() == CharacterBase::EGroundState::OnGround;
    // horizontal velocity relative to whatever we stand on; carries over between steps (and is
    // already clipped by last step's collisions, so walls kill momentum)
    Vec3 h = vel - ground;
    h.SetY(0);
    if (!grounded) h = Vec3(vel.GetX(), 0, vel.GetZ());
    Vec3 w(wish[0], 0, wish[1]);
    float ws = w.Length();
    // human-ish feet: ~0.4 s to walking pace, ~0.3 s to stop, sharp reversals brake before re-accelerating.
    // In the air there's almost no control: momentum from the takeoff carries the jump.
    const float accel = grounded ? 9.f : 1.2f, brake = grounded ? 12.f : 0.5f, lateral = grounded ? 7.f : 0.5f;
    if (ws > 1e-3f) {
        Vec3 d = w / ws;
        float along = h.Dot(d);
        Vec3 side = h - d * along;
        // velocity across the wish direction bleeds off like a braking foot plant, not an instant pivot
        float sl = side.Length(), sdrop = lateral * dt;
        side = sl > sdrop ? side * ((sl - sdrop) / sl) : Vec3::sZero();
        if (along < 0) along = std::min(0.f, along + brake * dt);  // still moving backwards: stop first
        else if (along < ws) {
            // accelerate harder from a standstill, taper near target speed (legs saturate)
            float t = along / ws;
            along = std::min(ws, along + accel * (1.f - 0.6f * t * t) * dt);
        } else along = std::max(ws, along - brake * dt);
        h = d * along + side;
    } else {
        float l = h.Length(), drop = brake * dt;
        h = l > drop ? h * ((l - drop) / l) : Vec3::sZero();
    }
    Vec3 nv;
    if (grounded && vel.GetY() - ground.GetY() < 0.1f) {
        nv = ground + h;
        if (jump) nv += Vec3(0, std::sqrt(2.f * -g.GetY() * 0.5f), 0);  // 0.5 m apex
    } else {
        nv = Vec3(0, vel.GetY(), 0) + h;
    }
    nv += g * dt;
    player->SetLinearVelocity(nv);
    CharacterVirtual::ExtendedUpdateSettings es;
    es.mWalkStairsStepUp = Vec3(0, 0.45f, 0);
    es.mStickToFloorStepDown = Vec3(0, -0.5f, 0);
    player->ExtendedUpdate(dt, g, es, sys->GetDefaultBroadPhaseLayerFilter(Layers::MOVING), sys->GetDefaultLayerFilter(Layers::MOVING), {}, {}, *temp);
    RVec3 p = player->GetPosition();
    outFeet[0] = (float)p.GetX();
    outFeet[1] = (float)p.GetY();
    outFeet[2] = (float)p.GetZ();
}

}  // namespace physics

// Bodies: static world geometry, dynamic bodies, and their state, impulses and forces.
#include "physics/jolt_internal.h"

namespace physics {
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

bool bodyCom(uint32_t id, float com[3]) {
    BodyLockRead lock(sys->GetBodyLockInterface(), BodyID(id));
    if (!lock.Succeeded()) return false;
    RVec3 c = lock.GetBody().GetCenterOfMassPosition();
    com[0] = (float)c.GetX(); com[1] = (float)c.GetY(); com[2] = (float)c.GetZ();
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
}  // namespace physics

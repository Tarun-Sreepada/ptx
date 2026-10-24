// Ragdolls: a body per bone joined by swing-twist constraints, driven by the animation or limp.
#include "physics/jolt_internal.h"

namespace physics {
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
}  // namespace physics

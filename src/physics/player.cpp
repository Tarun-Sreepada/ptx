// The player: a CharacterVirtual capsule walking, sliding and jumping over the world.
#include "physics/jolt_internal.h"

namespace physics {
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

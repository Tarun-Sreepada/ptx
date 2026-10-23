#include "character.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <string>

namespace character {
namespace {
HeightFn heightAt;
std::vector<Character*> chars;

// rotation part of a joint transform, Gram-Schmidt orthonormalized, and its column scales
void orthoRot(const Xf& g, float R[3][3], float s[3]) {
    float c[3][3];
    for (int k = 0; k < 3; k++) {
        for (int i = 0; i < 3; i++) c[k][i] = g.m[i][k];
        s[k] = sqrtf(c[k][0] * c[k][0] + c[k][1] * c[k][1] + c[k][2] * c[k][2]);
    }
    auto norm = [](float* v) { float l = sqrtf(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); for (int i = 0; i < 3; i++) v[i] = l > 1e-12f ? v[i] / l : (i == 0); };
    float x[3] = {c[0][0], c[0][1], c[0][2]};
    norm(x);
    float d = c[1][0] * x[0] + c[1][1] * x[1] + c[1][2] * x[2];
    float y[3] = {c[1][0] - d * x[0], c[1][1] - d * x[1], c[1][2] - d * x[2]};
    norm(y);
    float z[3] = {x[1] * y[2] - x[2] * y[1], x[2] * y[0] - x[0] * y[2], x[0] * y[1] - x[1] * y[0]};
    for (int i = 0; i < 3; i++) { R[i][0] = x[i]; R[i][1] = y[i]; R[i][2] = z[i]; }
}
void rotY(float yaw, float R[3][3]) {
    float c = cosf(yaw), s = sinf(yaw);
    float r[3][3] = {{c, 0, s}, {0, 1, 0}, {-s, 0, c}};  // +z (glTF forward) turns toward +x
    memcpy(R, r, sizeof r);
}
float dot3(const float* a, const float* b) { return a[0] * b[0] + a[1] * b[1] + a[2] * b[2]; }

// world frames of the physics bones for the current globals (rotation without scale, origin at the joint)
void partFrames(Character& c, std::vector<Xf>& out) {
    out.resize(c.partJoint.size());
    c.partScale.resize(c.partJoint.size() * 3);
    Xf W = worldXf(c);
    float Ry[3][3];
    rotY(c.yaw, Ry);
    for (size_t i = 0; i < out.size(); i++) {
        const Xf& g = c.globals[c.partJoint[i]];
        float R[3][3];
        orthoRot(g, R, &c.partScale[i * 3]);
        float t[3] = {g.m[0][3], g.m[1][3], g.m[2][3]}, tw[3];
        xfPoint(W, t, tw);
        for (int r = 0; r < 3; r++) {
            for (int k = 0; k < 3; k++) out[i].m[r][k] = Ry[r][0] * R[0][k] + Ry[r][1] * R[1][k] + Ry[r][2] * R[2][k];
            out[i].m[r][3] = tw[r];
        }
    }
}

// Physics bones from the skeleton and the mesh: a joint gets a body when enough vertices follow it (dominant
// weight) and its bone is long enough or it's an extremity; small joints ride on the nearest body above them.
// Each capsule is fitted to the vertices of its body.
void buildParts(Character& c, const std::vector<Xf>& G, const std::vector<std::vector<float>>& vpos, const std::vector<std::vector<int>>& vjoint,
                float E) {
    const SkinData& s = c.data->skins[c.skin];
    const int n = (int)s.parent.size();
    std::vector<int> vc(n, 0);
    size_t nv = 0;
    for (auto& vj : vjoint) { for (int j : vj) vc[j]++; nv += vj.size(); }
    std::vector<std::vector<int>> kids(n);
    for (int j = 0; j < n; j++) if (s.parent[j] >= 0) kids[s.parent[j]].push_back(j);
    auto P = [&](int j, int k) { return G[j].m[k][3]; };
    auto dist = [&](int a, int b) { float d = 0; for (int k = 0; k < 3; k++) d += (P(a, k) - P(b, k)) * (P(a, k) - P(b, k)); return sqrtf(d); };
    const int minV = std::max(8, (int)(nv / 400));
    c.jointPart.assign(n, -1);
    c.partJoint.clear();
    std::vector<int> partParent;
    for (int j : s.order) {
        float bone = 0;
        for (int k : kids[j]) bone = std::max(bone, dist(j, k));
        bool part = vc[j] >= minV && (bone >= 0.06f * E || kids[j].empty());
        if (!part) continue;
        int pp = -1;
        for (int a = s.parent[j]; a >= 0 && pp < 0; a = s.parent[a]) pp = c.jointPart[a];
        if (pp < 0 && !c.partJoint.empty()) pp = 0;  // one tree: a second root hangs off the first
        c.jointPart[j] = (int)c.partJoint.size();
        c.partJoint.push_back(j);
        partParent.push_back(pp);
    }
    // every joint's nearest body (itself or above), for the vertices
    std::vector<int> owner(n, -1);
    for (int j : s.order) owner[j] = c.jointPart[j] >= 0 ? c.jointPart[j] : s.parent[j] >= 0 ? owner[s.parent[j]] : -1;
    const int np = (int)c.partJoint.size();
    std::vector<std::vector<const float*>> pv(np);
    for (size_t m = 0; m < vpos.size(); m++)
        for (size_t v = 0; v < vjoint[m].size(); v++)
            if (int o = owner[vjoint[m][v]]; o >= 0) pv[o].push_back(&vpos[m][v * 3]);
    std::vector<std::vector<int>> childParts(np);
    for (int i = 1; i < np; i++) if (partParent[i] >= 0) childParts[partParent[i]].push_back(i);

    c.parts.assign(np, {});
    for (int i = 0; i < np; i++) {
        int j = c.partJoint[i];
        float p0[3] = {P(j, 0), P(j, 1), P(j, 2)}, end[3];
        if (childParts[i].size() == 1) for (int k = 0; k < 3; k++) end[k] = P(c.partJoint[childParts[i][0]], k);
        else {
            float cen[3] = {0, 0, 0}, d[3];
            if (!childParts[i].empty()) {
                for (int ch : childParts[i]) for (int k = 0; k < 3; k++) cen[k] += P(c.partJoint[ch], k) / childParts[i].size();
            } else {
                for (const float* v : pv[i]) for (int k = 0; k < 3; k++) cen[k] += v[k] / std::max<size_t>(pv[i].size(), 1);
            }
            for (int k = 0; k < 3; k++) d[k] = cen[k] - p0[k];
            float l = sqrtf(dot3(d, d));
            if (l < 0.02f * E) {  // no direction from the vertices: continue the parent bone
                int pj = partParent[i] >= 0 ? c.partJoint[partParent[i]] : -1;
                for (int k = 0; k < 3; k++) d[k] = pj >= 0 ? p0[k] - P(pj, k) : (k == 1);
                l = std::max(sqrtf(dot3(d, d)), 1e-6f);
            }
            for (int k = 0; k < 3; k++) d[k] /= l;
            float len = 0;
            for (const float* v : pv[i]) { float r[3] = {v[0] - p0[0], v[1] - p0[1], v[2] - p0[2]}; len = std::max(len, dot3(r, d)); }
            if (!childParts[i].empty()) len = std::max(len * 0.6f, l);
            len = std::clamp(len, 0.03f * E, 0.5f * E);
            for (int k = 0; k < 3; k++) end[k] = p0[k] + d[k] * len;
        }
        float ax[3] = {end[0] - p0[0], end[1] - p0[1], end[2] - p0[2]}, len = sqrtf(dot3(ax, ax));
        float dn[3] = {0, 1, 0};
        if (len > 1e-6f) for (int k = 0; k < 3; k++) dn[k] = ax[k] / len;
        // radius: a high quantile of the vertices' distance to the bone segment
        std::vector<float> dd;
        for (const float* v : pv[i]) {
            float r[3] = {v[0] - p0[0], v[1] - p0[1], v[2] - p0[2]};
            float t = std::clamp(dot3(r, dn), 0.f, len), q[3] = {r[0] - dn[0] * t, r[1] - dn[1] * t, r[2] - dn[2] * t};
            dd.push_back(sqrtf(dot3(q, q)));
        }
        float rad = 0.04f * E;
        if (!dd.empty()) { std::nth_element(dd.begin(), dd.begin() + dd.size() * 6 / 10, dd.end()); rad = dd[dd.size() * 6 / 10]; }
        rad = std::clamp(rad, 0.015f * E, 0.45f * len + 0.01f * E);
        float inset = std::min(0.5f * rad, 0.3f * len);
        float a[3], b[3];
        for (int k = 0; k < 3; k++) { a[k] = p0[k] + dn[k] * inset; b[k] = end[k] - dn[k] * inset; }
        // into the bone's frame (orthonormal rotation at the joint)
        float R[3][3], sc[3];
        orthoRot(G[j], R, sc);
        physics::RagdollPart& rp = c.parts[i];
        rp.parent = partParent[i];
        for (int k = 0; k < 3; k++) {
            rp.a[k] = R[0][k] * (a[0] - p0[0]) + R[1][k] * (a[1] - p0[1]) + R[2][k] * (a[2] - p0[2]);
            rp.b[k] = R[0][k] * (b[0] - p0[0]) + R[1][k] * (b[1] - p0[1]) + R[2][k] * (b[2] - p0[2]);
        }
        rp.radius = rad;
        bool branch = rp.parent >= 0 && childParts[rp.parent].size() >= 2;  // shoulders, hips, neck: ball joints
        rp.swing = branch ? 1.1f : 0.65f;
        rp.twist = branch ? 0.5f : 0.25f;
    }
}

void animate(Character& c, float dt) {
    const SkinData& s = c.data->skins[c.skin];
    if (c.walkSpeed > 0.f) {  // wander: a slow meander over the terrain
        c.wanderPhase += dt;
        c.yaw += sinf(c.wanderPhase * 0.37f) * sinf(c.wanderPhase * 0.11f + 1.f) * 0.8f * dt;
        c.pos[0] += sinf(c.yaw) * c.walkSpeed * dt;
        c.pos[2] += cosf(c.yaw) * c.walkSpeed * dt;
    }
    if (heightAt) c.pos[1] = heightAt(c.pos[0], c.pos[2]);
    skel::restPose(s, c.pose);
    if (c.clip >= 0) {
        c.t += dt * c.rate;
        skel::sample(c.data->clips[c.clip], c.t, true, c.pose);
    }
    if (c.next >= 0) {
        c.tNext += dt * c.rate;
        c.fade += dt / std::max(c.fadeLen, 1e-3f);
        skel::Pose b;
        skel::restPose(s, b);
        skel::sample(c.data->clips[c.next], c.tNext, true, b);
        if (c.fade >= 1.f) { c.pose.swap(b); c.clip = c.next; c.t = c.tNext; c.next = -1; }
        else skel::blend(c.pose, b, c.fade, c.pose);
    }
    skel::globals(s, c.pose, c.globals);
}

void createRagdoll(Character& c) {
    std::vector<Xf> fr;
    partFrames(c, fr);
    std::vector<physics::RagdollPart> wp = c.parts;  // scaled into world units
    for (size_t i = 0; i < wp.size(); i++) {
        wp[i].xf = fr[i];
        for (int k = 0; k < 3; k++) { wp[i].a[k] *= c.scale; wp[i].b[k] *= c.scale; }
        wp[i].radius *= c.scale;
    }
    c.ragdoll = physics::ragdollCreate(wp.data(), wp.size(), c.mass);
    c.teleport = false;
}
}  // namespace

void init(HeightFn height) { heightAt = std::move(height); }
std::vector<Character*>& all() { return chars; }

Xf worldXf(const Character& c) {
    float R[3][3];
    rotY(c.yaw, R);
    Xf x;
    for (int r = 0; r < 3; r++) {
        for (int k = 0; k < 3; k++) x.m[r][k] = R[r][k] * c.scale;
        x.m[r][3] = c.pos[r] + (r == 1 ? c.footOffset : 0.f);
    }
    return x;
}

int findClip(const ModelData& d, const char* name) {
    auto low = [](std::string s) { for (auto& ch : s) ch = (char)tolower((unsigned char)ch); return s; };
    std::string n = low(name);
    for (size_t i = 0; i < d.clips.size(); i++) if (low(d.clips[i].name).find(n) != std::string::npos) return (int)i;
    return -1;
}

Character* spawn(const ModelData& d, const float pos[3], float yaw, float height) {
    int skin = -1;
    for (auto& m : d.meshes) if (m.skin >= 0) { skin = m.skin; break; }
    if (skin < 0 || skin >= (int)d.skins.size()) return nullptr;
    auto* c = new Character;
    c->data = &d;
    c->skin = skin;
    memcpy(c->pos, pos, 12);
    c->yaw = yaw;
    for (size_t i = 0; i < d.clips.size(); i++) if (d.clips[i].skin == skin) { c->clip = (int)i; break; }
    // the rest pose on the CPU: model-space vertices for the bounds and the ragdoll fit
    const SkinData& s = d.skins[skin];
    skel::Pose rest;
    skel::restPose(s, rest);
    std::vector<Xf> G;
    skel::globals(s, rest, G);
    std::vector<float> M(G.size() * 12);
    skel::skinMatrices(s, G, M.data());
    float mn[3] = {1e30f, 1e30f, 1e30f}, mx[3] = {-1e30f, -1e30f, -1e30f};
    std::vector<std::vector<float>> vpos;
    std::vector<std::vector<int>> vjoint;
    for (auto& m : d.meshes) {
        if (m.skin != skin) continue;
        size_t nv = m.verts.size() / 8;
        vpos.emplace_back(nv * 3);
        vjoint.emplace_back(nv);
        for (size_t v = 0; v < nv; v++) {
            const float* w = &m.skinW[v * 8];
            const float* p = &m.verts[v * 8];
            float o[3] = {0, 0, 0};
            int best = 0;
            for (int k = 0; k < 4; k++) {
                int j = std::min((int)w[4 + k], (int)G.size() - 1);
                if (w[k] > w[best]) best = k;
                const float* q = &M[j * 12];
                for (int r = 0; r < 3; r++) o[r] += w[k] * (q[r * 4] * p[0] + q[r * 4 + 1] * p[1] + q[r * 4 + 2] * p[2] + q[r * 4 + 3]);
            }
            for (int r = 0; r < 3; r++) { vpos.back()[v * 3 + r] = o[r]; mn[r] = std::min(mn[r], o[r]); mx[r] = std::max(mx[r], o[r]); }
            vjoint.back()[v] = std::min((int)w[4 + best], (int)G.size() - 1);
        }
    }
    float h = mx[1] - mn[1], E = std::max({mx[0] - mn[0], h, mx[2] - mn[2]});
    c->scale = height > 0.f ? height / std::max(h, 1e-4f) : h > 20.f ? 0.01f : 1.f;  // > 20 units tall: a centimetre rig
    c->footOffset = -mn[1] * c->scale;
    float hw = h * c->scale;
    c->mass = std::clamp(70.f * hw * hw * hw / (1.75f * 1.75f * 1.75f), 3.f, 300.f);
    buildParts(*c, G, vpos, vjoint, E);
    chars.push_back(c);
    return c;
}

void remove(Character* c) {
    if (c->ragdoll) physics::ragdollDestroy(c->ragdoll);
    chars.erase(std::remove(chars.begin(), chars.end(), c), chars.end());
    delete c;
}

void play(Character& c, int clip, float fade) {
    if (clip < 0 || clip >= (int)c.data->clips.size() || clip == c.clip) return;
    if (fade <= 0.f || c.clip < 0) { c.clip = clip; c.t = 0; c.next = -1; return; }
    c.next = clip;
    c.tNext = 0;
    c.fade = 0;
    c.fadeLen = fade;
}

void ragdoll(Character& c, const float impulse[3], const float at[3], int part) {
    if (!c.ragdoll || c.mode == RAGDOLL) return;
    c.mode = RAGDOLL;
    c.restTime = 0.f;
    physics::ragdollSetLimp(c.ragdoll, true);
    if (impulse && at) physics::ragdollImpulse(c.ragdoll, part, impulse, at);
}

void getUp(Character& c) {
    if (c.mode != RAGDOLL) return;
    if (!c.partJoint.empty()) {  // stand up where the body lies, facing the way it came
        std::vector<Xf> fr(c.partJoint.size());
        physics::ragdollGetPose(c.ragdoll, fr.data());
        c.pos[0] = fr[0].m[0][3];
        c.pos[2] = fr[0].m[2][3];
    }
    c.mode = ANIMATED;
    physics::ragdollSetLimp(c.ragdoll, false);
    c.teleport = true;
}

void preStep(float dt) {
    std::vector<physics::RagdollHit> hits;
    physics::takeRagdollHits(hits);
    for (auto& h : hits)
        for (Character* c : chars)
            if (c->ragdoll == h.ragdoll && c->mode == ANIMATED) {
                // The kinematic body didn't give way, so the rock bounced as off a wall. Redo it as a collision with
                // the whole body (restitution ~0.2): most of the change of velocity carries the body, the struck
                // limb gets a capped extra kick so it folds first.
                float k = 1.2f / (h.mass + c->mass), dv[3], j[3];
                for (int i = 0; i < 3; i++) dv[i] = h.momentum[i] * k;
                float body[3] = {dv[0] * 0.7f, dv[1] * 0.7f, dv[2] * 0.7f};
                for (int i = 0; i < 3; i++) j[i] = dv[i] * 0.3f * c->mass * 0.25f;  // a limb is roughly a quarter of the mass
                ragdoll(*c, j, h.at, h.part);
                physics::ragdollAddVelocity(c->ragdoll, body);
            }
    for (Character* c : chars) {
        if (c->mode != ANIMATED) continue;
        animate(*c, dt);
        if (!c->ragdoll && !c->parts.empty()) { createRagdoll(*c); continue; }
        std::vector<Xf> fr;
        partFrames(*c, fr);
        if (c->teleport) { physics::ragdollTeleport(c->ragdoll, fr.data()); c->teleport = false; }
        else physics::ragdollDrive(c->ragdoll, fr.data(), std::max(dt, 1e-3f));
    }
}

void postStep(float dt) {
    for (Character* c : chars) {
        const SkinData& s = c->data->skins[c->skin];
        if (c->mode == RAGDOLL && c->ragdoll) {
            std::vector<Xf> fr(c->partJoint.size());
            physics::ragdollGetPose(c->ragdoll, fr.data());
            Xf Wi = skel::inverse(worldXf(*c));
            float Ry[3][3];
            rotY(c->yaw, Ry);
            std::vector<Xf> L(s.parent.size());
            for (int j : s.order) {
                int p = c->jointPart[j];
                if (p >= 0) {  // body frame -> model space, the bone's scale back on
                    Xf& g = c->globals[j];
                    float t[3] = {fr[p].m[0][3], fr[p].m[1][3], fr[p].m[2][3]}, tm[3];
                    xfPoint(Wi, t, tm);
                    for (int r = 0; r < 3; r++) {
                        for (int k = 0; k < 3; k++)
                            g.m[r][k] = (Ry[0][r] * fr[p].m[0][k] + Ry[1][r] * fr[p].m[1][k] + Ry[2][r] * fr[p].m[2][k]) * c->partScale[p * 3 + k];
                        g.m[r][3] = tm[r];
                    }
                } else if (s.parent[j] >= 0) {
                    c->globals[j] = xfMul(c->globals[s.parent[j]], skel::localXf(&c->pose[j * 10]));
                }
            }
            c->restTime = physics::ragdollResting(c->ragdoll) ? c->restTime + dt : 0.f;
            if (c->getUpAfter > 0.f && c->restTime > c->getUpAfter) getUp(*c);
        }
        c->skinMats.resize(c->globals.size() * 12);
        skel::skinMatrices(s, c->globals, c->skinMats.data());
        // world bounds: the joints, padded by the largest capsule
        Xf W = worldXf(*c);
        float pad = 0.f;
        for (auto& p : c->parts) pad = std::max(pad, p.radius + std::max(sqrtf(dot3(p.b, p.b)), sqrtf(dot3(p.a, p.a))));
        pad = pad * c->scale + 0.1f;
        for (int k = 0; k < 3; k++) { c->bmin[k] = 1e30f; c->bmax[k] = -1e30f; }
        for (const Xf& g : c->globals) {
            float t[3] = {g.m[0][3], g.m[1][3], g.m[2][3]}, w[3];
            xfPoint(W, t, w);
            for (int k = 0; k < 3; k++) { c->bmin[k] = std::min(c->bmin[k], w[k] - pad); c->bmax[k] = std::max(c->bmax[k], w[k] + pad); }
        }
    }
}
}  // namespace character

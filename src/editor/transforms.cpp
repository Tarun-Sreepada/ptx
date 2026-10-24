// Item transforms (position, Euler angles, scale), grid snapping and cell paths.
#include "editor/editor_internal.h"

namespace editor {

double now() { return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count(); }

// ---------------------------------------------------------------- math
Xf compose(const float p[3], const float r[3], const float s[3]) {
    float cx = cosf(r[0] * DEG), sx = sinf(r[0] * DEG), cy = cosf(r[1] * DEG), sy = sinf(r[1] * DEG), cz = cosf(r[2] * DEG), sz = sinf(r[2] * DEG);
    float R[3][3] = {{cy * cz + sy * sx * sz, -cy * sz + sy * sx * cz, sy * cx},
                     {cx * sz, cx * cz, -sx},
                     {-sy * cz + cy * sx * sz, sy * sz + cy * sx * cz, cy * cx}};
    Xf x;
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) x.m[i][j] = R[i][j] * s[j];
        x.m[i][3] = p[i];
    }
    return x;
}

void decompose(const Xf& x, float p[3], float r[3], float s[3]) {
    float R[3][3];
    for (int j = 0; j < 3; j++) {
        s[j] = sqrtf(x.m[0][j] * x.m[0][j] + x.m[1][j] * x.m[1][j] + x.m[2][j] * x.m[2][j]);
        for (int i = 0; i < 3; i++) R[i][j] = s[j] > 1e-8f ? x.m[i][j] / s[j] : 0.f;
    }
    for (int i = 0; i < 3; i++) p[i] = x.m[i][3];
    r[0] = asinf(std::clamp(-R[1][2], -1.f, 1.f)) / DEG;
    r[1] = atan2f(R[0][2], R[2][2]) / DEG;
    r[2] = atan2f(R[1][0], R[1][1]) / DEG;
}

Xf inverse(const Xf& a) {
    const float(*m)[4] = a.m;
    float det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    float id = 1.f / (fabsf(det) > 1e-20f ? det : 1e-20f);
    Xf r;
    r.m[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * id; r.m[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id; r.m[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id;
    r.m[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * id; r.m[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id; r.m[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * id;
    r.m[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * id; r.m[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * id; r.m[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id;
    for (int i = 0; i < 3; i++) r.m[i][3] = -(r.m[i][0] * m[0][3] + r.m[i][1] * m[1][3] + r.m[i][2] * m[2][3]);
    return r;
}

Xf itemXf(const Item& it) {
    static const float one[3] = {1, 1, 1};
    return compose(it.pos, it.rot, it.kind == K_PRIM ? one : it.scale);
}

float snapv(float v, float g) { return (snap != snapInvert) && g > 0 ? roundf(v / g) * g : v; }

int cellOf(float x) { return (int)floorf(x / stream::CELL); }

std::string cellPath(int cx, int cz) { return host.worldDir + "cells/" + std::to_string(cx) + "_" + std::to_string(cz) + ".glb"; }

}  // namespace editor

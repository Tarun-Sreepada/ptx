#include "destruction/buildings.h"
#include <vector>
#include <unordered_map>
#include <memory>
#include "world/world.h"

std::unordered_map<const MeshData*, MeshParts> meshPartsCache;  // cleared when the streamed set changes

std::vector<std::shared_ptr<Model>> damagedModels;               // keeps override copies' GPU side alive

std::vector<WallBreakReq> pendingWallBreaks;

Xf xfInverse(const Xf& a) {
    const float(*m)[4] = a.m;
    float det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
                m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    float id = 1.f / det;
    Xf r;
    r.m[0][0] = (m[1][1] * m[2][2] - m[1][2] * m[2][1]) * id; r.m[0][1] = (m[0][2] * m[2][1] - m[0][1] * m[2][2]) * id; r.m[0][2] = (m[0][1] * m[1][2] - m[0][2] * m[1][1]) * id;
    r.m[1][0] = (m[1][2] * m[2][0] - m[1][0] * m[2][2]) * id; r.m[1][1] = (m[0][0] * m[2][2] - m[0][2] * m[2][0]) * id; r.m[1][2] = (m[0][2] * m[1][0] - m[0][0] * m[1][2]) * id;
    r.m[2][0] = (m[1][0] * m[2][1] - m[1][1] * m[2][0]) * id; r.m[2][1] = (m[0][1] * m[2][0] - m[0][0] * m[2][1]) * id; r.m[2][2] = (m[0][0] * m[1][1] - m[0][1] * m[1][0]) * id;
    for (int i = 0; i < 3; i++) r.m[i][3] = -(r.m[i][0] * m[0][3] + r.m[i][1] * m[1][3] + r.m[i][2] * m[2][3]);
    return r;
}

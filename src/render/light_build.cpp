#include "render/light_build.h"
#include <vector>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <cstdio>
#include <cstdint>
#include "core/prof.h"
#include "render/lighttree.h"
#include "core/limits.h"
#include "core/math.h"
#include "render/bake.h"
#include "render/lights.h"
#include "scene/camera.h"
#include "scene/model_instances.h"
#include "scene/models.h"

// Emissive triangles of the placed models (not in switched-off buildings) + the active point
// lights -> triangle light buffer and the light BVH (lighttree.h). Rebuilt every frame: the orbs move.
void buildLightTree(const float* pointLights) {
    PROF_FUNC();
    struct Tri { float v[3][3]; float Le[3]; float area, score; };
    static std::vector<Tri> tris;
    tris.clear();
    for (uint32_t ii : emisItems) {
        const DrawItem& di = drawItems[ii];
        for (const EmisTri& e : di.m->meshes[di.mesh].emis) {
            Tri t;
            for (int v = 0; v < 3; v++)
                for (int r = 0; r < 3; r++)
                    t.v[v][r] = di.xf.m[r][0] * e.v[v][0] + di.xf.m[r][1] * e.v[v][1] + di.xf.m[r][2] * e.v[v][2] + di.xf.m[r][3];
            V3 a{t.v[0][0], t.v[0][1], t.v[0][2]}, b{t.v[1][0], t.v[1][1], t.v[1][2]}, c{t.v[2][0], t.v[2][1], t.v[2][2]};
            V3 cr = cross(b - a, c - a), ctr = (a + b + c) * (1.f / 3.f);
            t.area = 0.5f * sqrtf(dot(cr, cr));
            float cp[3] = {ctr.x, ctr.y, ctr.z};
            if (t.area < 1e-8f || lightOff(cp)) continue;  // switched-off building
            memcpy(t.Le, e.Le, 12);
            V3 dc = ctr - cam.pos;
            t.score = (0.2126f * t.Le[0] + 0.7152f * t.Le[1] + 0.0722f * t.Le[2]) * t.area / std::max(dot(dc, dc), 1.f);
            tris.push_back(t);
        }
    }
    allEmittersListed = tris.size() <= MAX_TRI_LIGHTS;
    if (!allEmittersListed) {
        static bool warned = false;
        if (!warned) fprintf(stderr, "lights: %zu emissive triangles, keeping the %u brightest near the camera\n", tris.size(), MAX_TRI_LIGHTS);
        warned = true;
        std::nth_element(tris.begin(), tris.begin() + MAX_TRI_LIGHTS, tris.end(), [](const Tri& a, const Tri& b) { return a.score > b.score; });
        tris.resize(MAX_TRI_LIGHTS);
    }
    numTriLights = (uint32_t)tris.size();
    float* T = (float*)triLightBuf.map;
    std::vector<lighttree::Prim> prims;
    prims.reserve(numLights + numTriLights);
    for (uint32_t i = 0; i < numLights; i++) {
        const float* l = pointLights + i * 8;
        float r = l[3];
        prims.push_back({{l[0] - r, l[1] - r, l[2] - r}, {l[0] + r, l[1] + r, l[2] + r}, 0.2126f * l[4] + 0.7152f * l[5] + 0.0722f * l[6]});
    }
    for (uint32_t i = 0; i < numTriLights; i++) {
        const Tri& t = tris[i];
        V3 a{t.v[0][0], t.v[0][1], t.v[0][2]}, b{t.v[1][0], t.v[1][1], t.v[1][2]}, c{t.v[2][0], t.v[2][1], t.v[2][2]};
        V3 n = normalize(cross(b - a, c - a));
        float rec[16] = {t.v[0][0], t.v[0][1], t.v[0][2], t.Le[0], t.v[1][0], t.v[1][1], t.v[1][2], t.Le[1],
                         t.v[2][0], t.v[2][1], t.v[2][2], t.Le[2], n.x, n.y, n.z, t.area};
        memcpy(T + i * 16, rec, sizeof rec);
        lighttree::Prim p;
        for (int k = 0; k < 3; k++) {
            p.mn[k] = std::min({t.v[0][k], t.v[1][k], t.v[2][k]});
            p.mx[k] = std::max({t.v[0][k], t.v[1][k], t.v[2][k]});
        }
        p.power = (0.2126f * t.Le[0] + 0.7152f * t.Le[1] + 0.0722f * t.Le[2]) * t.area;
        prims.push_back(p);
    }
    static std::vector<lighttree::Node> nodes;
    lighttree::build(prims, nodes);
    numTreeNodes = (uint32_t)nodes.size();
    memcpy(lightTreeBuf.map, nodes.data(), nodes.size() * sizeof(lighttree::Node));
    static std::vector<lighttree::AliasEntry> alias;
    lighttree::buildAlias(prims, alias);
    memcpy(lightAliasBuf.map, alias.data(), alias.size() * sizeof(lighttree::AliasEntry));
}

// Collision shapes: triangle meshes, terrain heightfields, convex hulls and compounds of them.
#include "physics/jolt_internal.h"

namespace physics {
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
}  // namespace physics

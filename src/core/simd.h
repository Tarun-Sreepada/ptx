// Batch kernels over SoA arrays, 8 lanes at a time with AVX2 (scalar tail / fallback).
#pragma once
#include <immintrin.h>

#include <cstddef>
#include <cstdint>

// Frustum test of n world AABBs. planes[i] = (nx, ny, nz, d), inside when dot(n, p) + d >= 0.
// A box is culled when its most-inside corner (the "p-vertex") is outside any plane.
inline void cullAabbs(const float* mnx, const float* mny, const float* mnz, const float* mxx, const float* mxy, const float* mxz,
                      size_t n, const float planes[][4], int np, uint8_t* visible) {
    size_t i = 0;
#ifdef __AVX2__
    for (; i + 8 <= n; i += 8) {
        __m256 bx0 = _mm256_loadu_ps(mnx + i), by0 = _mm256_loadu_ps(mny + i), bz0 = _mm256_loadu_ps(mnz + i);
        __m256 bx1 = _mm256_loadu_ps(mxx + i), by1 = _mm256_loadu_ps(mxy + i), bz1 = _mm256_loadu_ps(mxz + i);
        __m256 in = _mm256_castsi256_ps(_mm256_set1_epi32(-1));
        for (int p = 0; p < np; p++) {
            const float* P = planes[p];
            // the plane's sign picks min or max per axis for all 8 boxes at once
            __m256 px = P[0] >= 0 ? bx1 : bx0, py = P[1] >= 0 ? by1 : by0, pz = P[2] >= 0 ? bz1 : bz0;
            __m256 d = _mm256_fmadd_ps(px, _mm256_set1_ps(P[0]), _mm256_set1_ps(P[3]));
            d = _mm256_fmadd_ps(py, _mm256_set1_ps(P[1]), d);
            d = _mm256_fmadd_ps(pz, _mm256_set1_ps(P[2]), d);
            in = _mm256_and_ps(in, _mm256_cmp_ps(d, _mm256_setzero_ps(), _CMP_GE_OQ));
        }
        int m = _mm256_movemask_ps(in);
        for (int k = 0; k < 8; k++) visible[i + k] = (m >> k) & 1;
    }
#endif
    for (; i < n; i++) {
        bool in = true;
        for (int p = 0; p < np && in; p++) {
            const float* P = planes[p];
            float d = (P[0] >= 0 ? mxx[i] : mnx[i]) * P[0] + (P[1] >= 0 ? mxy[i] : mny[i]) * P[1] + (P[2] >= 0 ? mxz[i] : mnz[i]) * P[2] + P[3];
            in = d >= 0;
        }
        visible[i] = in;
    }
}

// Light importance from the camera: luminance / max(d^2, 1). Used to keep the strongest 256 lights.
inline void lightScores(const float* x, const float* y, const float* z, const float* lum, size_t n, float cx, float cy, float cz, float* out) {
    size_t i = 0;
#ifdef __AVX2__
    __m256 vx = _mm256_set1_ps(cx), vy = _mm256_set1_ps(cy), vz = _mm256_set1_ps(cz), one = _mm256_set1_ps(1);
    for (; i + 8 <= n; i += 8) {
        __m256 dx = _mm256_sub_ps(_mm256_loadu_ps(x + i), vx), dy = _mm256_sub_ps(_mm256_loadu_ps(y + i), vy), dz = _mm256_sub_ps(_mm256_loadu_ps(z + i), vz);
        __m256 d2 = _mm256_fmadd_ps(dx, dx, _mm256_fmadd_ps(dy, dy, _mm256_mul_ps(dz, dz)));
        _mm256_storeu_ps(out + i, _mm256_div_ps(_mm256_loadu_ps(lum + i), _mm256_max_ps(d2, one)));
    }
#endif
    for (; i < n; i++) {
        float dx = x[i] - cx, dy = y[i] - cy, dz = z[i] - cz, d2 = dx * dx + dy * dy + dz * dz;
        out[i] = lum[i] / (d2 > 1 ? d2 : 1);
    }
}

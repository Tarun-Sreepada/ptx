// World-space radiance cache (SHaRC-style): a hash grid of cells, keyed by position (cells grow with distance
// from the camera) and dominant normal axis. Each cell holds the demodulated outgoing radiance of diffuse
// surfaces in it (direct + all further bounces, without the surface's own albedo and emission), and the direct
// part of it on its own. A GI path computes direct light exactly at its first secondary hit and takes only the
// indirect part (total - direct) from the cache there, so a path costs one ray + its shadow rays; a rotating
// 1 in RC_TRAIN of the 8x8 tiles (and paths through cells not cached yet) trace on and add what they find,
// and the cache's feedback through later vertices adds unlimited bounces.
// radcache_resolve.comp folds each frame's sums into the stored value and evicts cells nobody touched.
#ifndef RADCACHE_GLSL
#define RADCACHE_GLSL

#define FLAG_GI_CACHE   256u   // stop paths at cached vertices (and train the cache)
#define FLAG_GI_CHECKER 512u   // trace new GI candidates in half the pixels each frame (ReSTIR reuse fills the rest)
#define FLAG_GI_PROBES 1024u   // secondary vertices inside baked volumes read the probes

#define RC_SIZE (1u << 20)     // entries (power of two), keep in sync with main.cpp RC_ENTRIES
#define RC_STRIDE 16u          // key, age, sum total rgb, sum direct rgb, sum count, value total rgb, value direct rgb, value samples
#define RC_PROBE 8u            // linear probing distance
#define RC_CELL 0.25           // cell size (m) near the camera; doubles every 6 m * 2^level
#define RC_SCALE 1024.0        // fixed point for the atomic sums
#define RC_CLAMP 32.0          // per-sample clamp (keeps the sums from overflowing; fireflies)
#define RC_MIN 4.0             // samples a cell needs before paths stop at it
#define RC_TRAIN 8u            // 1 in RC_TRAIN tiles train the cache each frame

bool rcTrain = true;           // this path traces on past cached cells (gi.comp sets it per tile)

layout(binding = 98, std430) buffer RadCache { uint rc[]; };

uint rcHash(uint x) {
    x ^= x >> 16; x *= 0x7feb352du;
    x ^= x >> 15; x *= 0x846ca68bu;
    return x ^ (x >> 16);
}

// slot = first entry to probe, chk = the key stored in the entry (never 0, which marks a free entry)
void rcKey(vec3 p, vec3 n, out uint slot, out uint chk) {
    float dist = length(p - F.camPos.xyz);
    uint lvl = uint(clamp(floor(log2(1.0 + dist / 6.0)), 0.0, 12.0));
    ivec3 c = ivec3(floor(p / (RC_CELL * float(1u << lvl))));
    vec3 a = abs(n);
    uint ax = a.x > a.y && a.x > a.z ? 0u : (a.y > a.z ? 1u : 2u);
    uint nd = ax * 2u + (n[ax] < 0.0 ? 1u : 0u);
    uint h = rcHash(uint(c.x) ^ rcHash(uint(c.y) ^ rcHash(uint(c.z) ^ rcHash(lvl * 8u + nd))));
    slot = h & (RC_SIZE - 1u);
    chk = max(rcHash(h ^ 0x9e3779b9u), 1u);
}

bool rcLookup(vec3 p, vec3 n, out vec3 L, out vec3 D) {
    L = vec3(0);
    D = vec3(0);
    uint slot, chk;
    rcKey(p, n, slot, chk);
    for (uint i = 0u; i < RC_PROBE; i++) {
        uint e = ((slot + i) & (RC_SIZE - 1u)) * RC_STRIDE;
        uint k = rc[e];
        if (k == 0u) return false;
        if (k != chk) continue;
        if (uintBitsToFloat(rc[e + 15u]) < RC_MIN) return false;
        L = vec3(uintBitsToFloat(rc[e + 9u]), uintBitsToFloat(rc[e + 10u]), uintBitsToFloat(rc[e + 11u]));
        D = vec3(uintBitsToFloat(rc[e + 12u]), uintBitsToFloat(rc[e + 13u]), uintBitsToFloat(rc[e + 14u]));
        return true;
    }
    return false;
}

// u: a random number for stochastic rounding (dim cells would otherwise round to black)
void rcAdd(vec3 p, vec3 n, vec3 L, vec3 D, float u) {
    if (any(isnan(L)) || any(isinf(L)) || any(isnan(D)) || any(isinf(D))) return;
    uvec3 q = uvec3(clamp(L, vec3(0), vec3(RC_CLAMP)) * RC_SCALE + u);
    uvec3 qd = uvec3(clamp(D, vec3(0), vec3(RC_CLAMP)) * RC_SCALE + u);
    uint slot, chk;
    rcKey(p, n, slot, chk);
    for (uint i = 0u; i < RC_PROBE; i++) {
        uint e = ((slot + i) & (RC_SIZE - 1u)) * RC_STRIDE;
        uint prev = atomicCompSwap(rc[e], 0u, chk);
        if (prev != 0u && prev != chk) continue;
        atomicAdd(rc[e + 2u], q.x);
        atomicAdd(rc[e + 3u], q.y);
        atomicAdd(rc[e + 4u], q.z);
        atomicAdd(rc[e + 5u], qd.x);
        atomicAdd(rc[e + 6u], qd.y);
        atomicAdd(rc[e + 7u], qd.z);
        atomicAdd(rc[e + 8u], 1u);
        return;
    }
}
#endif

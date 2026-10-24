// River surface, shaded in screen space over the already lit river bed (composite / RR input), after denoising.
// Published ideas, approximated for speed:
//  - flow: an analytic velocity field instead of a shallow-water solve. Fast laminar core, a shear layer toward
//    the banks, and faster water where the channel narrows / shallows (continuity: u ~ Q / (width * depth)).
//  - surface detail: flow-mapped normals (Vlachos, "Water Flow in Portal 2", SIGGRAPH 2010 course): two noise
//    layers advected along the flow with phases half a period apart, crossfaded so neither stretches for long.
//    Plus a few wind ripples with the deep-water dispersion w^2 = g k (Tessendorf, "Simulating Ocean Water").
//  - bank splashes: short-lived expanding rings with foam, wave-particle style (Yuksel, House & Keyser,
//    "Wave Particles", SIGGRAPH 2007), more of them on the outer bank of bends where the current hits the bank.
//  - optics: Fresnel reflection (one traced ray, reusing the lit screen pixel when the hit is on screen),
//    refraction by a screen-space offset, Beer-Lambert absorption + in-scatter over the water path length.
//  - foam: shallow water (anything under the surface, so rocks and banks both get it), shear and splashes,
//    advected with the flow and thresholded against slow noise so it comes and goes.
// The includer defines vec3 waterScene(ivec2 q): lit radiance of pixel q without aerial perspective.
#ifndef WATER_WATER_GLSL
#define WATER_WATER_GLSL
#include "water/flow.glsl"
#include "water/splashes.glsl"
#include "water/rocks.glsl"
#include "water/shade.glsl"
#endif

# ptx

A Vulkan ray-query path tracer and sandbox: streamed terrain, procedural forests, water, weather, sky and
destruction (fracturing rock, breaking trees, craters).

## Requirements

- Linux, x86-64 (AVX2), a GPU with Vulkan ray query (tested on an RTX 3070)
- CMake 3.20 or newer, Ninja, a C++20 compiler (GCC or Clang)
- Vulkan SDK headers and loader, `glslc` (shaderc), GLFW 3, SDL3
- Optional: the NVIDIA DLSS SDK cloned into `external/dlss` (DLSS Ray Reconstruction; turn it off with
  `-DPTX_DLSS=OFF`), and the CUDA toolkit (only for its NVTX header)

Everything else (Jolt Physics, enkiTS, Dear ImGui, cgltf, stb, VMA, meshoptimizer, bc7enc, miniaudio, Steam Audio,
Tracy, EnTT) is downloaded by CMake on the first configure.

## Build

Configure once, then build on every core:

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build -j$(nproc)
```

In fish, write `(nproc)` instead of `$(nproc)`. Ninja already uses all cores by default; `-j` just makes it explicit.

Shaders compile as part of the build (into `build/shaders/`). New `.cpp` files under `src/` and new shaders under
`shaders/` are picked up automatically: just rebuild. The build writes `build/compile_commands.json`, also linked
at the repository root, for clangd and other editor tooling.

## Run

```sh
./build/ptx
./build/ptx --help    # every command-line option
```

Startup work (vegetation generation, physics, audio, the world map, pipeline compilation) runs in parallel on all
cores but one. Compiled pipelines are cached in `build/pipeline_cache.bin`, keyed by the shader binaries, so later
starts are faster. The old `pipeline_cache.bin` files in the repository root are no longer used and can be deleted.

In the game: F1 shows the key help, Tab opens the editor, F8 / F9 the profiler, F12 takes a screenshot.

Settings and the tree / bush generator sliders are remembered in `settings.ini` in the folder you run from
(delete it to go back to the defaults). Values set by command-line flags aren't saved, and test runs
(`--shot`, `--bench`, `--replay`, ...) or `--defaults` leave the file alone.

Some flags worth knowing (all of them: `--help`):

| Flag | What it does |
| --- | --- |
| `--time <hours>` | runs the day / night cycle from that time of day (`--day-length <min>` sets how fast) |
| `--weather-cycle` | weather changes by itself: clear, fair, cloudy, overcast, rain, storms, fog |
| `--fire <x z radius>` | a fire burning from the start |
| `--rain <0..1>`, `--fog <density>` | weather |
| `--sun <elevation> <azimuth>`, `--night` | sun position |
| `--dlss <0..5>` | DLSS Ray Reconstruction mode |
| `--defaults` | ignore `settings.ini` |

## Source layout

Each folder holds one part of the program; files are kept small and named after what they do.

| Folder | What's in it |
| --- | --- |
| `src/app` | startup, the frame loop, command line, window, settings |
| `src/core` | job system, profiler, SIMD helpers, input, limits and physical constants |
| `src/gpu` | Vulkan context, swapchain, pipelines, descriptors, render graph, acceleration structures, DLSS |
| `src/render` | per-frame uniforms, culling, lights, ground virtual texture, the frame graph passes |
| `src/scene` | camera, terrain and grass meshes, rocks, vegetation instances, placement |
| `src/vegetation/trees` | tree and bush generator: growth (branching rules or space colonization), limb surfaces, foliage, seasons, presets, wood mechanics |
| `src/destruction` | craters, fracture, thrown bodies, tree breaking, impacts |
| `src/sim` | water forces on bodies (`hydro`), splashes, debris, fire, birds / butterflies / fireflies (`critters`), foxes (`wildlife`), footprints and trodden grass, world simulation |
| `src/sky` | sun, moon, stars, lightning, the day / night and season cycle, changing weather |
| `src/world` | terrain functions, world map, streaming, glTF load / save, materials, textures |
| `src/physics` | Jolt Physics wrapper, Voronoi fracture |
| `src/characters` | skinned characters and ragdolls |
| `src/audio` | spatial sound (Steam Audio), synthesized sounds, the ambience (birds, crickets, river, leaves) in `src/scene/ambience.cpp` |
| `src/editor` | in-game editor: cells, picking, asset import and browser, scatter brush, building generator, world map tools |
| `src/ui`, `src/tools` | ImGui panels, benchmarks and test hooks |
| `shaders/passes` | shader entry points (compute, vertex, fragment) |
| `shaders/core`, `terrain`, `materials`, `sky`, `lighting`, `water`, `scene`, `post` | shared GLSL, included by path from `shaders/` |

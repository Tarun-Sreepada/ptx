#pragma once


// ---------------------------------------------------------------- scene / settings
struct Settings {
    int raysFirst = 1, bounces = 2;
    bool rasterPrimary = true;
    bool occlusionCull = true;  // skip raster draws hidden behind last frame's depth (render/occlusion.h)
    bool gpuCull = true;        // trees, plants and rocks culled on the GPU, drawn indirect (render/veg_gpu)
    bool meshShaders = true;    // trees and plants as culled meshlets (needs gpuCull and VK_EXT_mesh_shader)
    // world generation (terrain.h gWorld / gRiver): the seed drives the ground, hills, rivers, trees, plants, rocks, dirt
    int worldSeed = 0;
    float hilliness = 0.35f, riverWidthVar = 0.6f, riverBraids = 0.45f, riverTribs = 0.5f;
    bool restir = true, restirTemporal = true, restirSpatial = true;
    int candidates = 16, spatialSamples = 4;
    float spatialRadius = 20.f;
    bool restirGI = true, giTemporal = true, giSpatial = true;
    bool giEnabled = true;  // off: direct light only (I key)
    bool giCache = true;     // radiance cache ends paths after the first secondary hit (radcache.glsl)
    bool giChecker = true;   // new GI candidates in half the pixels per frame (ReSTIR GI fills the rest)
    bool giQuarter = true;   // checkerboard GI at a quarter: 1 cell in each 2 x 2 traced per frame, in turn (~4 ms saved)
    int giCheckerCell = 8;   // checkerboard square size (px): 8+ skips whole warps (saves time), smaller looks finer
    bool giProbes = true;    // secondary hits in baked volumes read the probes
    int giSpatialSamples = 4;
    float giRadius = 30.f, giMaxM = 20.f;
    bool accumulate = true, denoise = true;
    int historyMax = 32, atrousIters = 4;
    float fireflyClamp = 20.f;
    float sunElev = 25.f, sunAzim = 40.f, sunIntensity = 8.f, sunRadiusDeg = 0.6f, skyIntensity = 1.f;
    // day / night cycle (sky/day_cycle.h): the clock runs and moves the sun; a whole day in dayMinutes real minutes
    bool dayCycle = false;
    float timeOfDay = 10.f, dayMinutes = 24.f, dayOfYear = 172.f;
    // weather that changes by itself (sky/weather_cycle.h): fronts bring clear skies, cloud, rain, storms, fog
    bool weatherCycle = false;
    float weatherChange = 1.f;  // how often it changes (x)
    float fogScale = 20.f;
    bool clouds = true;
    bool groundTextures = true, parallax = true;
    float groundNormal = 1.f, parallaxDist = 12.f;
    int groundMode = 3;  // 0 tiled textures, 1 fractal (generated per pixel, never repeats), 2 hybrid, 3 layered virtual texture
    int vtBudget = 96;   // ground virtual texture tiles (64^2 texels) generated per frame
    float fractalRelief = 1.f, fractalPebbles = 0.6f, fractalFinest = 0.001f;
    bool windMap = true;
    float gustStrength = 0.3f, gustSpreadDeg = 45.f, windWaveSpeed = 12.f, windDamping = 0.7f, windRelax = 1.6f;  // calm, plausible gusts
    float cloudCoverage = 0.45f, cloudDensity = 0.03f, cloudSpeed = 1.f, cloudBase = 1500.f, cloudThickness = 1800.f, cloudShadow = 1.f;
    // cloud genera besides the cumulus slab (clouds.glsl): low stratus / stratocumulus deck, cumulonimbus towers
    // rising out of the cumulus layer to an anvil, high cirrus sheet
    // each genus is a share of the one coverage: amount = coverage * weight, so coverage 0 is a clear sky
    float cumulus = 1.f;
    float stratus = 0.55f, stratusLumpy = 0.6f, stratusBase = 700.f, stratusThick = 350.f;
    float cumulonimbus = 0.55f, cbTop = 10000.f, cirrus = 0.8f, cirrusAlt = 9000.f;
    float cirrusLength = 1.f, cirrusFine = 0.5f, cirrusPuff = 0.5f;  // strand length, strand fineness, strands <-> tufts
    // grass blades (grass.comp): clumping, shape, variation
    float grassClump = 0.65f, grassClumpSize = 0.3f, grassWidth = 1.f, grassCurve = 1.f, grassTwist = 0.5f, grassTilt = 0.35f;
    float grassHeightVar = 0.6f, grassDead = 0.12f, grassColorVar = 1.f;
    // ground detail (ground.glsl)
    float groundWarp = 1.f, groundMottle = 1.f, groundMottleScale = 1.f, darkSpots = 0.4f, darkSpotSize = 1.f;
    float groundGrit = 1.f, groundCavity = 1.f, tileBreakup = 1.f, groundSat = 1.f;
    float bushAmount = 1.f, berryShare = 0.35f;  // bushes per forest edge / meadow, fraction of them berry bushes
    float fovMargin = 20.f, keepRadius = 5.f, offscreenDensity = 0.2f;
    float walkSpeed = 3.5f;   // m/s on foot (sprint: twice)
    float maxSpeed = 30.f;    // m/s: bunny hops gain less and less approaching it (sigmoid), never pass it
    bool speedUncapped = false;
    float speedFov = 20.f;    // deg added to the FOV at maxSpeed (sigmoid in speed; 0 = off)
    float headBob = 0.5f;  // walking: the eye dips at each footfall and sways a little (0 = off)
    bool wildlife = true, critters = true, ambience = true;  // foxes; birds / butterflies / fireflies; birdsong, crickets, river
    float grassDensity = 120.f, grassLength = 1.f, lodStart = 8.f, drawDist = 60.f;
    float craterInflow = 0.25f;
    float riverRainRise = 0.35f;  // river surface rise at full precipitation (m)
    float groundFriction = 0.75f, groundBounce = 0.12f;  // natural ground (soil / grass); other surfaces in groundSurfaceAt
    // world simulation (worldsim.comp): snow above the snow line, avalanches, melt water running to the river
    bool worldSim = true, snowFollowsClouds = true;
    float snowLine = 1500.f, snowfall = 2.f /* m/h at full precipitation */, meltRate = 0.6f /* m/h */;
    float reposeDeg = 38.f, avalanche = 1.f, riverRange = 250.f, dryRate = 0.004f, waterMobility = 1.f, mapTexTile = 2.f;
    float simSpeed = 1.f;   // world sim time scale (fast-forward snow, melt and runoff)
    int snowMeshEvery = 6;  // frames between terrain mesh refreshes while the snow changes  // share of the river discharge into a connected crater
    float viewDist = 800.f;  // world radius drawn: terrain mesh, rocks / trees, streamed cells
    float treeDensity = 0.25f, impostorDist = 110.f;  // impostors from here (+ a 20 m crossfade band)
    float clutter = 1.f, clutterRange = 45.f;      // ground clutter density and distance (ferns, pebbles, twigs, leaf litter)
    int lightCount = 48;
    float orbRange = 60.f;  // width of the tile the orbs drift in around the camera (m)
    float lightIntensity = 1.5f;
    float windStrength = 0.35f, windSpeed = 1.6f, windDirDeg = 30.f;
    float exposure = 1.4f;                          // manual exposure (auto off)
    bool autoExposure = true;
    float evComp = -2.f, adaptUp = 2.5f, adaptDown = 1.2f;  // EV compensation, eye adaptation speed (1/s)
    // physical camera (post.comp): exposure from f-stop, shutter and ISO (EV100) instead of metering; the lens
    // (full-frame thin lens, focal length from the FOV) sets depth of field and aperture vignetting, the ISO the noise
    bool physCam = false;
    float fStop = 8.f, shutterDen = 125.f, iso = 100.f, sceneLum = 11500.f;  // sceneLum: cd/m^2 per scene radiance unit
    bool dof = false, motionBlur = false;
    float focusDist = 0.f;                     // m, 0 = autofocus on the centre
    float mbShutter = 0.5f;                    // fraction of the frame the shutter is open (no physical camera)
    float vignette = 0.35f, chromAb = 0.f, grain = 0.f;
    int tonemapper = 0;                             // 0 AgX, 1 AgX punchy, 2 ACES, 3 none
    float bloom = 0.04f, sharpen = 0.35f;
    int hdrMode = 0;                                // 0 SDR, 1 HDR10 (PQ), 2 scRGB
    float paperWhite = 203.f, peakNits = 1000.f;
    bool grassRefit = true;                          // grass BLAS update in place between rebuilds
    bool fog = true, fogLights = true;               // volumetric fog / god rays (volfog.comp)
    int fogDiv = 8;                                  // fog resolution = render resolution / fogDiv (1..128)
    bool godRays = true;                             // fog steps take a shadow ray to the sun (off: evenly lit fog, cheaper)
    int cloudDiv = 4;                                // sky clouds resolution = render resolution / cloudDiv (cloud.comp; 2 -> 4: ~1 ms)
    float fogDensity = 0.0007f, fogFalloff = 0.08f, fogAniso = 0.6f, fogNoise = 0.6f, fogDist = 150.f, fogBlend = 0.12f;
    float rain = 0.f, wetness = 0.f, puddles = 0.f;  // puddles: the old noise-placed ones; the world sim makes real ones  // weather (surface_fx.comp, post.comp)
    bool rainWets = true;                            // wetness follows the rain (dries slowly after)
    float soak = 0.f, drip = 0.f;                    // water soaked into porous surfaces, still dripping from cover (surface_fx.comp)
    float rivulets = 1.f;                            // streaks of water running down steep faces in rain
    float caustics = 1.f;                            // river-bed caustics strength
    bool decals = true;
    float foamTrail = 20.f;
    float flow = 1.f, clarity = 1.f, foam = 1.f, splash = 1.f, absorb = 1.4f;
    float murk[3] = {0.03f, 0.07f, 0.065f};
    float throwSize = 0.4f, throwSpeed = 12.f;
    int viewMode = 0;
    // surface shading (brdf.glsl, sss.glsl): subsurface random walk, material micro-shadowing / specular occlusion
    bool sss = true, snowSSS = true, microShadow = true;
    int sssSteps = 24;
    float sssScale = 1.f;
    // night sky (night.glsl): moon phase (0 new, 0.5 full), observer latitude, how much brighter than physical the
    // moon / stars / airglow light the scene (eyes adapt further than a display can show), sky features
    float moonPhase = 0.5f, latitude = 45.f, nightBoost = 100.f;
    float stars = 1.f, milkyWay = 1.f, airglow = 1.f, twinkle = 1.f;
    bool purkinje = true;
    // lightning (lightning.glsl): strikes in thunderstorms (rain under cumulonimbus), thunder delayed by distance
    bool lightning = true;
    float strikesPerMin = 3.f;
    // rainbow in rain with the sun behind the viewer; fog banks drifting with the wind (volfog.comp)
    float rainbow = 1.f, dropMM = 1.f;
    bool fogBanks = false;
    float bankDensity = 0.03f, bankScale = 220.f, bankTop = 30.f, bankCover = 0.45f, bankDetail = 1.f;
};

extern Settings S;

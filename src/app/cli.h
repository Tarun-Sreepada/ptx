// Command line: launch options (window, world, screenshots) and test hooks. Every flag is one row of a table in
// cli.cpp; `ptx --help` prints them all.
#pragma once
#include <string>

struct LaunchOptions {
    bool validate = false;        // Vulkan validation layer + debug messenger
    const char* shotPath = nullptr;  // --shot: save a screenshot after shotFrames frames and exit
    int shotFrames = 240, shotSeq = 0;
    bool benchOnly = false;       // --bench: run the benchmark, print it, exit
    std::string worldDir;         // <world>/ with cells/, assets/, terrain/, bake/
};

// Test hooks: things a flag schedules for a given frame (-1 = never)
extern int profFrames;     // --prof N: capture N frames from frame 30, export a Chrome trace at exit
extern bool mapTest;       // --map-test: sculpt a hill and carve a river on the world map at startup
extern int ragdollFrame;   // --ragdoll-at F: every character goes limp at frame F
extern int mapClickFrame;  // --map-click F: drive the world map brush through ImGui input (M, then hold the mouse)
extern int editDemoFrame, editSaveFrame;  // --edit-demo F / --edit-save F

// Parses argv into opts and the engine settings it touches. False: --help was printed (exit).
bool parseArgs(int argc, char** argv, LaunchOptions& opts);

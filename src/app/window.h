// The GLFW window: creation, windowed / borderless / fullscreen switching, output resolution choices.
#pragma once
#include <cstdint>

extern uint32_t resOptions[5][2];  // output resolutions offered in the UI; [4] = the window's own size
extern bool resFollowWindow;       // output resolution = window size (follows resizes)
extern int windowMode;             // 0 windowed (maximised), 1 borderless (fills the screen), 2 exclusive fullscreen
extern int pendingWindowMode;      // set by the UI, applied at the top of the next frame (-1 = none)

// Creates the window (screen-sized). fixedOutput: screenshots / tests keep the default 1080p output.
void createWindow(bool fixedOutput);
void applyWindowMode(int mode);

// Scripted test actions triggered from the command line (cli.h): rock throws, tree-throw tests, the editor demo
// house / save, synthetic map brush input, ragdolls. Each runs at its point in the frame.
#pragma once

void testHooksUploadWindow();  // editor demo house / save (upload window)
void testHooksImGuiInput();    // --map-click: synthetic ImGui input (after the backend's NewFrame)
void testHooksSimulate();      // --throw / --tree-throw (before physics)
void testHooksAfterPhysics();  // --ragdoll-at

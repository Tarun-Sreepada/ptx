// Settings remembered between runs: settings.ini (key = value lines) in the folder ptx runs from. It holds every
// engine setting (Settings S) and the tree / bush generator parameters. Loaded before the command line is applied;
// saved at exit and a few seconds after anything changed. A value set by a command-line flag isn't remembered unless
// it's changed again in the UI. --defaults and test runs (screenshots, benchmarks, replays) don't read or write it.
// New fields of Settings, TreeParams or BushParams go into the tables in settings_io.cpp to be remembered.
#pragma once

void loadSettings();              // S now; the vegetation values wait for applyVegetationSettings
void markFlagsApplied();          // after the command line: what the flags changed stays out of the file
void applyVegetationSettings();   // after trees::defaults(): the remembered tree / bush parameters
void saveSettingsIfChanged(double now);
void saveSettings();
void setSettingsPersistence(bool on);
bool setSetting(const char* nameEqValue);  // "name=value" (settings.ini names); false if no such setting  // off: neither load nor save

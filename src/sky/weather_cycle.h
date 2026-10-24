// Weather that changes by itself (S.weatherCycle): every so often a front passes and brings its weather (clear,
// fair-weather cumulus, cloudy, overcast, rain, a thunderstorm or fog), and the sky, rain, wind and fog ease over to
// it in a few minutes. Mornings are more often foggy and summer afternoons stormier when the day cycle runs.
#pragma once

void advanceWeather(float dt);
const char* weatherName();  // the weather coming in (or here)

// Day and night. With the cycle on, game time runs (a whole day in S.dayMinutes real minutes) and the sun follows
// its real path for the latitude and the day of the year: it rises in the east, culminates in the south (north of
// the equator), sets in the west, higher and longer in summer. The moon's phase advances with the days. With the
// cycle off the sun stays where the sliders put it.
#pragma once

// Advances the time (if the cycle is on) and sets the sun's elevation and azimuth from it.
void advanceDay(float dt);
// The snow line the season brings (m): the valley floor in midwinter, high on the mountains in summer (temperature
// lags the solstices by about a month).
float seasonSnowLine(float dayOfYear);
// Sun elevation / azimuth (degrees) at hour (0..24) on day of year, latitude (degrees).
void sunAt(float hour, float dayOfYear, float latitude, float& elevDeg, float& azimDeg);

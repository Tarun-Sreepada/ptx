#pragma once
#include <string>
#include <cstdint>
#include "gpu/vk.h"

extern bool shotUIReq, shotUIPending, shotSceneReq;

extern uint32_t uiShotW, uiShotH;

extern std::string shotMsg;

extern double shotMsgT;

extern Buffer uiShotBuf;

std::string newShotPath();

void shotDone(const std::string& p, bool ok);

bool writePng(const std::string& p, const uint8_t* px, uint32_t w, uint32_t h, bool bgr);

void saveScenePng();

void saveShot(const char* path);

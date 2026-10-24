// Texture compression: model textures -> BC7 (4x smaller in VRAM and bandwidth than RGBA8), full mip chain
// built on the CPU (gamma-correct for sRGB), cached on disk as KTX2 in texcache/ keyed by a hash of the pixels.
// The first load of a texture encodes it (bc7enc, all cores); later loads read the .ktx2.
// TextureData::rgba stays as it is (editor, glTF export, emissive averages use it); uploads use bc7 when present.
#pragma once
#include "world/world.h"

namespace texcomp {
void init();                  // encoder tables; call once before prepare
void setEnabled(bool on);     // --no-bc7: uploads stay RGBA8 with GPU-generated mips
bool enabled();
// Fills t.bc7 / t.bcMips / t.bcLevels from t.rgba (any thread). No-op when disabled or already done.
void prepare(TextureData& t, const char* cacheDir = "texcache");
}  // namespace texcomp

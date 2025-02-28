#pragma once

#if !defined(__ROMANORENDER_INTEGRATOR)
#define __ROMANORENDER_INTEGRATOR

#include "romanorender/scene.h"

ROMANORENDER_NAMESPACE_BEGIN

using integrator_func = Vec4F(*)(Scene*, uint16_t, uint16_t, uint32_t);

Vec4F integrator_mask(Scene* scene, uint16_t x, uint16_t y, uint32_t sample) noexcept;

Vec4F integrator_ambient_occlusion(Scene* scene, uint16_t x, uint16_t y, uint32_t sample) noexcept;

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_INTEGRATOR) */
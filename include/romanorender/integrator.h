#pragma once

#if !defined(__ROMANORENDER_INTEGRATOR)
#define __ROMANORENDER_INTEGRATOR

#include "romanorender/scene.h"

ROMANORENDER_NAMESPACE_BEGIN

using integrator_func = Vec4F (*)(const Scene*, 
                                  const uint16_t,
                                  const uint16_t,
                                  const uint32_t,
                                  const uint16_t);

ROMANORENDER_API Vec4F integrator_pathtrace(const Scene* scene,
                                            const uint16_t x,
                                            const uint16_t y,
                                            const uint32_t sample,
                                            const uint16_t max_bounces) noexcept;

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_INTEGRATOR) */
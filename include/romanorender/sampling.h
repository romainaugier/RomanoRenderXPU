#pragma once

#if !defined(__ROMANORENDER_Vec3F)
#define __ROMANORENDER_Vec3F

#include "romanorender/maths.h"
#include "romanorender/vec3.h"

ROMANORENDER_NAMESPACE_BEGIN

ROMANORENDER_API Vec3F sample_hemisphere(const Vec3F& hit_normal, const float rx, const float ry) noexcept;

ROMANORENDER_API Vec3F sample_hemisphere_unsafe(const Vec3F& hit_normal, const float rx, const float ry) noexcept;

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_Vec3F) */
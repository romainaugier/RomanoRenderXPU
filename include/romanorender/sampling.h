#pragma once

#if !defined(__ROMANORENDER_SAMPLING)
#define __ROMANORENDER_SAMPLING

#include "romanorender/maths.h"
#include "romanorender/vec3.h"
#include "romanorender/vec2.h"

#include "stdromano/hash.h"
#include "stdromano/vector.h"

ROMANORENDER_NAMESPACE_BEGIN

ROMANORENDER_API Vec3F sample_hemisphere(const Vec3F& hit_normal, const float rx, const float ry) noexcept;

ROMANORENDER_API Vec3F sample_hemisphere_unsafe(const Vec3F& hit_normal, const float rx, const float ry) noexcept;

/* PMJ02 */

ROMANORENDER_API stdromano::Vector<Vec2F> generate_pmj02_samples(const uint32_t num_samples, const uint32_t seed) noexcept;

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_SAMPLING) */
#pragma once

#if !defined(__ROMANORENDER_RAY)
#define __ROMANORENDER_RAY

#include "romanorender/vec3.h"

ROMANORENDER_NAMESPACE_BEGIN

#define INVALID_GEOM_ID 0xFFFFFFFFUL
#define INVALID_PRIM_ID 0xFFFFFFFFUL

struct Ray
{
	Vec3F origin;
	Vec3F direction;
	Vec3F inverse_direction;

	float t;
};

struct Hit
{
	Vec3F pos;
	Vec3F normal;

	uint32_t geomID = INVALID_GEOM_ID;
	uint32_t primID = INVALID_PRIM_ID;
};

struct RayHit
{
	Ray ray;
	Hit hit;
};

struct ShadowRay
{
	Vec3F origin;
	Vec3F direction;
	Vec3F inverse_direction;

	float t;
};

ROMANORENDER_FORCE_INLINE void initialize_ray(RayHit& rayhit, const Vec3F& position, const Vec3F& direction, const float t) noexcept
{
	rayhit.ray.origin = position;
	rayhit.ray.direction = direction;
	rayhit.ray.inverse_direction = rcp_vec3f(direction);
	rayhit.ray.t = t;
}

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_RAY) */
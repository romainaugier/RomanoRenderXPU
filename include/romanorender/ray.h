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

ROMANORENDER_FORCE_INLINE Ray initialize_ray(const Vec3F& position, const Vec3F& direction, const float t) noexcept
{
	Ray ray;
	ray.origin = position;
	ray.direction = direction;
	ray.inverse_direction = 1.0f / direction;
	ray.t = t;
	return ray;
}

ROMANORENDER_FORCE_INLINE RayHit initialize_rayhit(const Vec3F& origin, const Vec3F& direction, const float t) noexcept
{
	RayHit rayhit;
	rayhit.ray.origin = origin;
	rayhit.ray.direction = direction;
	rayhit.ray.inverse_direction = 1.0f / direction;
	rayhit.ray.t = t;
	rayhit.hit.pos = Vec3F(0.0f);
	rayhit.hit.normal = Vec3F(0.0f);
	rayhit.hit.geomID = INVALID_GEOM_ID;
	rayhit.hit.primID = INVALID_PRIM_ID;
	return rayhit;
}

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_RAY) */
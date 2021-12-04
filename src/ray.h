#pragma once

#include "maths.h"
#include "camera.h"
#include "ispc/rand.h"

#include <limits>
#include <stdint.h>

#define INVALID_GEOM_ID -1
#define INVALID_MAT_ID -1

struct Ray
{
	vec3 origin;
	vec3 direction;
	vec3 inverseDirection;

	float t;
};

struct Hit
{
	vec3 pos;
	vec3 normal;

	uint32_t geomID = INVALID_GEOM_ID;
	uint32_t matID = INVALID_MAT_ID;
};

struct RayHit
{
	Ray ray;
	Hit hit;
};

struct ShadowRay
{
	vec3 origin;
	vec3 direction;
	vec3 inverseDirection;

	float t;
};

__forceinline void SetRay(RayHit& rayhit, const vec3& position, const vec3& direction, float t) noexcept
{
	rayhit.ray.origin = position;
	rayhit.ray.direction = direction;
	rayhit.ray.inverseDirection = 1.0f / direction;
	rayhit.ray.t = t;
}

__forceinline void SetPrimaryRay(RayHit& rayhit,
	   							 const Camera& cam,
	   							 const uint32_t x,
	   							 const uint32_t y,
	   							 const uint32_t xres,
	   							 const uint32_t yres,
	   							 const uint64_t sample) noexcept
{
	// Generate random numbers
	constexpr unsigned int floatAddr = 0x2f800004u;
	auto toFloat = float();
	memcpy(&toFloat, &floatAddr, 4);

	float randoms[2];
	int seeds[2] = { x * y * sample + 313, x * y * sample + 432 };
	ispc::randomFloatWangHash(seeds, randoms, toFloat, 2);

	// Generate xyz screen to normalized world coordinates

	// Very simple antialiasing
	const float dx = lerp(-0.5f, 0.5f, randoms[0]);
	const float dy = lerp(-0.5f, 0.5f, randoms[1]);

	// or

	// const float dy = sqrt(-0.5 * log(randoms[1])) * sin(2.0 * PI * randoms[2]);
	// const float dx = sqrt(-0.5 * log(randoms[1])) * cos(2.0 * PI * randoms[2]);

	const float xScreen = (2.0f * (x + dx) / float(xres) - 1.0f) * cam.aspect * cam.scale;
	const float yScreen = (1.0f - 2.0f * (y + dy) / float(yres)) * cam.scale;
	const float zScreen = -1.0f;

	// Transform to camera coordinates
	const vec3 rayDir(xScreen, yScreen, zScreen);
	const vec3 zero(0.0f);
	const vec3 rayPosWorld = transform(zero, cam.transformation_matrix);
	const vec3 rayDirWorld = transform(rayDir, cam.transformation_matrix);
	const vec3 rayDirNorm = normalize(rayDirWorld - rayPosWorld);

	SetRay(rayhit, cam.pos, rayDirNorm, 10000.0f);
}
#pragma once

#include "boundingbox.h"
#include "ispc/intersectors.h"
#include <algorithm>
#include "tbb/tbb.h"

#undef min

using floatAllocator = tbb::cache_aligned_allocator<float>;
using intAllocator   = tbb::cache_aligned_allocator<uint32_t>;

FORCEINLINE ispc::vec3 ToIspcVec(const vec3& v1) noexcept { ispc::vec3 v0; v0.x = v1.x; v0.y = v1.y; v0.z = v1.z; return v0; }

struct alignas(32) Sphere
{
	vec3 center;

	float radius;

	uint32_t id = 0;
	uint32_t matId = 0;

	uint32_t padding[2]; // ensure 32 bytes size

	Sphere(vec3 _center, float _radius, unsigned int _id, unsigned int _mat_id) :
		center(_center),
		radius(_radius),
		id(_id),
		matId(_mat_id) {}
};

struct SpheresN
{
	float* centerX;
	float* centerY;
	float* centerZ;

	float* radius;

	uint32_t* id;
	uint32_t* matId;
};

void AllocateSphereN(SpheresN& spheres, 
					 const size_t spheresCount, 
					 floatAllocator* fAllocator, 
					 intAllocator* iAllocator) noexcept;

void ReleaseSphereN(SpheresN& spheres,
					const size_t spheresCount,
					floatAllocator* fAllocator, 
					intAllocator* iAllocator) noexcept;

bool SphereHit(const Sphere& sphere, RayHit& rayhit) noexcept;
bool SphereHitN(const SpheresN& spheres, RayHit& rayhit, const int start, const int count) noexcept;
bool SphereOcclude(const Sphere& sphere, ShadowRay& shadow) noexcept;
bool SphereOccludeN(const SpheresN& spheres, ShadowRay& shadow, const int start, const int count) noexcept;


BoundingBox SphereBBox(const Sphere& sphere) noexcept;
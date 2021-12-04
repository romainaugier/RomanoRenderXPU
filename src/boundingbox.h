#pragma once

#include "ray.h"
#include <algorithm>

struct BoundingBox
{
	vec3 p0;
	vec3 p1;

	BoundingBox()
	{
		constexpr float minNum = std::numeric_limits<float>::lowest();
		constexpr float maxNum = std::numeric_limits<float>::max();

		p0 = vec3(maxNum);
		p1 = vec3(minNum);
	}

	BoundingBox(const vec3& a, const vec3& b) :
		p0(a),
		p1(b)
	{}
};

bool Slabs(const BoundingBox& bbox, const RayHit& rayhit) noexcept;
bool* Slabs8(const BoundingBox* bbox, const RayHit& rayhit) noexcept;
bool SlabsOcclude(const BoundingBox& bbox, const ShadowRay& shadow) noexcept;

BoundingBox Union(const BoundingBox& x, const BoundingBox& y) noexcept;
vec3 Offset(const BoundingBox& bbox, const vec3& p) noexcept;
float SurfaceArea(const BoundingBox& b) noexcept;
uint8_t MaximumDimension(const BoundingBox& a) noexcept;
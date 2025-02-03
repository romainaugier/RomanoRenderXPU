#include "romanorender/bbox.h"

#include <algorithm>

ROMANORENDER_NAMESPACE_BEGIN

int intersect_bbox(const BBox& bbox, const Ray& ray, float* t_min) noexcept
{
	*t_min = 0.0f;
	float t_max = maths::constants::inf;

	for (int i = 0; i < 3; i++)
	{
		const float inv_d = ray.inverse_direction[i];
		float t0 = (bbox.p0[i] - ray.origin[i]) * inv_d;
		float t1 = (bbox.p1[i] - ray.origin[i]) * inv_d;

		if (inv_d < 0.0f) std::swap(t0, t1);

		*t_min = t0 > *t_min ? t0 : *t_min;
		t_max = t1 < t_max ? t1 : t_max;

		if (t_max <= *t_min) return false;
	}

	return true;
}

ROMANORENDER_NAMESPACE_END
#include "romanorender/bbox.h"

#include <algorithm>

ROMANORENDER_NAMESPACE_BEGIN

int intersect_bbox(const BBox& bbox, const Vec3F& origin, const Vec3F& inverse_direction) 
{
	float t_min = 0.0f;
	float t_max = maths::constants::inf;

	for (int i = 0; i < 3; i++)
	{
		const float inv_d = inverse_direction[i];
		float t0 = (bbox.p0[i] - origin[i]) * inv_d;
		float t1 = (bbox.p1[i] - origin[i]) * inv_d;

		if (inv_d < 0.0f) std::swap(t0, t1);

		t_min = t0 > t_min ? t0 : t_min;
		t_max = t1 < t_max ? t1 : t_max;

		if (t_max <= t_min) return 0;
	}

	return 1;
}

ROMANORENDER_NAMESPACE_END
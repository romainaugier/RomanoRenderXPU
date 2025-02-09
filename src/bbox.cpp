#include "romanorender/bbox.h"

#include "stdromano/logger.h"

#include <algorithm>

ROMANORENDER_NAMESPACE_BEGIN

int intersect_bbox(const BBox& bbox, const Vec3F& origin, const Vec3F& inverse_direction) noexcept
{
	float t1 = (bbox.p0.x - origin.x) * inverse_direction.x;
	float t2 = (bbox.p1.x - origin.x) * inverse_direction.x;
	float tmin = maths::minf(t1, t2);
	float tmax = maths::maxf(t1, t2);

	for(int i = 1; i < 3; i++)
	{
		t1 = (bbox.p0[i] - origin[i]) * inverse_direction[i];
		t2 = (bbox.p1[i] - origin[i]) * inverse_direction[i];
		tmin = maths::minf(t1, t2);
		tmax = maths::maxf(t1, t2);
	}

	return tmax > maths::maxf(tmin, 0.0f);
}

ROMANORENDER_NAMESPACE_END
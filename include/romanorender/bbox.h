#pragma once

#if !defined(__ROMANORENDER_BOUNDINGBOX)
#define __ROMANORENDER_BOUNDINGBOX

#include "romanorender/ray.h"

ROMANORENDER_NAMESPACE_BEGIN

struct BBox
{
	Vec3F p0;
	Vec3F p1;

	BBox()
	{
		p0 = Vec3F(maths::constants::max_float);
		p1 = Vec3F(maths::constants::min_float);
	}

	BBox(const Vec3F& a, const Vec3F& b) :
		p0(a),
		p1(b)
	{}

	float surface_area() const noexcept
	{
		const Vec3F d = this->p1 - this->p0;

		return 2.0f * (d.x * d.y + d.x * d.z + d.y * d.z);
	}

	uint32_t maximum_dimension() const noexcept
	{
		const Vec3F diag = this->p1 - this->p0;

		if(diag.x > diag.y && diag.x > diag.z) return 0;
		else if(diag.y > diag.z) return 1;
		else return 2;
	}

	void union_with(const BBox& other) noexcept
	{
		this->p0 = min_vec3f(this->p0, other.p0);
		this->p1 = max_vec3f(this->p1, other.p1);
	}

	void union_with_vec3(const Vec3F& other) noexcept
	{
		this->p0 = min_vec3f(this->p0, other);
		this->p1 = max_vec3f(this->p1, other);
	}

	Vec3F offset(const Vec3F& p) const noexcept
	{
		Vec3F o = p - this->p0;
		if(this->p1.x > this->p0.x) o.x /= (this->p1.x - this->p0.x);
		if(this->p1.y > this->p0.y) o.y /= (this->p1.y - this->p0.y);
		if(this->p1.z > this->p0.z) o.z /= (this->p1.z - this->p0.z);
		return o;
	}
};

struct BBox4
{
	alignas(16) float p0_x[4];
	alignas(16) float p0_y[4];
	alignas(16) float p0_z[4];
	alignas(16) float p1_x[4];
	alignas(16) float p1_y[4];
	alignas(16) float p1_z[4];

	BBox4()
	{
	}
};

struct BBox8
{
	alignas(16) float p0_x[8];
	alignas(16) float p0_y[8];
	alignas(16) float p0_z[8];
	alignas(16) float p1_x[8];
	alignas(16) float p1_y[8];
	alignas(16) float p1_z[8];

	BBox8()
	{
	}
};

int intersect_bbox(const BBox& bbox, const Ray& ray, float* t_min) noexcept;
int intersect_bbox4(const BBox4& bbox, const Ray& ray, float* t_min) noexcept;
int intersect_bbox8(const BBox8& bbox, const Ray& ray, float* t_min) noexcept;

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_BOUNDINGBOX) */
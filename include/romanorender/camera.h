#pragma once

#if !defined(__ROMANORENDER_CAMERA)
#define __ROMANORENDER_CAMERA

#include "romanorender/mat44.h"

ROMANORENDER_NAMESPACE_BEGIN

class Camera
{
private:
	Mat44F transformation_matrix;

	uint32_t xres;
	uint32_t yres;

	float focal_length;
	float fov;
	float aspect;
	float scale;

	void update() noexcept
	{
		this->fov = 2 * maths::rad2degf(maths::atanf(36.0f / (2 * this->focal_length)));
		this->scale = maths::tanf(maths::deg2radf(this->fov * 0.5f));
	}

public:
	Camera() {}

	Camera(const Vec3F& pos, const Vec3F& lookat, const float focal, const uint32_t xres, const uint32_t yres) :
		xres(xres),
		yres(yres),
		focal_length(focal),
		aspect((float)xres / (float)yres)
	{
		this->transformation_matrix = Mat44F::lookat(pos, lookat);

		this->update();
	}

	ROMANORENDER_FORCE_INLINE void set_xres(const uint32_t xres) noexcept { this->xres = xres; }
	ROMANORENDER_FORCE_INLINE void set_yres(const uint32_t yres) noexcept { this->yres = yres; }
	ROMANORENDER_FORCE_INLINE void set_focal(const float focal) noexcept { this->focal_length = focal; this->update(); }
	ROMANORENDER_FORCE_INLINE void set_transform(const Mat44F& transform) noexcept { this->transformation_matrix = transform; }

	Vec3F get_ray_origin() const noexcept
	{
		return Vec3F(this->transformation_matrix[3], this->transformation_matrix[7], this->transformation_matrix[11]);
	}

	Vec3F get_ray_direction(const uint32_t x, const uint32_t y) const noexcept
	{
		const float px = (2.0f * ((x + 0.5f) / this->xres) - 1.0f) * maths::tanf(this->fov / 2.0f * maths::constants::pi / 180.0f) * this->aspect;
		const float py = (1.0f - 2.0f * ((y + 0.5f) / this->yres) * maths::tanf(this->fov / 2.0f * maths::constants::pi / 180.0f));

		Vec3F direction(px, py, -1.0f);

		return mat44f_mul(this->transformation_matrix, direction);
	}
};

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_CAMERA) */
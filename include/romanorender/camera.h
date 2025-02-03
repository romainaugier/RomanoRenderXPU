#pragma once

#if !defined(__ROMANORENDER_CAMERA)
#define __ROMANORENDER_CAMERA

#include "romanorender/mat44.h"

ROMANORENDER_NAMESPACE_BEGIN

class Camera
{
private:
	Mat44F transformation_matrix;

	float focal_length;
	float fov;
	float aspect;
	float scale;

public:
	Camera() {}

	Camera(Vec3F pos, Vec3F lookat, float focal, int xres, int yres) :
		focal_length(focal),
		aspect((float)xres / (float)yres)
	{
		this->transformation_matrix = Mat44F();

		fov = 2 * maths::rad2degf(maths::atanf(36.0f / (2 * focal_length)));
		scale = maths::tanf(maths::deg2radf(fov * 0.5f));
	}
};

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_CAMERA) */
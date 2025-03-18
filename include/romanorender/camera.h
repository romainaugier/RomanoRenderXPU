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

    Camera(const Vec3F& pos, const Vec3F& lookat, const float focal, const uint32_t xres, const uint32_t yres)
        : xres(xres), yres(yres), focal_length(focal), aspect((float)xres / (float)yres)
    {
        this->transformation_matrix = Mat44F::from_lookat(pos, lookat);

        this->update();
    }

    ROMANORENDER_FORCE_INLINE void set_xres(const uint32_t xres) noexcept
    {
        this->xres = xres;
        this->aspect = (float)this->xres / (float)this->yres;
    }

    ROMANORENDER_FORCE_INLINE void set_yres(const uint32_t yres) noexcept
    {
        this->yres = yres;
        this->aspect = (float)this->xres / (float)this->yres;
    }

    ROMANORENDER_FORCE_INLINE void set_focal(const float focal) noexcept
    {
        this->focal_length = focal;
        this->update();
    }

    ROMANORENDER_FORCE_INLINE void set_transform(const Mat44F& transform) noexcept
    {
        this->transformation_matrix = transform;
    }

    ROMANORENDER_FORCE_INLINE float get_fov() const noexcept { return this->fov; }

    ROMANORENDER_FORCE_INLINE float get_aspect() const noexcept { return this->aspect; }

    ROMANORENDER_FORCE_INLINE const Mat44F& get_transform() const noexcept
    {
        return this->transformation_matrix;
    }

    ROMANORENDER_FORCE_INLINE Vec3F get_ray_origin() const noexcept
    {
        return Vec3F(this->transformation_matrix[3], this->transformation_matrix[7], this->transformation_matrix[11]);
    }

    ROMANORENDER_FORCE_INLINE Vec3F get_ray_direction(const uint32_t x, const uint32_t y) const noexcept
    {
        const float ndc_x = (2.0f * (x + 0.5f) / this->xres - 1.0f) * this->aspect;
        const float ndc_y = 1.0f - 2.0f * (y + 0.5f) / this->yres;

        const float tan_half_fov = maths::tanf(maths::deg2radf(this->fov * 0.5f));
        const float px = ndc_x * tan_half_fov;
        const float py = ndc_y * tan_half_fov;

        Vec3F direction(px, py, -1.0f);

        return normalize_vec3f(mat44f_mul_dir(this->transformation_matrix, direction));
    }
};

class FlyingCamera
{
    Vec3F _pos;
    Vec3F _look;
    Vec3F _up;

    static constexpr float SPEED = 1.0f;
    static constexpr float DEGREES_PER_CURSOR_MOVE = 0.1f;
    static constexpr float MAX_PITCH_ROTATION_DEGREES = 80.0f;

public:
    FlyingCamera();

    void update(const float delta_time_seconds,
                const float delta_cursor_x,
                const float delta_cursor_y,
                const bool move_forward,
                const bool move_left,
                const bool move_backward,
                const bool move_right,
                const bool jump,
                const bool crouch) noexcept;

    Mat44F get_transform() const noexcept;

    void set_transform(const Mat44F& transform) noexcept;
};

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_CAMERA) */
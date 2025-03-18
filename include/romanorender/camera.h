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
public:
    Vec3F position = {0.0f, 0.0f, 0.0f};
    Vec3F forward = {0.0f, 0.0f, -1.0f};
    Vec3F right = {1.0f, 0.0f, 0.0f};
    Vec3F up = {0.0f, 1.0f, 0.0f};
    Vec3F world_up = {0.0f, 1.0f, 0.0f};

    float yaw = -90.0f;
    float pitch = 0.0f;

    float speed = 5.0f;
    float sensivity = 0.1f;

    FlyingCamera() { this->update_vectors(); }

    Mat44F get_transform() const noexcept
    {
        Mat44F view;

        const float tx = -dot_vec3f(this->right, this->position);
        const float ty = -dot_vec3f(this->up, this->position);
        const float tz = dot_vec3f(this->forward, this->position);

        view(0, 0) = this->right.x;
        view(0, 1) = this->right.y;
        view(0, 2) = this->right.z;
        view(0, 3) = tx;

        view(1, 0) = this->up.x;
        view(1, 1) = this->up.y;
        view(1, 2) = this->up.z;
        view(1, 3) = ty;

        view(2, 0) = -this->forward.x;
        view(2, 1) = -this->forward.y;
        view(2, 2) = -this->forward.z;
        view(2, 3) = tz;

        view(3, 0) = 0.0f;
        view(3, 1) = 0.0f;
        view(3, 2) = 0.0f;
        view(3, 3) = 1.0f;

        return view;
    }

    void set_transform(const Mat44F& transform) noexcept
    {
        this->right = { transform(0,0), transform(0,1), transform(0,2) };
        this->up = { transform(1,0), transform(1,1), transform(1,2) };
        this->forward = { -transform(2,0), -transform(2,1), -transform(2,2) };

        float tx = transform(0,3);
        float ty = transform(1,3);
        float tz = transform(2,3);

        this->position = -this->right * tx - this->up * ty + this->forward * tz;

        this->pitch = maths::rad2degf(maths::asinf(this->forward.y));
        this->yaw = maths::rad2degf(maths::atan2f(this->forward.z, this->forward.x));
    }

    void process_keyboard(float deltaTime, bool moveForward, bool moveBackward, bool moveLeft, bool moveRight)
    {
        float velocity = speed * deltaTime;

        if(moveForward)
            position += forward * velocity;
        if(moveBackward)
            position -= forward * velocity;
        if(moveLeft)
            position -= right * velocity;
        if(moveRight)
            position += right * velocity;

        this->update_vectors();
    }

    void process_mouse_movement(float xoffset, float yoffset, bool constrainPitch = true)
    {
        xoffset *= sensivity;
        yoffset *= sensivity;

        this->yaw += xoffset;
        this->pitch += yoffset;

        if(constrainPitch)
        {
            if(this->pitch > 89.0f)
                this->pitch = 89.0f;
            if(this->pitch < -89.0f)
                this->pitch = -89.0f;
        }

        this->update_vectors();
    }

private:
    void update_vectors()
    {
        const float yaw_radians = maths::deg2radf(this->yaw);
        const float pitch_radians = maths::deg2radf(this->pitch);

        this->forward = {maths::cosf(yaw_radians) * maths::cosf(pitch_radians),
                         maths::sinf(pitch_radians),
                         maths::sinf(yaw_radians) * std::cos(pitch_radians)};
        this->forward = normalize_vec3f(forward);

        this->right = normalize_vec3f(cross_vec3f(this->forward, this->world_up));
        this->up = normalize_vec3f(cross_vec3f(this->right, this->forward));
    }
};

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_CAMERA) */
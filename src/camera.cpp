#include "romanorender/camera.h"

#include "stdromano/logger.h"

/* Adapted from: https://github.com/nlguillemot/flythrough_camera */

ROMANORENDER_NAMESPACE_BEGIN

void FlyingCamera::update(const float delta_time_seconds,
                          const float delta_cursor_x,
                          const float delta_cursor_y,
                          const bool move_forward,
                          const bool move_left,
                          const bool move_backward,
                          const bool move_right,
                          const bool jump,
                          const bool crouch) noexcept
{
    const float delta_y = -delta_cursor_y;
    const float delta_x = -delta_cursor_x;

    Vec3F x, y, z, t;
    this->_transform.decompose_xyz(x, y, z);
    t = this->_transform.get_translation();

    if(move_forward || move_backward || move_left || move_right)
    {
        const float x_multiplier = (move_right ? 1.0f : 0.0f) - (move_left ? 1.0f : 0.0f);
        const float z_multiplier = (move_forward ? 1.0f : 0.0f) - (move_backward ? 1.0f : 0.0f);

        const Vec3F move_dir = (x * x_multiplier) + (z * z_multiplier);
        
        if(length2_vec3f(move_dir) > 0.0001f)
        {
            t += normalize_vec3f(move_dir) * SPEED * delta_time_seconds;
        }
    }

    if((jump && !crouch) || (!jump && crouch))
    {
        const float y_multiplier = (jump ? 1.0f : 0.0f) - (crouch ? 1.0f : 0.0f);
        const Vec3F move_dir = y * y_multiplier;

        if(length2_vec3f(move_dir) > 0.0001f)
        {
            t += normalize_vec3f(move_dir) * SPEED * delta_time_seconds;
        }
    }

    if(delta_x != 0)
    {
        const float yaw_radians = maths::deg2radf(-delta_x * DEGREES_PER_CURSOR_MOVE);
        const Mat44F rotation_x = Mat44F::from_axis_angle(y, yaw_radians);

        x = normalize_vec3f(mat44f_mul_dir(rotation_x, x));
        y = normalize_vec3f(mat44f_mul_dir(rotation_x, y));
        z = normalize_vec3f(mat44f_mul_dir(rotation_x, z));
    }

    if(delta_y != 0)
    {
        const float pitch_rads = delta_y * maths::deg2radf(DEGREES_PER_CURSOR_MOVE);
        const Mat44F rotation_y = Mat44F::from_axis_angle(x, pitch_rads);

        x = normalize_vec3f(mat44f_mul_dir(rotation_y, x));
        y = normalize_vec3f(mat44f_mul_dir(rotation_y, y));
        z = normalize_vec3f(mat44f_mul_dir(rotation_y, z));
    }

    this->_transform = Mat44F::from_xyzt(x, y, z, t);
}

Mat44F FlyingCamera::get_transform() const noexcept
{
    return this->_transform;
}

void FlyingCamera::set_transform(const Mat44F& transform) noexcept
{
    this->_transform = transform;
}

ROMANORENDER_NAMESPACE_END
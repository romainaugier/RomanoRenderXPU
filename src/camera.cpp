#include "romanorender/camera.h"

#include "stdromano/logger.h"

/* Adapted from: https://github.com/nlguillemot/flythrough_camera */

ROMANORENDER_NAMESPACE_BEGIN

FlyingCamera::FlyingCamera()
{
    this->_pos = Vec3F(0.0f);
    this->_look = Vec3F(0.0f, 0.0f, -1.0f);
    this->_up = Vec3F(0.0f, 1.0f, 0.0f);
}

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
    const float delta_x = delta_cursor_x;

    const Vec3F across = normalize_vec3f(cross_vec3f(this->_look, this->_up));

    if(move_forward || move_backward || move_left || move_right)
    {
        const float x_multiplier = (move_right ? 1.0f : 0.0f) - (move_left ? 1.0f : 0.0f);
        const float z_multiplier = (move_forward ? 1.0f : 0.0f) - (move_backward ? 1.0f : 0.0f);

        const Vec3F move_dir = (across * x_multiplier) + (this->_look * z_multiplier);
        
        if(length2_vec3f(move_dir) > 0.0001f)
        {
            this->_pos += normalize_vec3f(move_dir) * SPEED * delta_time_seconds;
        }
    }

    if((jump && !crouch) || (!jump && crouch))
    {
        const float y_multiplier = (jump ? 1.0f : 0.0f) - (crouch ? 1.0f : 0.0f);
        const Vec3F y_movement = this->_up * y_multiplier;

        this->_pos += y_movement * SPEED * delta_time_seconds;
    }

    if(delta_x != 0)
    {
        const float yaw_degrees = -delta_x * DEGREES_PER_CURSOR_MOVE;

        const float yaw_radians = maths::deg2radf(yaw_degrees);
        const float yaw_cos = maths::cosf(yaw_radians);
        const float yaw_sin = maths::sinf(yaw_radians);

        const Vec3F up = normalize_vec3f(this->_up);

        const Mat44F rotation_x = Mat44F::from_axis_angle(up, yaw_radians);

        const Vec3F new_look = normalize_vec3f(mat44f_mul_dir(rotation_x, this->_look));
        const Vec3F new_up = normalize_vec3f(mat44f_mul_dir(rotation_x, this->_up));

        this->_look = new_look;
        this->_up = new_up;
    }

    if(delta_y != 0)
    {
        static float current_pitch = 0.0f;
        const float MAX_PITCH = maths::deg2radf(89.0f);
        
        float pitch_rads = -delta_y * maths::deg2radf(DEGREES_PER_CURSOR_MOVE);
        current_pitch += pitch_rads;
        
        if(current_pitch > MAX_PITCH) {
            pitch_rads -= (current_pitch - MAX_PITCH);
            current_pitch = MAX_PITCH;
        }
        else if(current_pitch < -MAX_PITCH) {
            pitch_rads -= (current_pitch + MAX_PITCH);
            current_pitch = -MAX_PITCH;
        }

        const Mat44F y_rotation = Mat44F::from_axis_angle(across, pitch_rads);
        this->_look = normalize_vec3f(mat44f_mul_dir(y_rotation, this->_look));
        this->_up = normalize_vec3f(mat44f_mul_dir(y_rotation, this->_up));
    }

    this->_look = normalize_vec3f(this->_look);
    const Vec3F new_across = normalize_vec3f(cross_vec3f(this->_look, this->_up));
    this->_up = normalize_vec3f(cross_vec3f(new_across, this->_look));
}

Mat44F FlyingCamera::get_transform() const noexcept
{
    return Mat44F::from_lookat(this->_pos, this->_pos + this->_look);
}

void FlyingCamera::set_transform(const Mat44F& transform) noexcept
{
    this->_pos = Vec3F(transform[3], transform[7], transform[11]);
    this->_look = normalize_vec3f(Vec3F(transform[2], transform[6], transform[10]));
    this->_up = normalize_vec3f(Vec3F(transform[1], transform[5], transform[9]));
}

ROMANORENDER_NAMESPACE_END
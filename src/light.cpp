#include "romanorender/light.h"
#include "romanorender/sampling.h"

ROMANORENDER_NAMESPACE_BEGIN

/* Square light */

Vec3F LightSquare::sample_direction(const Vec3F& hit_position, const Vec2F& sample, const Vec3F& hit_normal) const noexcept
{
    const Vec3F sample_pos((sample.x - 0.5f) * this->_size_x, (sample.y - 0.5f) * this->_size_y, 0.0f);
    const Vec3F sample_world = this->get_transform().transform_point(sample_pos);

    return normalize_vec3f(sample_world - hit_position);
}

Vec3F LightSquare::sample_intensity(const float distance) const noexcept
{
    return this->get_color() * this->get_intensity() * maths::sqrf(distance);
}

/* Dome / Environment light */

Vec3F LightDome::sample_direction(const Vec3F& hit_position, const Vec2F& sample, const Vec3F& hit_normal) const noexcept
{
    return sample_hemisphere(hit_normal, sample.x, sample.y);
}

Vec3F LightDome::sample_intensity(const float distance) const noexcept
{
    return this->get_color() * this->get_intensity();
}

/* Distant light */

Vec3F LightDistant::sample_direction(const Vec3F& hit_position, const Vec2F& sample, const Vec3F& hit_normal) const noexcept
{
    const Vec3F min_orientation = -this->_orientation;
	const Vec3F position = min_orientation * 100.0f;
	Vec3F up(0, 1, 0);

	if(maths::absf(dot_vec3f(up, this->_orientation)) > 0.9f)
    {
        up = Vec3F(1, 0, 0);
    }

	const float random_angle = sample.x * 2.0f * maths::constants::pi;
	const Vec3F z = cross_vec3f(min_orientation, up);
	const Vec3F y = cross_vec3f(min_orientation, z);
	const Vec3F rand_pos = z * sample.y * this->_angle * maths::cosf(random_angle) + 
                           y * sample.y * this->_angle * maths::sinf(random_angle) + position;

    return normalize_vec3f(rand_pos - hit_position);
}

Vec3F LightDistant::sample_intensity(const float distance) const noexcept
{
    return this->get_color() * this->get_intensity();
}

/* Circle light (same as square light but unit circle instead) */

Vec3F LightCircle::sample_direction(const Vec3F& hit_position, const Vec2F& sample, const Vec3F& hit_normal) const noexcept
{
    return Vec3F(0.0f);
}

Vec3F LightCircle::sample_intensity(const float distance) const noexcept
{
    return this->get_color() * this->get_intensity() * maths::sqrf(distance);
}

/* Spherical light */

Vec3F LightSpherical::sample_direction(const Vec3F& hit_position, const Vec2F& sample, const Vec3F& hit_normal) const noexcept
{
    return Vec3F(0.0f);
}

Vec3F LightSpherical::sample_intensity(const float distance) const noexcept
{
    return this->get_color() * this->get_intensity() * maths::sqrf(distance);
}

ROMANORENDER_NAMESPACE_END
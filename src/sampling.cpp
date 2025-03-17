#include "romanorender/sampling.h"

ROMANORENDER_NAMESPACE_BEGIN

Vec3F sample_hemisphere(const Vec3F& hit_normal, const float rx, const float ry) noexcept
{
    float signZ = (hit_normal.z >= 0.0f) ? 1.0f : -1.0f;
    float a = -1.0f / (signZ + hit_normal.z);
    float b = hit_normal.x * hit_normal.y * a;
    Vec3F b1 = Vec3F(1.0f + signZ * hit_normal.x * hit_normal.x * a,
                     signZ * b,
                     -signZ * hit_normal.x);
    Vec3F b2 = Vec3F(b, signZ + hit_normal.y * hit_normal.y * a, -hit_normal.y);

    float phi = 2.0f * maths::constants::pi * rx;
    float cosTheta = maths::sqrtf(ry);
    float sinTheta = maths::sqrtf(1.0f - ry);

    return normalize_vec3f(((b1 * maths::cosf(phi) + b2 * maths::sinf(phi)) * cosTheta
                            + hit_normal * sinTheta));
}

Vec3F sample_hemisphere_unsafe(const Vec3F& hit_normal, const float rx, const float ry) noexcept
{
    float a = 1.0f - 2.0f * rx;
    float b = maths::sqrtf(1.0f - a * a);
    float phi = 2.0f * maths::constants::pi * ry;

    return Vec3F(hit_normal.x + b * maths::cosf(phi),
                 hit_normal.y + b * maths::sinf(phi),
                 hit_normal.z + a);
}

ROMANORENDER_NAMESPACE_END
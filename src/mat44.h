#pragma once

#include "vec3.h"

struct mat44
{
    float m[4][4];

    mat44() : m{ { 1.0f, 0.0f, 0.0f, 0.0f },
                { 0.0f, 1.0f, 0.0f, 0.0f },
                { 0.0f, 0.0f, 1.0f, 0.0f },
                { 0.0f, 0.0f, 0.0f, 1.0f } } {}

    mat44(float a, float b, float c, float d,
        float e, float f, float g, float h,
        float i, float j, float k, float l,
        float m, float n, float o, float p)
        : m{ { a, b, c, d }, { e, f, g, h }, { i, j, k, l }, { m, n, o, p } } {};

    const float* operator [] (int i) const { return m[i]; }
    float* operator [] (int i) { return m[i]; }
};


inline mat44 operator*(const mat44& m1, const mat44& m2) noexcept
{
    mat44 temp = mat44();

    const __m128 m1_row1 = _mm_load_ps(m1[0]);
    const __m128 m1_row2 = _mm_load_ps(m1[1]);
    const __m128 m1_row3 = _mm_load_ps(m1[2]);
    const __m128 m1_row4 = _mm_load_ps(m1[3]);

    for (int j = 0; j < 4; j++)
    {
        const __m128 m2_col = _mm_load_ps(m2[j]);

        const __m128 res = _mm_add_ps(
            _mm_add_ps(
                _mm_mul_ps(m1_row1, m2_col),
                _mm_mul_ps(m1_row2, m2_col)
            ),
            _mm_add_ps(
                _mm_mul_ps(m1_row3, m2_col),
                _mm_mul_ps(m1_row4, m2_col)
            )
        );

        _mm_store_ps(temp[j], res);
    }

    return temp;
}

inline void transpose(mat44& m) noexcept
{
    m = mat44(m[0][0], m[1][0], m[2][0], m[3][0],
        m[0][1], m[1][1], m[2][1], m[3][1],
        m[0][2], m[1][2], m[2][2], m[3][2],
        m[0][3], m[1][3], m[2][3], m[3][3]);
}


inline void set_translation(mat44& m, const vec3& t) noexcept
{
    m[0][3] = t.x;
    m[1][3] = t.y;
    m[2][3] = t.z;
}


inline void set_rotation(mat44& m, const vec3& r) noexcept
{
    const float sinThetaX = maths::sin(maths::deg2rad(r.x));
    const float cosThetaX = maths::cos(maths::deg2rad(r.x));
    const float sinThetaY = maths::sin(maths::deg2rad(r.y));
    const float cosThetaY = maths::cos(maths::deg2rad(r.y));
    const float sinThetaZ = maths::sin(maths::deg2rad(r.z));
    const float cosThetaZ = maths::cos(maths::deg2rad(r.z));

    mat44 rotation = mat44();
    rotation[0][0] = cosThetaY * cosThetaZ;
    rotation[0][1] = -cosThetaX * sinThetaZ + sinThetaX * sinThetaY * cosThetaZ;
    rotation[0][2] = sinThetaX * sinThetaZ + cosThetaX * sinThetaY * cosThetaZ;
    rotation[1][0] = cosThetaY * sinThetaZ;
    rotation[1][1] = cosThetaX * cosThetaZ + sinThetaX * sinThetaY * sinThetaZ;
    rotation[1][2] = -sinThetaX * cosThetaZ + cosThetaX * sinThetaY * sinThetaZ;
    rotation[2][0] = -sinThetaY;
    rotation[2][1] = sinThetaX * cosThetaY;
    rotation[2][2] = cosThetaX * cosThetaY;

    m = m * rotation;
}


inline void set_scale(mat44& m, const vec3& s) noexcept
{
    m[0][0] = s.x;
    m[1][1] = s.y;
    m[2][2] = s.z;
}


inline vec3 transform(const vec3& v, const mat44& m) noexcept
{
    float w = v.x * m[3][0] + v.y * m[3][1] + v.z * m[3][2] + m[3][3];

    return vec3((v.x * m[0][0] + v.y * m[0][1] + v.z * m[0][2] + m[0][3]),
        (v.x * m[1][0] + v.y * m[1][1] + v.z * m[1][2] + m[1][3]),
        (v.x * m[2][0] + v.y * m[2][1] + v.z * m[2][2] + m[2][3]));
}

inline vec3 transform_dir(const vec3& v, const mat44& m) noexcept
{
    return vec3(v.x * m[0][0] + v.y * m[0][1] + v.z * m[0][2],
        v.x * m[1][0] + v.y * m[1][1] + v.z * m[1][2],
        v.x * m[2][0] + v.y * m[2][1] + v.z * m[2][2]);
}
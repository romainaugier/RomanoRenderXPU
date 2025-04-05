#include "romanorender/mat44.h"

#include <cstdio>

ROMANORENDER_NAMESPACE_BEGIN

/* Static constructors */

Mat44F Mat44F::zeros() noexcept
{
    Mat44F res;
    std::memset(res.data(), 0, 16 * sizeof(float));
    return res;
}

Mat44F Mat44F::identity() noexcept
{
    Mat44F res;
    std::memset(res.data(), 0, 16 * sizeof(float));
    res(0, 0) = res(1, 1) = res(2, 2) = res(3, 3) = 1.0f;
    return res;
}

Mat44F Mat44F::from_translation(const Vec3F& t) noexcept
{
    return Mat44F(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, t.x, t.y, t.z, 1.0f);
}

Mat44F Mat44F::from_scale(const Vec3F& s) noexcept
{
    return Mat44F(s.x, 0.0f, 0.0f, 0.0f, 0.0f, s.y, 0.0f, 0.0f, 0.0f, 0.0f, s.z, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
}

Mat44F Mat44F::from_rotx(const float rx) noexcept
{
    float c, s;
    maths::sincosf(maths::deg2radf(rx), &s, &c);

    return Mat44F(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, c, s, 0.0f, 0.0f, -s, c, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
}

Mat44F Mat44F::from_roty(const float ry) noexcept
{
    float c, s;
    maths::sincosf(maths::deg2radf(ry), &s, &c);

    return Mat44F(c, 0.0f, -s, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, s, 0.0f, c, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
}

Mat44F Mat44F::from_rotz(const float rz) noexcept
{
    float c, s;
    maths::sincosf(maths::deg2radf(rz), &s, &c);

    return Mat44F(c, s, 0.0f, 0.0f, -s, c, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
}

Mat44F Mat44F::from_axis_angle(const Vec3F& axis, const float angle) noexcept
{
    const Vec3F n = normalize_vec3f(axis);

    float c, s;
    maths::sincosf(maths::deg2radf(angle), &s, &c);

    const float t = 1.0f - c;

    const float x = n.x;
    const float y = n.y;
    const float z = n.z;

    return Mat44F(t * x * x + c,
                  t * x * y + s * z,
                  t * x * z - s * y,
                  0.0f,
                  t * x * y - s * z,
                  t * y * y + c,
                  t * y * z + s * x,
                  0.0f,
                  t * x * z + s * y,
                  t * y * z - s * x,
                  t * z * z + c,
                  0.0f,
                  0.0f,
                  0.0f,
                  0.0f,
                  1.0f);
}

Mat44F Mat44F::from_trs(const Vec3F& translation,
                        const Vec3F& rotation,
                        const Vec3F& scale,
                        const Mat44FTransformOrder_ to,
                        const Mat44FRotationOrder_ ro) noexcept
{
    const Mat44F T = Mat44F::from_translation(translation);
    const Mat44F S = Mat44F::from_scale(scale);

    const Mat44F RX = Mat44F::from_rotx(rotation.x);
    const Mat44F RY = Mat44F::from_roty(rotation.y);
    const Mat44F RZ = Mat44F::from_rotz(rotation.z);

    Mat44F R;

    switch(ro)
    {
    case Mat44FRotationOrder_XYZ:
        R = RZ * RY * RX;
        break;
    case Mat44FRotationOrder_XZY:
        R = RY * RZ * RX;
        break;
    case Mat44FRotationOrder_YXZ:
        R = RZ * RX * RY;
        break;
    case Mat44FRotationOrder_YZX:
        R = RX * RZ * RY;
        break;
    case Mat44FRotationOrder_ZXY:
        R = RY * RX * RZ;
        break;
    case Mat44FRotationOrder_ZYX:
        R = RX * RY * RZ;
        break;
    default:
        R = RZ * RY * RX;
    }

    switch(to)
    {
    case Mat44FTransformOrder_SRT:
        return S * R * T;
    case Mat44FTransformOrder_STR:
        return S * T * R;
    case Mat44FTransformOrder_RST:
        return R * S * T;
    case Mat44FTransformOrder_RTS:
        return R * T * S;
    case Mat44FTransformOrder_TSR:
        return T * S * R;
    case Mat44FTransformOrder_TRS:
        return T * R * S;
    default:
        return T * R * S;
    }
}

Mat44F Mat44F::from_xyzt(const Vec3F& x, const Vec3F& y, const Vec3F& z, const Vec3F& t) noexcept
{
    return Mat44F(x.x,
                  y.x,
                  z.x,
                  0.0f,
                  x.y,
                  y.y,
                  z.y,
                  0.0f,
                  x.z,
                  y.z,
                  z.z,
                  0.0f,
                  t.x,
                  t.y,
                  t.z,
                  1.0f);
}

Mat44F Mat44F::from_lookat(const Vec3F& eye, const Vec3F& target, const Vec3F& up) noexcept
{
    const Vec3F z = normalize_vec3f(Vec3F(eye.x - target.x, eye.y - target.y, eye.z - target.z));

    const Vec3F x = normalize_vec3f(cross_vec3f(up, z));
    const Vec3F y = cross_vec3f(z, x);

    return Mat44F(x.x,
                  y.x,
                  z.x,
                  0.0f,
                  x.y,
                  y.y,
                  z.y,
                  0.0f,
                  x.z,
                  y.z,
                  z.z,
                  0.0f,
                  -dot_vec3f(x, eye),
                  -dot_vec3f(y, eye),
                  -dot_vec3f(z, eye),
                  1.0f);
}

/* Mat Mat mul */

#define SIMD_MAT_MUL 1

Mat44F Mat44F::operator*(const Mat44F& other) const noexcept
{
    Mat44F result = Mat44F::zeros();

#if SIMD_MAT_MUL
    __m128 row0 = _mm_load_ps(&other.m[0][0]);
    __m128 row1 = _mm_load_ps(&other.m[1][0]);
    __m128 row2 = _mm_load_ps(&other.m[2][0]);
    __m128 row3 = _mm_load_ps(&other.m[3][0]);

    for(int i = 0; i < 4; i++)
    {
        __m128 res = _mm_setzero_ps();

        res = _mm_fmadd_ps(_mm_set1_ps(this->m[i][0]), row0, res);
        res = _mm_fmadd_ps(_mm_set1_ps(this->m[i][1]), row1, res);
        res = _mm_fmadd_ps(_mm_set1_ps(this->m[i][2]), row2, res);
        res = _mm_fmadd_ps(_mm_set1_ps(this->m[i][3]), row3, res);

        _mm_store_ps(&result.m[i][0], res);
    }

    return result;
#else
    for(int i = 0; i < 4; i++)
    {
        for(int j = 0; j < 4; j++)
        {
            for(int k = 0; k < 4; k++)
            {
                result.m[i][j] = maths::maddf(this->m[i][k], other.m[k][j], result.m[i][j]);
            }
        }
    }
#endif /* SIMD_MAT_MUL */

    return result;
}

/* Mat Vec mul */

Vec3F Mat44F::transform_point(const Vec3F& point) const noexcept
{
    Vec3F res;

    res.x = this->m[0][0] * point.x + this->m[1][0] * point.y + this->m[2][0] * point.z + this->m[3][0];
    res.y = this->m[0][1] * point.x + this->m[1][1] * point.y + this->m[2][1] * point.z + this->m[3][1];
    res.z = this->m[0][2] * point.x + this->m[1][2] * point.y + this->m[2][2] * point.z + this->m[3][2];

    return res;
}

Vec3F Mat44F::transform_dir(const Vec3F& point) const noexcept
{
    Vec3F res;

    res.x = this->m[0][0] * point.x + this->m[1][0] * point.y + this->m[2][0] * point.z;
    res.y = this->m[0][1] * point.x + this->m[1][1] * point.y + this->m[2][1] * point.z;
    res.z = this->m[0][2] * point.x + this->m[1][2] * point.y + this->m[2][2] * point.z;

    return res;
}

/* Decomposition */

void Mat44F::decomp_translation(Vec3F* t) const noexcept
{
    ROMANORENDER_ASSERT(t != nullptr, "t cannot be a nullptr");

    *t = Vec3F(this->m[3][0], this->m[3][1], this->m[3][2]);
}

void Mat44F::decomp_scale(Vec3F* s) const noexcept
{
    ROMANORENDER_ASSERT(s != nullptr, "t cannot be a nullptr");

    s->x = maths::sqrtf(this->m[0][0] * this->m[0][0] + this->m[0][1] * this->m[0][1] + this->m[0][2] * this->m[0][2]);
    s->y = maths::sqrtf(this->m[1][0] * this->m[1][0] + this->m[1][1] * this->m[1][1] + this->m[1][2] * this->m[1][2]);
    s->z = maths::sqrtf(this->m[2][0] * this->m[2][0] + this->m[2][1] * this->m[2][1] + this->m[2][2] * this->m[2][2]);
}

void Mat44F::decomp_xyzt(Vec3F* x, Vec3F* y, Vec3F* z, Vec3F* t) const noexcept
{
    ROMANORENDER_ASSERT(x != nullptr, "x cannot be a nullptr");
    ROMANORENDER_ASSERT(y != nullptr, "y cannot be a nullptr");
    ROMANORENDER_ASSERT(z != nullptr, "z cannot be a nullptr");

    Vec3F scale;
    this->decomp_scale(&scale);

    *x = Vec3F(this->m[0][0], this->m[0][1], this->m[0][2]);
    *y = Vec3F(this->m[1][0], this->m[1][1], this->m[1][2]);
    *z = Vec3F(this->m[2][0], this->m[2][1], this->m[2][2]);

    if(scale.x > 1e-6f)
    {
        x->x /= scale.x;
        x->y /= scale.x;
        x->z /= scale.x;
    }

    if(scale.y > 1e-6f)
    {
        y->x /= scale.y;
        y->y /= scale.y;
        y->z /= scale.y;
    }

    if(scale.z > 1e-6f)
    {
        z->x /= scale.z;
        z->y /= scale.z;
        z->z /= scale.z;
    }

    if(t != nullptr)
    {
        this->decomp_translation(t);
    }
}

void Mat44F::decomp_euler(Vec3F* angles) const noexcept
{
    ROMANORENDER_ASSERT(angles != nullptr, "angles cannot be a nullptr");

    angles->x = maths::deg2radf(maths::asinf(-this->m[2][1]));
    
    if(maths::cosf(angles->x) > 1e-6f) 
    {
        angles->y = maths::deg2radf(maths::atan2f(this->m[2][0], this->m[2][2]));
        angles->z = maths::deg2radf(maths::atan2f(this->m[0][1], this->m[1][1]));
    } 
    else
    {
        angles->y = maths::deg2radf(maths::atan2f(-this->m[0][2], this->m[0][0]));
        angles->z = 0.0f;
    }
}

void Mat44F::decomp_trs(Vec3F* t, Vec3F* r, Vec3F* s) const noexcept
{
    if(t != nullptr)
    {
        this->decomp_translation(t);
    }

    if(s != nullptr)
    {
        this->decomp_scale(s);
    }

    Mat44F rotation_matrix = *this;

    for(int i = 0; i < 3; i++)
    {
        float inv_scale = s->x > 1e-6f ? maths::rcpf(s->x) : 0.0f;
        rotation_matrix.m[0][i] *= inv_scale;

        inv_scale = s->y > 1e-6f ? maths::rcpf(s->y) : 0.0f;
        rotation_matrix.m[1][i] *= inv_scale;

        inv_scale = s->z > 1e-6f ? maths::rcpf(s->z) : 0.0f;
        rotation_matrix.m[2][i] *= inv_scale;
    }

    rotation_matrix.decomp_euler(r);
}

ROMANORENDER_NAMESPACE_END
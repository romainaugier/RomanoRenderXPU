#include "romanorender/mat44.h"

#include <cstdio>

ROMANORENDER_NAMESPACE_BEGIN

Mat44F Mat44F::from_lookat(const Vec3F& position, const Vec3F& lookat) noexcept
{
    Mat44F res;

    const Vec3F up(0.0f, 1.0f, 0.0f);

    const Vec3F z = normalize_safe_vec3f(position - lookat);
    const Vec3F x = normalize_safe_vec3f(cross_vec3f(up, z));
    const Vec3F y = cross_vec3f(z, x);

    res[0] = x.x;
    res[1] = x.y;
    res[2] = x.z;
    res[3] = position.x;

    res[4] = y.x;
    res[5] = y.y;
    res[6] = y.z;
    res[7] = position.y;

    res[8] = z.x;
    res[9] = z.y;
    res[10] = z.z;
    res[11] = position.z;

    return res;
}

Mat44F
Mat44F::from_trs(const Vec3F& t, const Vec3F& r, const Vec3F& s, const Mat44FTransformOrder_ to, const Mat44FRotationOrder_ ro) noexcept
{
    // Translation Matrix
    Mat44F translation;
    translation[3] = t.x;
    translation[7] = t.y;
    translation[11] = t.z;

    // Rotation Matrix
    const float rx = maths::deg2radf(maths::fmodf(r.x, 360.0f));
    const float ry = maths::deg2radf(maths::fmodf(r.y, 360.0f));
    const float rz = maths::deg2radf(maths::fmodf(r.z, 360.0f));

    Mat44F rotationX(1.0f,
                     0.0f,
                     0.0f,
                     0.0f,
                     0.0f,
                     maths::cosf(rx),
                     -maths::sinf(rx),
                     0.0f,
                     0.0f,
                     maths::sinf(rx),
                     maths::cosf(rx),
                     0.0f,
                     0.0f,
                     0.0f,
                     0.0f,
                     1.0f);
    Mat44F rotationY(maths::cosf(ry),
                     0.0f,
                     maths::sinf(ry),
                     0.0f,
                     0.0f,
                     1.0f,
                     0.0f,
                     0.0f,
                     -maths::sinf(ry),
                     0.0f,
                     maths::cosf(ry),
                     0.0f,
                     0.0f,
                     0.0f,
                     0.0f,
                     1.0f);
    Mat44F rotationZ(maths::cosf(rz),
                     -maths::sinf(rz),
                     0.0f,
                     0.0f,
                     maths::sinf(rz),
                     maths::cosf(rz),
                     0.0f,
                     0.0f,
                     0.0f,
                     0.0f,
                     1.0f,
                     0.0f,
                     0.0f,
                     0.0f,
                     0.0f,
                     1.0f);

    Mat44F rotation;

    switch(ro)
    {
    case Mat44FRotationOrder_XYZ:
        rotation = mat44f_mul(mat44f_mul(rotationX, rotationY), rotationZ);
        break;
    case Mat44FRotationOrder_XZY:
        rotation = mat44f_mul(mat44f_mul(rotationX, rotationZ), rotationY);
        break;
    case Mat44FRotationOrder_YXZ:
        rotation = mat44f_mul(mat44f_mul(rotationY, rotationX), rotationZ);
        break;
    case Mat44FRotationOrder_YZX:
        rotation = mat44f_mul(mat44f_mul(rotationY, rotationZ), rotationX);
        break;
    case Mat44FRotationOrder_ZXY:
        rotation = mat44f_mul(mat44f_mul(rotationZ, rotationX), rotationY);
        break;
    case Mat44FRotationOrder_ZYX:
        rotation = mat44f_mul(mat44f_mul(rotationZ, rotationY), rotationX);
        break;
    }

    // Scale Matrix
    Mat44F scale;
    scale[0] = s.x;
    scale[5] = s.y;
    scale[10] = s.z;

    switch(to)
    {
    case Mat44FTransformOrder_SRT:
        return mat44f_mul(mat44f_mul(scale, rotation), translation);
    case Mat44FTransformOrder_STR:
        return mat44f_mul(mat44f_mul(scale, translation), rotation);
    case Mat44FTransformOrder_RST:
        return mat44f_mul(mat44f_mul(rotation, scale), translation);
    case Mat44FTransformOrder_RTS:
        return mat44f_mul(mat44f_mul(rotation, translation), scale);
    case Mat44FTransformOrder_TSR:
        return mat44f_mul(mat44f_mul(translation, scale), rotation);
    case Mat44FTransformOrder_TRS:
        return mat44f_mul(mat44f_mul(translation, rotation), scale);
    }

    return Mat44F();
}

void Mat44F::debug() const noexcept
{
    std::printf("Mat44F:\n");

    for(uint32_t i = 0; i < 4; i++)
    {
        for(uint32_t j = 0; j < 4; j++)
        {
            std::printf("%f ", this->operator()(i, j));
        }

        std::printf("\n");
    }

    std::printf("\n");
}

Mat44F mat44f_mul(const Mat44F& A, const Mat44F& B) noexcept
{
    const Mat44F B_t = B.transposed();

    Mat44F C;

#if defined(ROMANORENDER_GCC)
#pragma unroll
#endif /* defined(ROMANORENDER_GCC) */
    for(uint32_t i = 0; i < 4; i++)
    {
        __m128 sums = _mm_setzero_ps();
        const __m128 row = _mm_load_ps(std::addressof(A[i * 4]));

        for(uint32_t j = 0; j < 4; j++)
        {
            const __m128 col = _mm_load_ps(std::addressof(B[j * 4]));
            sums = _mm_fmadd_ps(row, col, sums);
        }

        _mm_store_ps(std::addressof(C[i * 4]), sums);
    }

    return C;
}

Vec3F mat44f_mul_point(const float* M, const Vec3F& v) noexcept
{
    return Vec3F(v.x * M[0] + v.y * M[1] + v.z * M[2] + M[3],
                 v.x * M[4] + v.y * M[5] + v.z * M[6] + M[7],
                 v.x * M[8] + v.y * M[9] + v.z * M[10] + M[11]);
}

Vec3F mat44f_mul_point(const Mat44F& M, const Vec3F& v) noexcept
{
    return Vec3F(v.x * M[0] + v.y * M[1] + v.z * M[2] + M[3],
                 v.x * M[4] + v.y * M[5] + v.z * M[6] + M[7],
                 v.x * M[8] + v.y * M[9] + v.z * M[10] + M[11]);
}

Vec3F mat44f_mul_dir(const float* M, const Vec3F& v) noexcept
{
    return Vec3F(v.x * M[0] + v.y * M[1] + v.z * M[2],
                 v.x * M[4] + v.y * M[5] + v.z * M[6],
                 v.x * M[8] + v.y * M[9] + v.z * M[10]);
}

Vec3F mat44f_mul_dir(const Mat44F& M, const Vec3F& v) noexcept
{
    return Vec3F(v.x * M[0] + v.y * M[1] + v.z * M[2],
                 v.x * M[4] + v.y * M[5] + v.z * M[6],
                 v.x * M[8] + v.y * M[9] + v.z * M[10]);
}

ROMANORENDER_NAMESPACE_END
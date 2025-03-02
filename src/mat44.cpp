#include "romanorender/mat44.h"

#include <cstdio>

ROMANORENDER_NAMESPACE_BEGIN

Mat44F Mat44F::lookat(const Vec3F& position, const Vec3F& lookat) noexcept
{
    Mat44F res;
    std::memset(res._data, 0, 16 * sizeof(float));

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

Vec3F mat44f_mul_point(const Mat44F& M, const Vec3F& v) noexcept
{
    return Vec3F(
        v.x * M[0] + v.y * M[4] + v.z * M[8] + M[3],
        v.x * M[1] + v.y * M[5] + v.z * M[9] + M[7],
        v.x * M[2] + v.y * M[6] + v.z * M[10] + M[11]
    );
}

Vec3F mat44f_mul_dir(const Mat44F& M, const Vec3F& v) noexcept
{
    return Vec3F(
        v.x * M[0] + v.y * M[4] + v.z * M[8],
        v.x * M[1] + v.y * M[5] + v.z * M[9],
        v.x * M[2] + v.y * M[6] + v.z * M[10]
    );
}

ROMANORENDER_NAMESPACE_END
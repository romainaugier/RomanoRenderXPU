#include "romanorender/mat44.h"

#include <cstdio>

ROMANORENDER_NAMESPACE_BEGIN

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

ROMANORENDER_NAMESPACE_END
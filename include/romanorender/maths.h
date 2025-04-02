#pragma once

#if !defined(__ROMANORENDER_MATHS)
#define __ROMANORENDER_MATHS

#include "romanorender/romanorender.h"

#include <cstring>
#include <immintrin.h>
#include <limits>


#if defined(ROMANORENDER_GCC) || defined(ROMANORENDER_CLANG)
#include <cmath>
#endif

#define MATHS_NAMESPACE_BEGIN                                                                      \
    namespace maths                                                                                \
    {
#define MATHS_NAMESPACE_END }

#define CONSTANTS_NAMESPACE_BEGIN                                                                  \
    namespace constants                                                                            \
    {
#define CONSTANTS_NAMESPACE_END }

ROMANORENDER_NAMESPACE_BEGIN

MATHS_NAMESPACE_BEGIN

CONSTANTS_NAMESPACE_BEGIN

static constexpr float pi = 3.14159265358979323846f;
static constexpr float two_pi = 2.0f * 3.14159265358979323846f;
static constexpr float pi_over_two = 1.57079632679489661923f;
static constexpr float pi_over_four = 0.785398163397448309616f;
static constexpr float one_over_pi = 0.318309886183790671538f;
static constexpr float two_over_pi = 0.636619772367581343076;

static constexpr float inf = std::numeric_limits<float>::infinity();
static constexpr float neginf = -std::numeric_limits<float>::infinity();

static constexpr float sqrt2 = 1.41421356237309504880f;
static constexpr float one_over_sqrt2 = 0.707106781186547524401f;

static constexpr float e = 2.71828182845904523536f;

static constexpr float log2e = 1.44269504088896340736f;
static constexpr float log10e = 0.434294481903251827651f;
static constexpr float ln2 = 0.693147180559945309417f;
static constexpr float ln10 = 2.30258509299404568402f;

static constexpr float zero = 0.0f;
static constexpr float one = 1.0f;
static constexpr float min_float = std::numeric_limits<float>::lowest();
static constexpr float max_float = std::numeric_limits<float>::max();

static constexpr float flt_large_epsilon = 0.001f;
static constexpr float flt_epsilon = std::numeric_limits<float>::epsilon();

CONSTANTS_NAMESPACE_END

#if defined(ROMANORENDER_WIN)
ROMANORENDER_FORCE_INLINE bool isinff(const float x) noexcept { return _finite(x) == 0; }

ROMANORENDER_FORCE_INLINE bool isnanf(const float x) noexcept { return _isnan(x) != 0; }

ROMANORENDER_FORCE_INLINE bool isfinitef(const float x) noexcept { return _finite(x) != 0; }
#else
ROMANORENDER_FORCE_INLINE bool isinff(const float x) noexcept { return __builtin_isinf(x); }

ROMANORENDER_FORCE_INLINE bool isnanf(const float x) noexcept { return __builtin_isnan(x) != 0; }

ROMANORENDER_FORCE_INLINE bool isfinitef(const float x) noexcept
{
    return __builtin_isfinite(x) != 0;
}
#endif /* defined(ROMANORENDER_WIN) */
ROMANORENDER_FORCE_INLINE float sqrf(const float x) noexcept { return x * x; }

ROMANORENDER_FORCE_INLINE float rcpf(const float x) noexcept
{
    const __m128 a = _mm_set_ss(x);
    const __m128 r = _mm_rcp_ss(a);

#if defined(__AVX2__)
    return _mm_cvtss_f32(_mm_mul_ss(r, _mm_fnmadd_ss(r, a, _mm_set_ss(2.0f))));
#else
    return _mm_cvtss_f32(_mm_mul_ss(r, _mm_sub_ss(_mm_set_ss(2.0f), _mm_mul_ss(r, a))));
#endif
}

ROMANORENDER_FORCE_INLINE float rcp_safef(float a) noexcept { return 1.0f / a; }

ROMANORENDER_FORCE_INLINE float minf(float a, float b) noexcept { return a < b ? a : b; }

ROMANORENDER_FORCE_INLINE float maxf(float a, float b) noexcept { return a > b ? a : b; }

ROMANORENDER_FORCE_INLINE float fitf(float s, float a1, float a2, float b1, float b2) noexcept
{
    return b1 + ((s - a1) * (b2 - b1)) / (a2 - a1);
}

ROMANORENDER_FORCE_INLINE float fit01f(float x, float a, float b) noexcept
{
    return x * (b - a) + a;
}

ROMANORENDER_FORCE_INLINE float lerpf(float a, float b, float t) noexcept
{
    return (1.0f - t) * a + t * b;
}

ROMANORENDER_FORCE_INLINE float
clampf(float n, float lower = constants::zero, float upper = constants::one) noexcept
{
    return maxf(lower, minf(n, upper));
}

ROMANORENDER_FORCE_INLINE float clampzf(float n, float upper = constants::one) noexcept
{
    return maxf(constants::zero, minf(n, upper));
}

ROMANORENDER_FORCE_INLINE float deg2radf(const float deg) noexcept
{
    return deg * constants::pi / 180.0f;
}

ROMANORENDER_FORCE_INLINE float rad2degf(const float rad) noexcept
{
    return rad * 180.0f / constants::pi;
}

ROMANORENDER_FORCE_INLINE float absf(const float x) noexcept { return ::fabsf(x); }

ROMANORENDER_FORCE_INLINE float expf(const float x) noexcept { return ::expf(x); }

ROMANORENDER_FORCE_INLINE float sqrtf(const float x) noexcept { return ::sqrtf(x); }

ROMANORENDER_FORCE_INLINE float rsqrtf(const float x) noexcept
{
    const __m128 a = _mm_set_ss(x);
    __m128 r = _mm_rsqrt_ss(a);
    r = _mm_add_ss(_mm_mul_ss(_mm_set_ss(1.5f), r),
                   _mm_mul_ss(_mm_mul_ss(_mm_mul_ss(a, _mm_set_ss(-0.5f)), r), _mm_mul_ss(r, r)));
    return _mm_cvtss_f32(r);
}

ROMANORENDER_FORCE_INLINE float fmodf(const float x, const float y) noexcept
{
    return ::fmodf(x, y);
}

ROMANORENDER_FORCE_INLINE float logf(const float x) noexcept { return ::logf(x); }

ROMANORENDER_FORCE_INLINE float log2f(const float x) noexcept { return ::log2f(x); }

ROMANORENDER_FORCE_INLINE float log10f(const float x) noexcept { return ::log10f(x); }

ROMANORENDER_FORCE_INLINE float powf(const float x, const float y) noexcept { return ::powf(x, y); }

ROMANORENDER_FORCE_INLINE float floorf(const float x) noexcept { return ::floorf(x); }

ROMANORENDER_FORCE_INLINE float ceilf(const float x) noexcept { return ::ceilf(x); }

ROMANORENDER_FORCE_INLINE float fracf(const float x) noexcept { return x - ::floorf(x); }

ROMANORENDER_FORCE_INLINE float acosf(const float x) noexcept { return ::acosf(x); }

ROMANORENDER_FORCE_INLINE float asinf(const float x) noexcept { return ::asinf(x); }

ROMANORENDER_FORCE_INLINE float atanf(const float x) noexcept { return ::atanf(x); }

ROMANORENDER_FORCE_INLINE float atan2f(const float y, const float x) noexcept
{
    return ::atan2f(y, x);
}

ROMANORENDER_FORCE_INLINE float cosf(const float x) noexcept { return ::cosf(x); }

ROMANORENDER_FORCE_INLINE float sinf(const float x) noexcept { return ::sinf(x); }

ROMANORENDER_FORCE_INLINE float tanf(const float x) noexcept { return ::tanf(x); }

ROMANORENDER_FORCE_INLINE float coshf(const float x) noexcept { return ::coshf(x); }

ROMANORENDER_FORCE_INLINE float sinhf(const float x) noexcept { return ::sinhf(x); }

ROMANORENDER_FORCE_INLINE float tanhf(const float x) noexcept { return ::tanhf(x); }

#if defined(__AVX2__)
ROMANORENDER_FORCE_INLINE float maddf(const float a, const float b, const float c) noexcept
{
    return _mm_cvtss_f32(_mm_fmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)));
}

ROMANORENDER_FORCE_INLINE float msubf(const float a, const float b, const float c) noexcept
{
    return _mm_cvtss_f32(_mm_fmsub_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)));
}

ROMANORENDER_FORCE_INLINE float nmaddf(const float a, const float b, const float c) noexcept
{
    return _mm_cvtss_f32(_mm_fnmadd_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)));
}

ROMANORENDER_FORCE_INLINE float nmsubf(const float a, const float b, const float c) noexcept
{
    return _mm_cvtss_f32(_mm_fnmsub_ss(_mm_set_ss(a), _mm_set_ss(b), _mm_set_ss(c)));
}
#else
ROMANORENDER_FORCE_INLINE float maddf(const float a, const float b, const float c) noexcept
{
    return ::fmaf(a, b, c);
}

ROMANORENDER_FORCE_INLINE float msubf(const float a, const float b, const float c) noexcept
{
    return a * b - c;
}

ROMANORENDER_FORCE_INLINE float nmaddf(const float a, const float b, const float c) noexcept
{
    return -a * b + c;
}

ROMANORENDER_FORCE_INLINE float nmsubf(const float a, const float b, const float c) noexcept
{
    return -a * b - c;
}
#endif

MATHS_NAMESPACE_END

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_MATHS) */
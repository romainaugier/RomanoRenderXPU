#pragma once

#include "defines.cuh"

namespace constants {
    static constexpr float pi = 3.14159265358979323846f;
    static constexpr float two_pi = 3.14159265358979323846f * 2.0f;
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
}

static __device__ __forceinline__ float lerpf(float v0, float v1, float t)
{
    return __fmaf_rn(t, v1, __fmaf_rn(-t, v0, v0));
}

static __device__ __forceinline__ float deg2radf(float degrees) { return degrees / M_PI * 180.0f; }
#pragma once

#include "math.cuh"

/* Filtering */

__device__ float2 sample_gaussian(float u1, float u2) noexcept
{
    float f = __fsqrt_rn(-2.0f * __logf(u1));
    float a = constants::two_pi * u2;

    float sin_a, cos_a;
    __sincosf(a, &sin_a, &cos_a);

    return make_float2(f * cos_a, f * sin_a);
}

/* Heitz - A Low-Distortion Map Between Triangle and Square */
__device__ float2 sample_triangle(float u1, float u2) noexcept
{
    if(u2 > u1)
    {
        u1 *= 0.5f;
        u2 -= u1;
    }
    else
    {
        u2 *= 0.5f;
        u1 -= u2;
    }

    return make_float2(u1, u2);
}

__device__ float2 sample_disk(float2 uv) noexcept
{
	float theta = constants::two_pi * uv.x;
	float r = __fsqrt_rn(uv.y);
	return make_float2(__cosf(theta), __sinf(theta)) * r;
}
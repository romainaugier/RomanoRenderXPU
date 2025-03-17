#pragma once

#include "defines.cuh"

static __device__ __forceinline__ float lerpf(float v0, float v1, float t)
{
    return __fmaf_rn(t, v1, __fmaf_rn(-t, v0, v0));
}

static __device__ __forceinline__ float deg2radf(float degrees) { return degrees / M_PI * 180.0f; }
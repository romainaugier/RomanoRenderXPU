#pragma once

#include <optix.h>

struct RayData
{
    float4 color;

    float3 pos;

    unsigned int seed;
};

typedef union
{
    RayData* ptr;
    uint2 dat;
} Payload;

__forceinline__ __device__ uint2 split_ptr(RayData* ptr)
{
    Payload payload;

    payload.ptr = ptr;

    return payload.dat;
}

__forceinline__ __device__ RayData* merge_ptr(unsigned int p0, unsigned int p1)
{
    Payload payload;

    payload.dat.x = p0;
    payload.dat.y = p1;

    return payload.ptr;
}
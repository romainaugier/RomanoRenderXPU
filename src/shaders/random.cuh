#pragma once

#include "defines.cuh"
#include "float2.cuh"
#include "uint2.cuh"
#include "uint3.cuh"
#include "uint4.cuh"

__forceinline__ __device__ float to_float(uint x) noexcept
{
    constexpr uint tofloat = 0x2F800004U;

    return static_cast<float>(x) * reinterpret_cast<const float&>(tofloat);
}

__forceinline__ __device__ uint wang_hash(uint seed) noexcept
{
    seed = (seed ^ 61UL) ^ (seed >> 16UL);
    seed *= 9U;
    seed = seed ^ (seed >> 4UL);
    seed *= 0x27D4EB2DU;
    seed = seed ^ (seed >> 15UL);
    return 1U + seed;
}

__forceinline__ __device__ uint xorshift32(uint state) noexcept
{
    state ^= state << 13U;
    state ^= state >> 17U;
    state ^= state << 5U;
    return state;
}

__forceinline__ __device__ float xorshit_float(uint state) noexcept
{
    return to_float(xorshift32(wang_hash(state)));
}

__forceinline__ __device__ uint pcg_1d(uint v)
{
    uint state = v * 747796405U + 2891336453U;
    uint word = ((state >> ((state >> 28UL) + 4UL)) ^ state) * 277803737U;
    return (word >> 22UL) ^ word;
}

__forceinline__ __device__ float pcg_1d_float(uint v) { return to_float(pcg_1d(v)); }

__forceinline__ __device__ uint2 pcg_2d(uint2 v)
{
    v = v * 1664525U + 1013904223U;
    v.x += v.y * 1664525U;
    v.y += v.x * 1664525U;
    v = v ^ (v >> 16UL);
    v.x += v.y * 1664525U;
    v.y += v.x * 1664525U;
    v = v ^ (v >> 16UL);
    return v;
}

__forceinline__ __device__ float2 pcg_2d_float(uint2 v)
{
    v = pcg_2d(v);
    return make_float2(to_float(v.x), to_float(v.y));
}

__forceinline__ __device__ uint3 pcg_3d(uint3 v)
{
    v = v * 1664525U + 1013904223U;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v ^= v >> 16U;
    v.x += v.y * v.z;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    return v;
}

__forceinline__ __device__ float3 pcg_3d_float(uint3 v)
{
    v = pcg_3d(v);
    return make_float3(to_float(v.x), to_float(v.y), to_float(v.z));
}

__forceinline__ __device__ uint4 pcg_4d(uint4 v)
{
    v = v * 1664525U + 1013904223U;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    v ^= v >> 16U;
    v.x += v.y * v.w;
    v.y += v.z * v.x;
    v.z += v.x * v.y;
    v.w += v.y * v.z;
    return v;
}

__forceinline__ __device__ float4 pcg_4d_float(uint4 v)
{
    v = pcg_4d(v);
    return make_float4(to_float(v.x), to_float(v.y), to_float(v.z), to_float(v.w));
}

__forceinline__ __device__ uint pcg_random_uint32(uint value) noexcept
{
    ulong state = (ulong)value;
    ulong inc = (value | 1) ^ 0xDA3E39CB94B95BDBULL;
    state = state * 6364136223846793005ULL + inc;
    uint xorshifted = static_cast<uint>(((state >> 18UL) ^ state) >> 27UL);
    uint rot = state >> 59U;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31UL));
}

__forceinline__ __device__ float pcg_random_float(uint state) noexcept
{
    return to_float(pcg_random_uint32(state));
}
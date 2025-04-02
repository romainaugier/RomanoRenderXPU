#pragma once

#include "defines.cuh"

__forceinline__ __device__ uint1 wang_hash(uint1 seed) noexcept
{
    seed = (seed ^ 61UL) ^ (seed >> 16UL);
    seed *= 9UL;
    seed = seed ^ (seed >> 4UL);
    seed *= 0x27D4EB2DUL;
    seed = seed ^ (seed >> 15UL);
    return 1UL + seed;
}

__forceinline__ __device__ uint1 xorshift32(uint1 state) noexcept
{
    state ^= state << 13UL;
    state ^= state >> 17UL;
    state ^= state << 5UL;
    return state;
}

__forceinline__ __device__ float random_float_01(uint1 state) noexcept
{
    constexpr uint1 tofloat = 0x2F800004UL;

    const uint1 x = xorshift32(wang_hash(state));

    return static_cast<float>(x) * reinterpret_cast<const float&>(tofloat);
}

__forceinline__ __device__ float rng(unsigned int& previous)
{
    previous = previous * 1664525UL + 1013904223UL;

    return float(previous & 0x00FFFFFF) / float(0x01000000UL);
}

__forceinline__ __device__ uint1 pcg_random_uint32(uint1 value) noexcept
{
    ulong1 state = (ulong1)value;
    ulong1 inc = (value | 1) ^ 0xDA3E39CB94B95BDBULL;
    state = state * 6364136223846793005ULL + inc;
    uint1 xorshifted = static_cast<uint1>(((state >> 18UL) ^ state) >> 27UL);
    uint1 rot = state >> 59UL;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31UL));
}

__forceinline__ __device__ float pcg_random_float(uint1 state) noexcept
{
    constexpr uint1 tofloat = 0x2F800004UL;

    return static_cast<float>(pcg_random_uint32(state)) * reinterpret_cast<const float&>(tofloat);
}
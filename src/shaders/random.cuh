#pragma once

__forceinline__ __device__ uint wang_hash(uint seed) noexcept
{
    seed = (seed ^ 61u) ^ (seed >> 16u);
    seed *= 9u;
    seed = seed ^ (seed >> 4u);
    seed *= 0x27d4eb2du;
    seed = seed ^ (seed >> 15u);
    return 1u + seed;
}

__forceinline__ __device__ uint xorshift32(uint state) noexcept
{
    state ^= state << 13u;
    state ^= state >> 17u;
    state ^= state << 5u;
    return state;
}

__forceinline__ __device__ float random_float_01(uint state) noexcept
{
    constexpr uint tofloat = 0x2f800004u;

    const uint x = xorshift32(wang_hash(state));

    return static_cast<float>(x) * reinterpret_cast<const float&>(tofloat);
}

__forceinline__ __device__ float rng(unsigned int& previous)
{
    previous = previous * 1664525u + 1013904223u;

    return float(previous & 0x00FFFFFF) / float(0x01000000u);
}
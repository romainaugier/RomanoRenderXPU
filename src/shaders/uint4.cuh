#pragma once

#include "defines.cuh"

__device__ __forceinline__ uint4 operator+(const uint4& vec, const uint4& other) noexcept
{
    return make_uint4(vec.x + other.x, vec.y + other.y, vec.z + other.z, vec.w + other.w);
}

__device__ __forceinline__ uint4 operator+(const uint4& vec, const uint t) noexcept
{
    return make_uint4(vec.x + t, vec.y + t, vec.z + t, vec.w + t);
}

__device__ __forceinline__ uint4 operator-(const uint4& vec, const uint4& other) noexcept
{
    return make_uint4(vec.x - other.x, vec.y - other.y, vec.z - other.z, vec.w - other.w);
}

__device__ __forceinline__ uint4 operator-(const uint4& vec, const uint t) noexcept
{
    return make_uint4(vec.x - t, vec.y - t, vec.z - t, vec.w - t);
}

__device__ __forceinline__ uint4 operator*(const uint4& vec, const uint4& other) noexcept
{
    return make_uint4(vec.x * other.x, vec.y * other.y, vec.z * other.z, vec.w * other.w);
}

__device__ __forceinline__ uint4 operator*(const uint4& vec, const uint t) noexcept
{
    return make_uint4(vec.x * t, vec.y * t, vec.z * t, vec.w * t);
}

__device__ __forceinline__ uint4 operator*(const uint t, const uint4& vec) noexcept
{
    return make_uint4(vec.x * t, vec.y * t, vec.z * t, vec.w * t);
}

__device__ __forceinline__ uint4 operator/(const uint4& vec, const uint4& other) noexcept
{
    return make_uint4(vec.x / other.x, vec.y / other.y, vec.z / other.z, vec.w / other.w);
}

__device__ __forceinline__ uint4 operator/(const uint4& vec, const uint t) noexcept
{
    return make_uint4(vec.x / t, vec.y / t, vec.z / t, vec.w / t);
}

__device__ __forceinline__ uint4 operator/(const uint t, const uint4& vec) noexcept
{
    return make_uint4(t / vec.x, t / vec.y, t / vec.z, t / vec.w);
}

__device__ __forceinline__ uint4 operator~(const uint4 vec) noexcept
{
    return make_uint4(~vec.x, ~vec.y, ~vec.z, ~vec.w);
}

__device__ __forceinline__ uint4 operator&(const uint4& vec, const uint4& other) noexcept
{
    return make_uint4(vec.x & other.x, vec.y & other.y, vec.z & other.z, vec.w & other.w);
}

__device__ __forceinline__ uint4 operator&(const uint4& vec, const uint t) noexcept
{
    return make_uint4(vec.x & t, vec.y & t, vec.z & t, vec.w & t);
}

__device__ __forceinline__ uint4 operator|(const uint4& vec, const uint4& other) noexcept
{
    return make_uint4(vec.x | other.x, vec.y | other.y, vec.z | other.z, vec.w | other.w);
}

__device__ __forceinline__ uint4 operator|(const uint4& vec, const uint t) noexcept
{
    return make_uint4(vec.x | t, vec.y | t, vec.z | t, vec.w | t);
}

__device__ __forceinline__ uint4 operator^(const uint4& vec, const uint4& other) noexcept
{
    return make_uint4(vec.x ^ other.x, vec.y ^ other.y, vec.z ^ other.z, vec.w ^ other.w);
}

__device__ __forceinline__ uint4 operator^(const uint4& vec, const uint t) noexcept
{
    return make_uint4(vec.x ^ t, vec.y ^ t, vec.z ^ t, vec.w ^ t);
}

__device__ __forceinline__ uint4 operator>>(const uint4& vec, const uint t) noexcept
{
    return make_uint4(vec.x >> t, vec.y >> t, vec.z >> t, vec.w >> t);
}

__device__ __forceinline__ uint4 operator<<(const uint4& vec, const uint t) noexcept
{
    return make_uint4(vec.x << t, vec.y << t, vec.z << t, vec.w << t);
}

__device__ __forceinline__ bool operator==(const uint4& v0, const uint4& v1) noexcept
{
    return v0.x == v1.x && v0.y == v1.y && v0.z == v1.z && v0.w == v1.w;
}

__device__ __forceinline__ bool operator>(const uint4& v0, const uint4& v1) noexcept
{
    return v0.x > v1.x && v0.y > v1.y && v0.z > v1.z && v0.w > v1.w;
}

__device__ __forceinline__ bool operator>=(const uint4& v0, const uint4& v1) noexcept
{
    return v0.x >= v1.x && v0.y >= v1.y && v0.z >= v1.z && v0.w >= v1.w;
}

__device__ __forceinline__ bool operator<(const uint4& v0, const uint4& v1) noexcept
{
    return v0.x < v1.x && v0.y < v1.y && v0.z < v1.z && v0.w < v1.w;
}

__device__ __forceinline__ bool operator<=(const uint4& v0, const uint4& v1) noexcept
{
    return v0.x <= v1.x && v0.y <= v1.y && v0.z <= v1.z && v0.w <= v1.w;
}

__device__ __forceinline__ uint4 operator+=(uint4& v0, const uint4 v1) noexcept
{
    v0 = v0 + v1;
    return v0;
}

__device__ __forceinline__ uint4 operator+=(uint4& v0, const uint t) noexcept
{
    v0 = v0 + t;
    return v0;
}

__device__ __forceinline__ uint4 operator-=(uint4& v0, const uint4 v1) noexcept
{
    v0 = v0 - v1;
    return v0;
}

__device__ __forceinline__ uint4 operator-=(uint4& v0, const uint t) noexcept
{
    v0 = v0 - t;
    return v0;
}

__device__ __forceinline__ uint4 operator*=(uint4& v0, const uint4 v1) noexcept
{
    v0 = v0 * v1;
    return v0;
}

__device__ __forceinline__ uint4 operator*=(uint4& v0, const uint t) noexcept
{
    v0 = v0 * t;
    return v0;
}

__device__ __forceinline__ uint4 operator/=(uint4& v0, const uint4 v1) noexcept
{
    v0 = v0 / v1;
    return v0;
}

__device__ __forceinline__ uint4 operator/=(uint4& v0, const uint t) noexcept
{
    v0 = v0 / t;
    return v0;
}

__device__ __forceinline__ uint4 operator&=(uint4& v0, const uint4& v1) noexcept
{
    v0 = v0 & v1;
    return v0;
}

__device__ __forceinline__ uint4 operator&=(uint4& v0, const uint t) noexcept
{
    v0 = v0 & t;
    return v0;
}

__device__ __forceinline__ uint4 operator|=(uint4& v0, const uint4& v1) noexcept
{
    v0 = v0 | v1;
    return v0;
}

__device__ __forceinline__ uint4 operator|=(uint4& v0, const uint t) noexcept
{
    v0 = v0 | t;
    return v0;
}

__device__ __forceinline__ uint4 operator^=(uint4& v0, const uint4& v1) noexcept
{
    v0 = v0 ^ v1;
    return v0;
}

__device__ __forceinline__ uint4 operator^=(uint4& v0, const uint t) noexcept
{
    v0 = v0 ^ t;
    return v0;
}

__device__ __forceinline__ uint4 operator>>=(uint4& v0, const uint t) noexcept
{
    v0 = v0 >> t;
    return v0;
}

__device__ __forceinline__ uint4 operator<<=(uint4& v0, const uint t) noexcept
{
    v0 = v0 << t;
    return v0;
}
#pragma once

#include "defines.cuh"

__device__ __forceinline__ uint2 operator+(const uint2& vec, const uint2& other) noexcept
{
    return make_uint2(vec.x + other.x, vec.y + other.y);
}

__device__ __forceinline__ uint2 operator+(const uint2& vec, const uint t) noexcept
{
    return make_uint2(vec.x + t, vec.y + t);
}

__device__ __forceinline__ uint2 operator-(const uint2& vec, const uint2& other) noexcept
{
    return make_uint2(vec.x - other.x, vec.y - other.y);
}

__device__ __forceinline__ uint2 operator-(const uint2& vec, const uint t) noexcept
{
    return make_uint2(vec.x - t, vec.y - t);
}

__device__ __forceinline__ uint2 operator*(const uint2& vec, const uint2& other) noexcept
{
    return make_uint2(vec.x * other.x, vec.y * other.y);
}

__device__ __forceinline__ uint2 operator*(const uint2& vec, const uint t) noexcept
{
    return make_uint2(vec.x * t, vec.y * t);
}

__device__ __forceinline__ uint2 operator*(const uint t, const uint2& vec) noexcept
{
    return make_uint2(vec.x * t, vec.y * t);
}

__device__ __forceinline__ uint2 operator/(const uint2& vec, const uint2& other) noexcept
{
    return make_uint2(vec.x / other.x, vec.y / other.y);
}

__device__ __forceinline__ uint2 operator/(const uint2& vec, const uint t) noexcept
{
    return make_uint2(vec.x / t, vec.y / t);
}

__device__ __forceinline__ uint2 operator/(const uint t, const uint2& vec) noexcept
{
    return make_uint2(t / vec.x, t / vec.y);
}

__device__ __forceinline__ uint2 operator~(const uint2 vec) noexcept
{
    return make_uint2(~vec.x, ~vec.y);
}

__device__ __forceinline__ uint2 operator&(const uint2& vec, const uint2& other) noexcept
{
    return make_uint2(vec.x & other.x, vec.y & other.y);
}

__device__ __forceinline__ uint2 operator&(const uint2& vec, const uint t) noexcept
{
    return make_uint2(vec.x & t, vec.y & t);
}

__device__ __forceinline__ uint2 operator|(const uint2& vec, const uint2& other) noexcept
{
    return make_uint2(vec.x | other.x, vec.y | other.y);
}

__device__ __forceinline__ uint2 operator|(const uint2& vec, const uint t) noexcept
{
    return make_uint2(vec.x | t, vec.y | t);
}

__device__ __forceinline__ uint2 operator^(const uint2& vec, const uint2& other) noexcept
{
    return make_uint2(vec.x ^ other.x, vec.y ^ other.y);
}

__device__ __forceinline__ uint2 operator^(const uint2& vec, const uint t) noexcept
{
    return make_uint2(vec.x ^ t, vec.y ^ t);
}

__device__ __forceinline__ uint2 operator>>(const uint2& vec, const uint t) noexcept
{
    return make_uint2(vec.x >> t, vec.y >> t);
}

__device__ __forceinline__ uint2 operator<<(const uint2& vec, const uint t) noexcept
{
    return make_uint2(vec.x << t, vec.y << t);
}

__device__ __forceinline__ bool operator==(const uint2& v0, const uint2& v1) noexcept
{
    return v0.x == v1.x && v0.y == v1.y;
}

__device__ __forceinline__ bool operator>(const uint2& v0, const uint2& v1) noexcept
{
    return v0.x > v1.x && v0.y > v1.y;
}

__device__ __forceinline__ bool operator>=(const uint2& v0, const uint2& v1) noexcept
{
    return v0.x >= v1.x && v0.y >= v1.y;
}

__device__ __forceinline__ bool operator<(const uint2& v0, const uint2& v1) noexcept
{
    return v0.x < v1.x && v0.y < v1.y;
}

__device__ __forceinline__ bool operator<=(const uint2& v0, const uint2& v1) noexcept
{
    return v0.x <= v1.x && v0.y <= v1.y;
}

__device__ __forceinline__ uint2 operator+=(uint2& v0, const uint2 v1) noexcept
{
    v0 = v0 + v1;
    return v0;
}

__device__ __forceinline__ uint2 operator+=(uint2& v0, const uint t) noexcept
{
    v0 = v0 + t;
    return v0;
}

__device__ __forceinline__ uint2 operator-=(uint2& v0, const uint2 v1) noexcept
{
    v0 = v0 - v1;
    return v0;
}

__device__ __forceinline__ uint2 operator-=(uint2& v0, const uint t) noexcept
{
    v0 = v0 - t;
    return v0;
}

__device__ __forceinline__ uint2 operator*=(uint2& v0, const uint2 v1) noexcept
{
    v0 = v0 * v1;
    return v0;
}

__device__ __forceinline__ uint2 operator*=(uint2& v0, const uint t) noexcept
{
    v0 = v0 * t;
    return v0;
}

__device__ __forceinline__ uint2 operator/=(uint2& v0, const uint2 v1) noexcept
{
    v0 = v0 / v1;
    return v0;
}

__device__ __forceinline__ uint2 operator/=(uint2& v0, const uint t) noexcept
{
    v0 = v0 / t;
    return v0;
}

__device__ __forceinline__ uint2 operator&=(uint2& v0, const uint t) noexcept
{
    v0 = v0 & t;
    return v0;
}

__device__ __forceinline__ uint2 operator|=(uint2& v0, const uint t) noexcept
{
    v0 = v0 | t;
    return v0;
}

__device__ __forceinline__ uint2 operator^=(uint2& v0, const uint t) noexcept
{
    v0 = v0 ^ t;
    return v0;
}

__device__ __forceinline__ uint2 operator>>=(uint2& v0, const uint t) noexcept
{
    v0 = v0 >> t;
    return v0;
}

__device__ __forceinline__ uint2 operator<<=(uint2& v0, const uint t) noexcept
{
    v0 = v0 << t;
    return v0;
}
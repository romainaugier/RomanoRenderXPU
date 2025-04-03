#pragma once

#include "defines.cuh"

__device__ __forceinline__ uint3 operator+(const uint3& vec, const uint3& other) noexcept
{
    return make_uint3(vec.x + other.x, vec.y + other.y, vec.z + other.z);
}

__device__ __forceinline__ uint3 operator+(const uint3& vec, const uint t) noexcept
{
    return make_uint3(vec.x + t, vec.y + t, vec.z + t);
}

__device__ __forceinline__ uint3 operator-(const uint3& vec, const uint3& other) noexcept
{
    return make_uint3(vec.x - other.x, vec.y - other.y, vec.z - other.z);
}

__device__ __forceinline__ uint3 operator-(const uint3& vec, const uint t) noexcept
{
    return make_uint3(vec.x - t, vec.y - t, vec.z - t);
}

__device__ __forceinline__ uint3 operator*(const uint3& vec, const uint3& other) noexcept
{
    return make_uint3(vec.x * other.x, vec.y * other.y, vec.z * other.z);
}

__device__ __forceinline__ uint3 operator*(const uint3& vec, const uint t) noexcept
{
    return make_uint3(vec.x * t, vec.y * t, vec.z * t);
}

__device__ __forceinline__ uint3 operator*(const uint t, const uint3& vec) noexcept
{
    return make_uint3(vec.x * t, vec.y * t, vec.z * t);
}

__device__ __forceinline__ uint3 operator/(const uint3& vec, const uint3& other) noexcept
{
    return make_uint3(vec.x / other.x, vec.y / other.y, vec.z / other.z);
}

__device__ __forceinline__ uint3 operator/(const uint3& vec, const uint t) noexcept
{
    return make_uint3(vec.x / t, vec.y / t, vec.z / t);
}

__device__ __forceinline__ uint3 operator/(const uint t, const uint3& vec) noexcept
{
    return make_uint3(t / vec.x, t / vec.y, t / vec.z);
}

__device__ __forceinline__ uint3 operator~(const uint3 vec) noexcept
{
    return make_uint3(~vec.x, ~vec.y, ~vec.z);
}

__device__ __forceinline__ uint3 operator&(const uint3& vec, const uint3& other) noexcept
{
    return make_uint3(vec.x & other.x, vec.y & other.y, vec.z & other.z);
}

__device__ __forceinline__ uint3 operator&(const uint3& vec, const uint t) noexcept
{
    return make_uint3(vec.x & t, vec.y & t, vec.z & t);
}

__device__ __forceinline__ uint3 operator|(const uint3& vec, const uint3& other) noexcept
{
    return make_uint3(vec.x | other.x, vec.y | other.y, vec.z | other.z);
}

__device__ __forceinline__ uint3 operator|(const uint3& vec, const uint t) noexcept
{
    return make_uint3(vec.x | t, vec.y | t, vec.z | t);
}

__device__ __forceinline__ uint3 operator^(const uint3& vec, const uint3& other) noexcept
{
    return make_uint3(vec.x ^ other.x, vec.y ^ other.y, vec.z ^ other.z);
}

__device__ __forceinline__ uint3 operator^(const uint3& vec, const uint t) noexcept
{
    return make_uint3(vec.x ^ t, vec.y ^ t, vec.z ^ t);
}

__device__ __forceinline__ uint3 operator>>(const uint3& vec, const uint t) noexcept
{
    return make_uint3(vec.x >> t, vec.y >> t, vec.z >> t);
}

__device__ __forceinline__ uint3 operator<<(const uint3& vec, const uint t) noexcept
{
    return make_uint3(vec.x << t, vec.y << t, vec.z << t);
}

__device__ __forceinline__ bool operator==(const uint3& v0, const uint3& v1) noexcept
{
    return v0.x == v1.x && v0.y == v1.y && v0.z == v1.z;
}

__device__ __forceinline__ bool operator>(const uint3& v0, const uint3& v1) noexcept
{
    return v0.x > v1.x && v0.y > v1.y && v0.z > v1.z;
}

__device__ __forceinline__ bool operator>=(const uint3& v0, const uint3& v1) noexcept
{
    return v0.x >= v1.x && v0.y >= v1.y && v0.z >= v1.z;
}

__device__ __forceinline__ bool operator<(const uint3& v0, const uint3& v1) noexcept
{
    return v0.x < v1.x && v0.y < v1.y && v0.z < v1.z;
}

__device__ __forceinline__ bool operator<=(const uint3& v0, const uint3& v1) noexcept
{
    return v0.x <= v1.x && v0.y <= v1.y && v0.z <= v1.z;
}

__device__ __forceinline__ uint3 operator+=(uint3& v0, const uint3 v1) noexcept
{
    v0 = v0 + v1;
    return v0;
}

__device__ __forceinline__ uint3 operator+=(uint3& v0, const uint t) noexcept
{
    v0 = v0 + t;
    return v0;
}

__device__ __forceinline__ uint3 operator-=(uint3& v0, const uint3 v1) noexcept
{
    v0 = v0 - v1;
    return v0;
}

__device__ __forceinline__ uint3 operator-=(uint3& v0, const uint t) noexcept
{
    v0 = v0 - t;
    return v0;
}

__device__ __forceinline__ uint3 operator*=(uint3& v0, const uint3 v1) noexcept
{
    v0 = v0 * v1;
    return v0;
}

__device__ __forceinline__ uint3 operator*=(uint3& v0, const uint t) noexcept
{
    v0 = v0 * t;
    return v0;
}

__device__ __forceinline__ uint3 operator/=(uint3& v0, const uint3 v1) noexcept
{
    v0 = v0 / v1;
    return v0;
}

__device__ __forceinline__ uint3 operator/=(uint3& v0, const uint t) noexcept
{
    v0 = v0 / t;
    return v0;
}

__device__ __forceinline__ uint3 operator&=(uint3& v0, const uint3& v1) noexcept
{
    v0 = v0 & v1;
    return v0;
}

__device__ __forceinline__ uint3 operator&=(uint3& v0, const uint t) noexcept
{
    v0 = v0 & t;
    return v0;
}

__device__ __forceinline__ uint3 operator|=(uint3& v0, const uint3& v1) noexcept
{
    v0 = v0 | v1;
    return v0;
}

__device__ __forceinline__ uint3 operator|=(uint3& v0, const uint t) noexcept
{
    v0 = v0 | t;
    return v0;
}

__device__ __forceinline__ uint3 operator^=(uint3& v0, const uint3& v1) noexcept
{
    v0 = v0 ^ v1;
    return v0;
}

__device__ __forceinline__ uint3 operator^=(uint3& v0, const uint t) noexcept
{
    v0 = v0 ^ t;
    return v0;
}

__device__ __forceinline__ uint3 operator>>=(uint3& v0, const uint t) noexcept
{
    v0 = v0 >> t;
    return v0;
}

__device__ __forceinline__ uint3 operator<<=(uint3& v0, const uint t) noexcept
{
    v0 = v0 << t;
    return v0;
}
#pragma once

#include "math.cuh"

__device__ __forceinline__ float4 make_float4(const float3& vec)
{
    return make_float4(vec.x, vec.y, vec.z, 0.0f);
}

__device__ __forceinline__ float4 make_float4(const float3& vec, const float t)
{
    return make_float4(vec.x, vec.y, vec.z, t);
}

__device__ __forceinline__ float4 make_float4(const float t) { return make_float4(t, t, t, t); }

__device__ __forceinline__ float4 operator+(const float4& vec, const float4& other) noexcept
{
    return make_float4(vec.x + other.x, vec.y + other.y, vec.z + other.z, vec.w + other.w);
}

__device__ __forceinline__ float4 operator+(const float4& vec, const float t) noexcept
{
    return make_float4(vec.x + t, vec.y + t, vec.z + t, vec.w + t);
}

__device__ __forceinline__ float4 operator-(const float4& vec, const float4& other) noexcept
{
    return make_float4(vec.x - other.x, vec.y - other.y, vec.z - other.z, vec.w - other.w);
}

__device__ __forceinline__ float4 operator-(const float4& vec, const float t) noexcept
{
    return make_float4(vec.x - t, vec.y - t, vec.z - t, vec.w - t);
}

__device__ __forceinline__ float4 operator*(const float4& vec, const float4& other) noexcept
{
    return make_float4(vec.x * other.x, vec.y * other.y, vec.z * other.z, vec.w * other.w);
}

__device__ __forceinline__ float4 operator*(const float4& vec, const float t) noexcept
{
    return make_float4(vec.x * t, vec.y * t, vec.z * t, vec.w * t);
}

__device__ __forceinline__ float4 operator*(const float t, const float4& vec) noexcept
{
    return make_float4(vec.x * t, vec.y * t, vec.z * t, vec.w * t);
}

__device__ __forceinline__ float4 operator/(const float4& vec, const float4& other) noexcept
{
    return make_float4(__fdiv_rn(vec.x, other.x),
                       __fdiv_rn(vec.y, other.y),
                       __fdiv_rn(vec.z, other.z),
                       __fdiv_rn(vec.w, other.w));
}

__device__ __forceinline__ float4 operator/(const float4& vec, const float t) noexcept
{
    return make_float4(__fdiv_rn(vec.x, t), __fdiv_rn(vec.y, t), __fdiv_rn(vec.z, t), __fdiv_rn(vec.w, t));
}

__device__ __forceinline__ float4 operator/(const float t, const float4& vec) noexcept
{
    return make_float4(__fdiv_rn(t, vec.x), __fdiv_rn(t, vec.y), __fdiv_rn(t, vec.z), __fdiv_rn(t, vec.w));
}

__device__ __forceinline__ bool operator==(const float4& v0, const float4& v1) noexcept
{
    if(v0.x == v1.x && v0.y == v1.y && v0.z == v1.z && v0.w == v1.w)
        return true;
    else
        return false;
}

__device__ __forceinline__ bool operator>(const float4& v0, const float4& v1) noexcept
{
    if(v0.x > v1.x && v0.y > v1.y && v0.z > v1.z && v0.w > v1.w)
        return true;
    else
        return false;
}

__device__ __forceinline__ bool operator>=(const float4& v0, const float4& v1) noexcept
{
    if(v0.x >= v1.x && v0.y >= v1.y && v0.z >= v1.z && v0.w >= v1.w)
        return true;
    else
        return false;
}

__device__ __forceinline__ bool operator<(const float4& v0, const float4& v1) noexcept
{
    if(v0.x < v1.x && v0.y < v1.y && v0.z < v1.z && v0.w < v1.w)
        return true;
    else
        return false;
}

__device__ __forceinline__ bool operator<=(const float4& v0, const float4& v1) noexcept
{
    if(v0.x <= v1.x && v0.y <= v1.y && v0.z <= v1.z && v0.w <= v1.w)
        return true;
    else
        return false;
}

__device__ __forceinline__ float4 operator+=(float4& v0, float4 v1) noexcept
{
    v0 = v0 + v1;
    return v0;
}

__device__ __forceinline__ float4 operator-=(float4& v0, float4 v1) noexcept
{
    v0 = v0 - v1;
    return v0;
}

__device__ __forceinline__ float4 operator*=(float4& v0, float4 v1) noexcept
{
    v0 = v0 * v1;
    return v0;
}

__device__ __forceinline__ float4 operator*=(float4& v0, float t) noexcept
{
    v0 = v0 * t;
    return v0;
}

__device__ __forceinline__ float4 operator/=(float4& v0, float4 v1) noexcept
{
    v0 = v0 / v1;
    return v0;
}

__device__ __forceinline__ float4 operator/=(float4& v0, float t) noexcept
{
    v0 = v0 / t;
    return v0;
}

__device__ __forceinline__ float dot_float4(const float4& v1, const float4& v2) noexcept
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}

__device__ __forceinline__ float4 cross_float4(const float4& v1, const float4& v2) noexcept
{
    return make_float4(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x, 0.0f);
}

__device__ __forceinline__ float length_float4(const float4& a) noexcept
{
    return __fsqrt_rn(dot_float4(a, a));
}

__device__ __forceinline__ float length2_float4(const float4& a) noexcept
{
    return dot_float4(a, a);
}

__device__ __forceinline__ float4 normalize_safe_float4(const float4& v) noexcept
{
    float t = 1.0f / length_float4(v);
    return make_float4(v.x * t, v.y * t, v.z * t, v.w * t);
}

__device__ __forceinline__ float4 normalize_float4(const float4& v) noexcept
{
    float t = __frsqrt_rn(dot_float4(v, v));
    return make_float4(v.x * t, v.y * t, v.z * t, v.w * t);
}

__device__ __forceinline__ float dist_float4(const float4& a, const float4& b) noexcept
{
    return __fsqrt_rn((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)
                      + (a.z - b.z) * (a.z - b.z) + (a.w - b.w) * (a.w - b.w));
}

__device__ __forceinline__ float4 sum_float4(const float4& v1, const float4& v2, const float4& v3) noexcept
{
    return make_float4((v1.x + v2.x + v3.x) * 0.33333f,
                       (v1.y + v2.y + v3.y) * 0.33333f,
                       (v1.z + v2.z + v3.z) * 0.33333f,
                       (v1.w + v2.w + v3.w) * 0.33333f);
}

__device__ __forceinline__ float4 pow_float4(const float4& v, const float p) noexcept
{
    return make_float4(__powf(v.x, p), __powf(v.y, p), __powf(v.z, p), __powf(v.w, p));
}

__device__ __forceinline__ float4 min_float4(const float4& a, const float4& b) noexcept
{
    return make_float4(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z), fminf(a.w, b.w));
}

__device__ __forceinline__ float4 max_float4(const float4& a, const float4& b) noexcept
{
    return make_float4(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z), fmaxf(a.w, b.w));
}

__device__ __forceinline__ float4 lerp_float4f(const float4& a, const float4& b, const float t) noexcept
{
    return make_float4(lerpf(a.x, b.x, t), lerpf(a.y, b.y, t), lerpf(a.z, b.z, t), lerpf(a.w, b.w, t));
}

__device__ __forceinline__ float4 lerp_float4v(const float4& a, const float4& b, const float4& t) noexcept
{
    return make_float4(lerpf(a.x, b.x, t.x), lerpf(a.y, b.y, t.y), lerpf(a.z, b.z, t.z), lerpf(a.w, b.w, t.w));
}

__device__ __forceinline__ float4 rcp_float4(const float4& v) noexcept
{
    return make_float4(__frcp_rn(v.x), __frcp_rn(v.y), __frcp_rn(v.z), __frcp_rn(v.w));
}

__device__ __forceinline__ float4 sign_float4(const float4& v) noexcept
{
    return make_float4(copysignf(1.0f, v.x), copysignf(1.0f, v.y), copysignf(1.0f, v.z), copysignf(1.0f, v.w));
}

__device__ __forceinline__ float4 abs_float4(const float4& v) noexcept
{
    return make_float4(fabsf(v.x), fabsf(v.y), fabsf(v.z), fabsf(v.w));
}

__device__ __forceinline__ float4 floor_float4(const float4& v) noexcept
{
    return make_float4(floorf(v.x), floorf(v.y), floorf(v.z), floorf(v.w));
}

__device__ __forceinline__ float4 ceil_float4(const float4& v) noexcept
{
    return make_float4(ceilf(v.x), ceilf(v.y), ceilf(v.z), ceilf(v.w));
}

__device__ __forceinline__ bool isnan_float4(const float4& v) noexcept
{
    return __isnanf(v.x) | __isnanf(v.y) | __isnanf(v.z) | __isnanf(v.w);
}
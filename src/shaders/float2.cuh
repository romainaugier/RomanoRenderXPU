#pragma once

#include "math.cuh"

__device__ __forceinline__ float2 make_float2(const float f)
{
    return make_float2(f, f);
}

__device__ __forceinline__ float2 make_float2(const float3& vec)
{
    return make_float2(vec.x, vec.y);
}

__device__ __forceinline__ float2 make_float2(const float4& vec)
{
    return make_float2(vec.x, vec.y);
}

__device__ __forceinline__ float2 operator+(const float2& vec, const float2& other) noexcept
{
    return make_float2(vec.x + other.x, vec.y + other.y);
}

__device__ __forceinline__ float2 operator+(const float2& vec, const float t) noexcept
{
    return make_float2(vec.x + t, vec.y + t);
}

__device__ __forceinline__ float2 operator-(const float2& vec, const float2& other) noexcept
{
    return make_float2(vec.x - other.x, vec.y - other.y);
}

__device__ __forceinline__ float2 operator-(const float2& vec, const float t) noexcept
{
    return make_float2(vec.x - t, vec.y - t);
}

__device__ __forceinline__ float2 operator*(const float2& vec, const float2& other) noexcept
{
    return make_float2(vec.x * other.x, vec.y * other.y);
}

__device__ __forceinline__ float2 operator*(const float2& vec, const float t) noexcept
{
    return make_float2(vec.x * t, vec.y * t);
}

__device__ __forceinline__ float2 operator*(const float t, const float2& vec) noexcept
{
    return make_float2(vec.x * t, vec.y * t);
}

__device__ __forceinline__ float2 operator/(const float2& vec, const float2& other) noexcept
{
    return make_float2(__fdiv_rn(vec.x, other.x), __fdiv_rn(vec.y, other.y));
}

__device__ __forceinline__ float2 operator/(const float2& vec, const float t) noexcept
{
    float t_inv = __frcp_rn(t);
    return make_float2(vec.x * t_inv, vec.y * t_inv);
}

__device__ __forceinline__ float2 operator/(const float t, const float2& vec) noexcept
{
    return make_float2(__fdiv_rn(t, vec.x), __fdiv_rn(t, vec.y));
}

__device__ __forceinline__ bool operator==(const float2& v0, const float2& v1) noexcept
{
    return v0.x == v1.x && v0.y == v1.y;
}

__device__ __forceinline__ bool operator>(const float2& v0, const float2& v1) noexcept
{
    return v0.x > v1.x && v0.y > v1.y;
}

__device__ __forceinline__ bool operator>=(const float2& v0, const float2& v1) noexcept
{
    return v0.x >= v1.x && v0.y >= v1.y;
}

__device__ __forceinline__ bool operator<(const float2& v0, const float2& v1) noexcept
{
    return v0.x < v1.x && v0.y < v1.y;
}

__device__ __forceinline__ bool operator<=(const float2& v0, const float2& v1) noexcept
{
    return v0.x <= v1.x && v0.y <= v1.y;
}

__device__ __forceinline__ float2 operator+=(float2& v0, float2 v1) noexcept
{
    v0 = v0 + v1;
    return v0;
}

__device__ __forceinline__ float2 operator-=(float2& v0, float2 v1) noexcept
{
    v0 = v0 - v1;
    return v0;
}

__device__ __forceinline__ float2 operator*=(float2& v0, float2 v1) noexcept
{
    v0 = v0 * v1;
    return v0;
}

__device__ __forceinline__ float2 operator*=(float2& v0, float t) noexcept
{
    v0 = v0 * t;
    return v0;
}

__device__ __forceinline__ float2 operator/=(float2& v0, float2 v1) noexcept
{
    v0 = v0 / v1;
    return v0;
}

__device__ __forceinline__ float2 operator/=(float2& v0, float t) noexcept
{
    return make_float2(__fdiv_rn(v0.x, t), __fdiv_rn(v0.y, t));
}

__device__ __forceinline__ float dot_float2(const float2& v1, const float2& v2) noexcept
{
    return v1.x * v2.x + v1.y * v2.y;
}

__device__ __forceinline__ float length_float2(const float2& a) noexcept
{
    return __fsqrt_rn(dot_float2(a, a));
}

__device__ __forceinline__ float length2_float2(const float2& a) noexcept
{
    return dot_float2(a, a);
}

__device__ __forceinline__ float2 normalize_safe_float2(const float2& v) noexcept
{
    float t = 1.0f / length_float2(v);
    return make_float2(v.x * t, v.y * t);
}

__device__ __forceinline__ float2 normalize_float2(const float2& v) noexcept
{
    float t = __frsqrt_rn(dot_float2(v, v));
    return make_float2(v.x * t, v.y * t);
}

__device__ __forceinline__ float dist_float2(const float2& a, const float2& b) noexcept
{
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return __fsqrt_rn(dx * dx + dy * dy);
}

__device__ __forceinline__ float dist2_float2(const float2& a, const float2& b) noexcept
{
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    return dx * dx + dy * dy;
}

__device__ __forceinline__ float2 sum_float2(const float2& v1, const float2& v2, const float2& v3) noexcept
{
    return make_float2((v1.x + v2.x + v3.x) * 0.33333f, (v1.y + v2.y + v3.y) * 0.33333f);
}

__device__ __forceinline__ float2 pow_float2(const float2& v, const float p) noexcept
{
    return make_float2(__powf(v.x, p), __powf(v.y, p));
}

__device__ __forceinline__ float2 min_float2(const float2& a, const float2& b) noexcept
{
    return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}

__device__ __forceinline__ float2 max_float2(const float2& a, const float2& b) noexcept
{
    return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}

__device__ __forceinline__ float2 lerp_float2f(const float2& a, const float2& b, const float t) noexcept
{
    return make_float2(lerpf(a.x, b.x, t), lerpf(a.y, b.y, t));
}

__device__ __forceinline__ float2 lerp_float2v(const float2& a, const float2& b, const float2& t) noexcept
{
    return make_float2(lerpf(a.x, b.x, t.x), lerpf(a.y, b.y, t.y));
}

__device__ __forceinline__ float2 rcp_float2(const float2& v) noexcept
{
    return make_float2(__frcp_rn(v.x), __frcp_rn(v.y));
}

__device__ __forceinline__ float2 rotate_float2(const float2& v, float angle_radians) noexcept
{
    float sin_angle;
    float cos_angle;

    __sincosf(angle_radians, &sin_angle, &cos_angle);

    return make_float2(v.x * cos_angle - v.y * sin_angle, v.x * sin_angle + v.y * cos_angle);
}

__device__ __forceinline__ float2 perpendicular_float2(const float2& v) noexcept
{
    return make_float2(-v.y, v.x);
}

__device__ __forceinline__ float angle_between_float2(const float2& v1, const float2& v2) noexcept
{
    return acosf(fminf(fmaxf(dot_float2(normalize_float2(v1), normalize_float2(v2)), -1.0f), 1.0f));
}

__device__ __forceinline__ float2 reflect_float2(const float2& v, const float2& n) noexcept
{
    float factor = 2.0f * dot_float2(v, n);
    return make_float2(v.x - factor * n.x, v.y - factor * n.y);
}

__device__ __forceinline__ float2 sign_float2(const float2& v) noexcept
{
    return make_float2(copysignf(1.0f, v.x), copysignf(1.0f, v.y));
}

__device__ __forceinline__ float2 floor_float2(const float2& v) noexcept
{
    return make_float2(floorf(v.x), floorf(v.y));
}

__device__ __forceinline__ float2 ceil_float2(const float2& v) noexcept
{
    return make_float2(ceilf(v.x), ceilf(v.y));
}

__device__ __forceinline__ float2 saturate_float2(const float2& v) noexcept
{
    return make_float2(__saturatef(v.x), __saturatef(v.y));
}

__device__ __forceinline__ bool isnan_float2(const float2& v) noexcept
{
    return __isnanf(v.x) | __isnanf(v.y);
}
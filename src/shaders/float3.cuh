#pragma once

#include "math.cuh"

__device__ __forceinline__ float3 make_float3(const float f) { return make_float3(f, f, f); }

__device__ __forceinline__ float3 make_float3(const float2& vec)
{
    return make_float3(vec.x, vec.y, 0.0f);
}

__device__ __forceinline__ float3 make_float3(const float4& vec)
{
    return make_float3(vec.x, vec.y, vec.z);
}

__device__ __forceinline__ float3 operator+(const float3& vec, const float3& other) noexcept
{
    return make_float3(vec.x + other.x, vec.y + other.y, vec.z + other.z);
}

__device__ __forceinline__ float3 operator+(const float3& vec, const float t) noexcept
{
    return make_float3(vec.x + t, vec.y + t, vec.z + t);
}

__device__ __forceinline__ float3 operator-(const float3& vec, const float3& other) noexcept
{
    return make_float3(vec.x - other.x, vec.y - other.y, vec.z - other.z);
}

__device__ __forceinline__ float3 operator-(const float3& vec, const float t) noexcept
{
    return make_float3(vec.x - t, vec.y - t, vec.z - t);
}

__device__ __forceinline__ float3 operator*(const float3& vec, const float3& other) noexcept
{
    return make_float3(vec.x * other.x, vec.y * other.y, vec.z * other.z);
}

__device__ __forceinline__ float3 operator*(const float3& vec, const float t) noexcept
{
    return make_float3(vec.x * t, vec.y * t, vec.z * t);
}

__device__ __forceinline__ float3 operator*(const float t, const float3& vec) noexcept
{
    return make_float3(vec.x * t, vec.y * t, vec.z * t);
}

__device__ __forceinline__ float3 operator/(const float3& vec, const float3& other) noexcept
{
    return make_float3(__fdiv_rn(vec.x, other.x), __fdiv_rn(vec.y, other.y), __fdiv_rn(vec.z, other.z));
}

__device__ __forceinline__ float3 operator/(const float3& vec, const float t) noexcept
{
    float t_inv = __frcp_rn(t);
    return make_float3(vec.x * t_inv, vec.y * t_inv, vec.z * t_inv);
}

__device__ __forceinline__ float3 operator/(const float t, const float3& vec) noexcept
{
    return make_float3(__fdiv_rn(t, vec.x), __fdiv_rn(t, vec.y), __fdiv_rn(t, vec.z));
}

__device__ __forceinline__ bool operator==(const float3& v0, const float3& v1) noexcept
{
    return v0.x == v1.x && v0.y == v1.y && v0.z == v1.z;
}

__device__ __forceinline__ bool operator>(const float3& v0, const float3& v1) noexcept
{
    return v0.x > v1.x && v0.y > v1.y && v0.z > v1.z;
}

__device__ __forceinline__ bool operator>=(const float3& v0, const float3& v1) noexcept
{
    return v0.x >= v1.x && v0.y >= v1.y && v0.z >= v1.z;
}

__device__ __forceinline__ bool operator<(const float3& v0, const float3& v1) noexcept
{
    return v0.x < v1.x && v0.y < v1.y && v0.z < v1.z;
}

__device__ __forceinline__ bool operator<=(const float3& v0, const float3& v1) noexcept
{
    return v0.x <= v1.x && v0.y <= v1.y && v0.z <= v1.z;
}

__device__ __forceinline__ float3 operator+=(float3& v0, float3 v1) noexcept
{
    v0 = v0 + v1;
    return v0;
}

__device__ __forceinline__ float3 operator-=(float3& v0, float3 v1) noexcept
{
    v0 = v0 - v1;
    return v0;
}

__device__ __forceinline__ float3 operator*=(float3& v0, float3 v1) noexcept
{
    v0 = v0 * v1;
    return v0;
}

__device__ __forceinline__ float3 operator*=(float3& v0, float t) noexcept
{
    v0 = v0 * t;
    return v0;
}

__device__ __forceinline__ float3 operator/=(float3& v0, float3 v1) noexcept
{
    v0 = v0 / v1;
    return v0;
}

__device__ __forceinline__ float3 operator/=(float3& v0, float t) noexcept
{
    return make_float3(__fdiv_rn(v0.x, t), __fdiv_rn(v0.y, t), __fdiv_rn(v0.z, t));
}

__device__ __forceinline__ float dot_float3(const float3& v1, const float3& v2) noexcept
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__device__ __forceinline__ float3 cross_float3(const float3& v1, const float3& v2) noexcept
{
    return make_float3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

__device__ __forceinline__ float length_float3(const float3& a) noexcept
{
    return __fsqrt_rn(dot_float3(a, a));
}

__device__ __forceinline__ float length2_float3(const float3& a) noexcept
{
    return dot_float3(a, a);
}

__device__ __forceinline__ float3 normalize_safe_float3(const float3& v) noexcept
{
    float t = 1.0f / length_float3(v);
    return make_float3(v.x * t, v.y * t, v.z * t);
}

__device__ __forceinline__ float3 normalize_float3(const float3& v) noexcept
{
    float t = __frsqrt_rn(dot_float3(v, v));
    return make_float3(v.x * t, v.y * t, v.z * t);
}

__device__ __forceinline__ float dist_float3(const float3& a, const float3& b) noexcept
{
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return __fsqrt_rn(dx * dx + dy * dy + dz * dz);
}

__device__ __forceinline__ float dist2_float3(const float3& a, const float3& b) noexcept
{
    const float dx = a.x - b.x;
    const float dy = a.y - b.y;
    const float dz = a.z - b.z;
    return dx * dx + dy * dy + dz * dz;
}

__device__ __forceinline__ float3 sum_float3(const float3& v1, const float3& v2, const float3& v3) noexcept
{
    return make_float3((v1.x + v2.x + v3.x) * 0.33333f,
                       (v1.y + v2.y + v3.y) * 0.33333f,
                       (v1.z + v2.z + v3.z) * 0.33333f);
}

__device__ __forceinline__ float3 pow_float3(const float3& v, const float p) noexcept
{
    return make_float3(__powf(v.x, p), __powf(v.y, p), __powf(v.z, p));
}

__device__ __forceinline__ float3 min_float3(const float3& a, const float3& b) noexcept
{
    return make_float3(fminf(a.x, b.x), fminf(a.y, b.y), fminf(a.z, b.z));
}

__device__ __forceinline__ float3 max_float3(const float3& a, const float3& b) noexcept
{
    return make_float3(fmaxf(a.x, b.x), fmaxf(a.y, b.y), fmaxf(a.z, b.z));
}

__device__ __forceinline__ float3 lerp_float3f(const float3& a, const float3& b, const float t) noexcept
{
    return make_float3(lerpf(a.x, b.x, t), lerpf(a.y, b.y, t), lerpf(a.z, b.z, t));
}

__device__ __forceinline__ float3 lerp_float3v(const float3& a, const float3& b, const float3& t) noexcept
{
    return make_float3(lerpf(a.x, b.x, t.x), lerpf(a.y, b.y, t.y), lerpf(a.z, b.z, t.z));
}

__device__ __forceinline__ float3 rcp_float3(const float3& v) noexcept
{
    return make_float3(__frcp_rn(v.x), __frcp_rn(v.y), __frcp_rn(v.z));
}

__device__ __forceinline__ float3 rotate_float3_x(const float3& v, float angle_radians) noexcept
{
    float sin_angle;
    float cos_angle;

    __sincosf(angle_radians, &sin_angle, &cos_angle);

    return make_float3(v.x, v.y * cos_angle - v.z * sin_angle, v.y * sin_angle + v.z * cos_angle);
}

__device__ __forceinline__ float3 rotate_float3_y(const float3& v, float angle_radians) noexcept
{
    float sin_angle;
    float cos_angle;

    __sincosf(angle_radians, &sin_angle, &cos_angle);

    return make_float3(v.x * cos_angle + v.z * sin_angle, v.y, -v.x * sin_angle + v.z * cos_angle);
}

__device__ __forceinline__ float3 rotate_float3_z(const float3& v, float angle_radians) noexcept
{
    float sin_angle;
    float cos_angle;

    __sincosf(angle_radians, &sin_angle, &cos_angle);

    return make_float3(v.x * cos_angle - v.y * sin_angle, v.x * sin_angle + v.y * cos_angle, v.z);
}

__device__ __forceinline__ float3 rotate_float3(const float3& v, const float3& axis, const float angle_radians) noexcept
{
    float3 normalized_axis = normalize_float3(axis);

    float sin_angle;
    float cos_angle;

    __sincosf(angle_radians, &sin_angle, &cos_angle);

    float one_minus_cos = 1.0f - cos_angle;

    float xx = normalized_axis.x * normalized_axis.x * one_minus_cos;
    float xy = normalized_axis.x * normalized_axis.y * one_minus_cos;
    float xz = normalized_axis.x * normalized_axis.z * one_minus_cos;
    float yy = normalized_axis.y * normalized_axis.y * one_minus_cos;
    float yz = normalized_axis.y * normalized_axis.z * one_minus_cos;
    float zz = normalized_axis.z * normalized_axis.z * one_minus_cos;

    float xs = normalized_axis.x * sin_angle;
    float ys = normalized_axis.y * sin_angle;
    float zs = normalized_axis.z * sin_angle;

    float3 result;
    result.x = (xx + cos_angle) * v.x + (xy - zs) * v.y + (xz + ys) * v.z;
    result.y = (xy + zs) * v.x + (yy + cos_angle) * v.y + (yz - xs) * v.z;
    result.z = (xz - ys) * v.x + (yz + xs) * v.y + (zz + cos_angle) * v.z;

    return result;
}

__device__ __forceinline__ float angle_between_float3(const float3& v1, const float3& v2) noexcept
{
    float dot_product = dot_float3(normalize_float3(v1), normalize_float3(v2));
    return acosf(fminf(fmaxf(dot_product, -1.0f), 1.0f));
}

__device__ __forceinline__ float3 reflect_float3(const float3& v, const float3& n) noexcept
{
    float factor = 2.0f * dot_float3(v, n);
    return make_float3(v.x - factor * n.x, v.y - factor * n.y, v.z - factor * n.z);
}

__device__ __forceinline__ float3 sign_float3(const float3& v) noexcept
{
    return make_float3(copysignf(1.0f, v.x), copysignf(1.0f, v.y), copysignf(1.0f, v.z));
}

__device__ __forceinline__ float3 abs_float3(const float3& v) noexcept
{
    return make_float3(fabsf(v.x), fabsf(v.y), fabsf(v.z));
}

__device__ __forceinline__ float3 floor_float3(const float3& v) noexcept
{
    return make_float3(floorf(v.x), floorf(v.y), floorf(v.z));
}

__device__ __forceinline__ float3 ceil_float3(const float3& v) noexcept
{
    return make_float3(ceilf(v.x), ceilf(v.y), ceilf(v.z));
}

__device__ __forceinline__ bool isnan_float3(const float3& v) noexcept
{
    return __isnanf(v.x) | __isnanf(v.y) | __isnanf(v.z);
}
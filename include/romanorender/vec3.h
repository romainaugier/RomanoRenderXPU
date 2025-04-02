#pragma once

#if !defined(__ROMANORENDER_VEC3)
#define __ROMANORENDER_VEC3

#include "romanorender/maths.h"

#include "spdlog/fmt/fmt.h"

ROMANORENDER_NAMESPACE_BEGIN

struct Vec3F
{
    float x, y, z;

    Vec3F() : x(0.0f), y(0.0f), z(0.0f) {}

    Vec3F(float t) : x(t), y(t), z(t) {}

    Vec3F(float X, float Y, float Z) : x(X), y(Y), z(Z) {}

    const Vec3F operator-() const { return Vec3F(-x, -y, -z); }

    const float& operator[](int i) const { return (&x)[i]; }

    float& operator[](int i) { return (&x)[i]; }
};

ROMANORENDER_FORCE_INLINE Vec3F operator+(const Vec3F& vec, const Vec3F& other) noexcept
{
    return Vec3F(vec.x + other.x, vec.y + other.y, vec.z + other.z);
}

ROMANORENDER_FORCE_INLINE Vec3F operator+(const Vec3F& vec, const float t) noexcept
{
    return Vec3F(vec.x + t, vec.y + t, vec.z + t);
}

ROMANORENDER_FORCE_INLINE Vec3F operator-(const Vec3F& vec, const Vec3F& other) noexcept
{
    return Vec3F(vec.x - other.x, vec.y - other.y, vec.z - other.z);
}

ROMANORENDER_FORCE_INLINE Vec3F operator-(const Vec3F& vec, const float t) noexcept
{
    return Vec3F(vec.x - t, vec.y - t, vec.z - t);
}

ROMANORENDER_FORCE_INLINE Vec3F operator*(const Vec3F& vec, const Vec3F& other) noexcept
{
    return Vec3F(vec.x * other.x, vec.y * other.y, vec.z * other.z);
}

ROMANORENDER_FORCE_INLINE Vec3F operator*(const Vec3F& vec, const float t) noexcept
{
    return Vec3F(vec.x * t, vec.y * t, vec.z * t);
}

ROMANORENDER_FORCE_INLINE Vec3F operator*(const float t, const Vec3F& vec) noexcept
{
    return Vec3F(vec.x * t, vec.y * t, vec.z * t);
}

ROMANORENDER_FORCE_INLINE Vec3F operator/(const Vec3F& vec, const Vec3F& other) noexcept
{
    return Vec3F(vec.x / other.x, vec.y / other.y, vec.z / other.z);
}

ROMANORENDER_FORCE_INLINE Vec3F operator/(const Vec3F& vec, const float t) noexcept
{
    return Vec3F(vec.x / t, vec.y / t, vec.z / t);
}

ROMANORENDER_FORCE_INLINE Vec3F operator/(const float t, const Vec3F& vec) noexcept
{
    return Vec3F(t / vec.x, t / vec.y, t / vec.z);
}

ROMANORENDER_FORCE_INLINE bool operator==(const Vec3F& v0, const Vec3F& v1) noexcept
{
    if(v0.x == v1.x && v0.y == v1.y && v0.z == v1.z)
        return true;
    else
        return false;
}

ROMANORENDER_FORCE_INLINE bool operator>(const Vec3F& v0, const Vec3F& v1) noexcept
{
    if(v0.x > v1.x && v0.y > v1.y && v0.z > v1.z)
        return true;
    else
        return false;
}

ROMANORENDER_FORCE_INLINE bool operator>=(const Vec3F& v0, const Vec3F& v1) noexcept
{
    if(v0.x >= v1.x && v0.y >= v1.y && v0.z >= v1.z)
        return true;
    else
        return false;
}

ROMANORENDER_FORCE_INLINE bool operator<(const Vec3F& v0, const Vec3F& v1) noexcept
{
    if(v0.x < v1.x && v0.y < v1.y && v0.z < v1.z)
        return true;
    else
        return false;
}

ROMANORENDER_FORCE_INLINE bool operator<=(const Vec3F& v0, const Vec3F& v1) noexcept
{
    if(v0.x <= v1.x && v0.y <= v1.y && v0.z <= v1.z)
        return true;
    else
        return false;
}

ROMANORENDER_FORCE_INLINE Vec3F operator+=(Vec3F& v0, Vec3F v1) noexcept
{
    v0 = v0 + v1;
    return v0;
}

ROMANORENDER_FORCE_INLINE Vec3F operator-=(Vec3F& v0, Vec3F v1) noexcept
{
    v0 = v0 - v1;
    return v0;
}

ROMANORENDER_FORCE_INLINE Vec3F operator*=(Vec3F& v0, Vec3F v1) noexcept
{
    v0 = v0 * v1;
    return v0;
}

ROMANORENDER_FORCE_INLINE Vec3F operator*=(Vec3F& v0, float t) noexcept
{
    v0 = v0 * t;
    return v0;
}

ROMANORENDER_FORCE_INLINE Vec3F operator/=(Vec3F& v0, Vec3F v1) noexcept
{
    v0 = v0 / v1;
    return v0;
}

ROMANORENDER_FORCE_INLINE Vec3F operator/=(Vec3F& v0, float t) noexcept
{
    float inv = maths::rcpf(t);
    return Vec3F(v0.x *= inv, v0.y *= inv, v0.z *= inv);
}

ROMANORENDER_FORCE_INLINE float dot_vec3f(const Vec3F& v1, const Vec3F& v2) noexcept
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

ROMANORENDER_FORCE_INLINE float length_vec3f(const Vec3F& a) noexcept
{
    return maths::sqrtf(dot_vec3f(a, a));
}

ROMANORENDER_FORCE_INLINE float length2_vec3f(const Vec3F& a) noexcept { return dot_vec3f(a, a); }

ROMANORENDER_FORCE_INLINE Vec3F normalize_safe_vec3f(const Vec3F& v) noexcept
{
    float t = 1.0f / length_vec3f(v);
    return Vec3F(v.x * t, v.y * t, v.z * t);
}

ROMANORENDER_FORCE_INLINE Vec3F normalize_vec3f(const Vec3F& v) noexcept
{
    float t = maths::rsqrtf(dot_vec3f(v, v));
    return Vec3F(v.x * t, v.y * t, v.z * t);
}

ROMANORENDER_FORCE_INLINE Vec3F cross_vec3f(const Vec3F& v1, const Vec3F& v2) noexcept
{
    return Vec3F(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x);
}

ROMANORENDER_FORCE_INLINE float dist_vec3f(const Vec3F& a, const Vec3F& b) noexcept
{
    return maths::sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z));
}

ROMANORENDER_FORCE_INLINE float dist2_vec3f(const Vec3F& a, const Vec3F& b) noexcept
{
    return (a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z);
}

ROMANORENDER_FORCE_INLINE Vec3F abs_vec3f(const Vec3F& a) noexcept
{
    return Vec3F(maths::absf(a.x), maths::absf(a.y), maths::absf(a.z));
}

ROMANORENDER_FORCE_INLINE Vec3F sum_vec3f(const Vec3F& v1, const Vec3F& v2, const Vec3F& v3) noexcept
{
    return Vec3F((v1.x + v2.x + v3.x) / 3, (v1.y + v2.y + v3.y) / 3, (v1.z + v2.z + v3.z) / 3);
}

ROMANORENDER_FORCE_INLINE Vec3F pow_vec3f(const Vec3F& v, const float p) noexcept
{
    return Vec3F(maths::powf(v.x, p), maths::powf(v.y, p), maths::powf(v.z, p));
}

ROMANORENDER_FORCE_INLINE Vec3F min_vec3f(const Vec3F& a, const Vec3F& b) noexcept
{
    return Vec3F(maths::minf(a.x, b.x), maths::minf(a.y, b.y), maths::minf(a.z, b.z));
}

ROMANORENDER_FORCE_INLINE Vec3F max_vec3f(const Vec3F& a, const Vec3F& b) noexcept
{
    return Vec3F(maths::maxf(a.x, b.x), maths::maxf(a.y, b.y), maths::maxf(a.z, b.z));
}

ROMANORENDER_FORCE_INLINE Vec3F lerp_vec3ff(const Vec3F& a, const Vec3F& b, const float t) noexcept
{
    return Vec3F(maths::lerpf(a.x, b.x, t), maths::lerpf(a.y, b.y, t), maths::lerpf(a.z, b.z, t));
}

ROMANORENDER_FORCE_INLINE Vec3F lerp_vec3fv(const Vec3F& a, const Vec3F& b, const Vec3F& t) noexcept
{
    return Vec3F(maths::lerpf(a.x, b.x, t.x), maths::lerpf(a.y, b.y, t.y), maths::lerpf(a.z, b.z, t.z));
}

ROMANORENDER_FORCE_INLINE Vec3F rcp_vec3f(const Vec3F& v) noexcept
{
    const __m128 a = _mm_set_ps(0.0f, v.z, v.y, v.x);
    const __m128 r = _mm_rcp_ps(a);

#if defined(__AVX2__)
    const __m128 result = _mm_mul_ps(r, _mm_fnmadd_ps(r, a, _mm_set1_ps(2.0f)));
#else
    const __m128 temp = _mm_sub_ps(_mm_set1_ps(2.0f), _mm_mul_ps(r, a));
    const __m128 result = _mm_mul_ps(r, temp);
#endif

    alignas(16) float components[4];
    _mm_store_ps(components, result);
    return Vec3F(components[0], components[1], components[2]);
}

ROMANORENDER_FORCE_INLINE bool isnan_vec3f(const Vec3F& v) noexcept
{
    const __m128 data = _mm_set_ps(v.x, v.y, v.z, 0.0f);
    const __m128 cmp = _mm_cmpeq_ps(data, data);
    const int mask = _mm_movemask_ps(cmp);

    return mask != 0xF;
}

#define ROMANORENDER_ABORT_IF_VEC3F_NAN(v) ROMANORENDER_ASSERT(!isnan_vec3f(v), "Vec3F contains nans")

ROMANORENDER_NAMESPACE_END

template <>
struct fmt::formatter<romanorender::Vec3F>
{
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    auto format(romanorender::Vec3F& v, format_context& ctx) const
    {
        return format_to(ctx.out(), "{}, {}, {}", v.x, v.y, v.z);
    }

    auto format(const romanorender::Vec3F& v, format_context& ctx) const
    {
        return format_to(ctx.out(), "{}, {}, {}", v.x, v.y, v.z);
    }
};

#endif /* !defined(__ROMANORENDER_VEC3) */
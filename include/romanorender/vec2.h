#pragma once

#if !defined(__ROMANORENDER_VEC2)
#define __ROMANORENDER_VEC2

#include "romanorender/maths.h"

#include "spdlog/fmt/fmt.h"

ROMANORENDER_NAMESPACE_BEGIN

struct Vec2F
{
    float x, y;

    Vec2F() : x(0.0f), y(0.0f) {}

    Vec2F(float t) : x(t), y(t) {}

    Vec2F(float X, float Y) : x(X), y(Y) {}

    const Vec2F operator-() const { return Vec2F(-x, -y); }

    const float& operator[](int i) const { return (&x)[i]; }

    float& operator[](int i) { return (&x)[i]; }
};

ROMANORENDER_FORCE_INLINE Vec2F operator+(const Vec2F& vec, const Vec2F& other) noexcept
{
    return Vec2F(vec.x + other.x, vec.y + other.y);
}

ROMANORENDER_FORCE_INLINE Vec2F operator+(const Vec2F& vec, const float t) noexcept
{
    return Vec2F(vec.x + t, vec.y + t);
}

ROMANORENDER_FORCE_INLINE Vec2F operator-(const Vec2F& vec, const Vec2F& other) noexcept
{
    return Vec2F(vec.x - other.x, vec.y - other.y);
}

ROMANORENDER_FORCE_INLINE Vec2F operator-(const Vec2F& vec, const float t) noexcept
{
    return Vec2F(vec.x - t, vec.y - t);
}

ROMANORENDER_FORCE_INLINE Vec2F operator*(const Vec2F& vec, const Vec2F& other) noexcept
{
    return Vec2F(vec.x * other.x, vec.y * other.y);
}

ROMANORENDER_FORCE_INLINE Vec2F operator*(const Vec2F& vec, const float t) noexcept
{
    return Vec2F(vec.x * t, vec.y * t);
}

ROMANORENDER_FORCE_INLINE Vec2F operator*(const float t, const Vec2F& vec) noexcept
{
    return Vec2F(vec.x * t, vec.y * t);
}

ROMANORENDER_FORCE_INLINE Vec2F operator/(const Vec2F& vec, const Vec2F& other) noexcept
{
    return Vec2F(vec.x / other.x, vec.y / other.y);
}

ROMANORENDER_FORCE_INLINE Vec2F operator/(const Vec2F& vec, const float t) noexcept
{
    return Vec2F(vec.x / t, vec.y / t);
}

ROMANORENDER_FORCE_INLINE Vec2F operator/(const float t, const Vec2F& vec) noexcept
{
    return Vec2F(t / vec.x, t / vec.y);
}

ROMANORENDER_FORCE_INLINE bool operator==(const Vec2F& v0, const Vec2F& v1) noexcept
{
    if(v0.x == v1.x && v0.y == v1.y)
        return true;
    else
        return false;
}

ROMANORENDER_FORCE_INLINE bool operator>(const Vec2F& v0, const Vec2F& v1) noexcept
{
    if(v0.x > v1.x && v0.y > v1.y)
        return true;
    else
        return false;
}

ROMANORENDER_FORCE_INLINE bool operator>=(const Vec2F& v0, const Vec2F& v1) noexcept
{
    if(v0.x >= v1.x && v0.y >= v1.y)
        return true;
    else
        return false;
}

ROMANORENDER_FORCE_INLINE bool operator<(const Vec2F& v0, const Vec2F& v1) noexcept
{
    if(v0.x < v1.x && v0.y < v1.y)
        return true;
    else
        return false;
}

ROMANORENDER_FORCE_INLINE bool operator<=(const Vec2F& v0, const Vec2F& v1) noexcept
{
    if(v0.x <= v1.x && v0.y <= v1.y)
        return true;
    else
        return false;
}

ROMANORENDER_FORCE_INLINE Vec2F operator+=(Vec2F& v0, Vec2F v1) noexcept
{
    v0 = v0 + v1;
    return v0;
}

ROMANORENDER_FORCE_INLINE Vec2F operator-=(Vec2F& v0, Vec2F v1) noexcept
{
    v0 = v0 - v1;
    return v0;
}

ROMANORENDER_FORCE_INLINE Vec2F operator*=(Vec2F& v0, Vec2F v1) noexcept
{
    v0 = v0 * v1;
    return v0;
}

ROMANORENDER_FORCE_INLINE Vec2F operator*=(Vec2F& v0, float t) noexcept
{
    v0 = v0 * t;
    return v0;
}

ROMANORENDER_FORCE_INLINE Vec2F operator/=(Vec2F& v0, Vec2F v1) noexcept
{
    v0 = v0 / v1;
    return v0;
}

ROMANORENDER_FORCE_INLINE Vec2F operator/=(Vec2F& v0, float t) noexcept
{
    float inv = maths::rcpf(t);
    return Vec2F(v0.x *= inv, v0.y *= inv);
}

ROMANORENDER_FORCE_INLINE float dot_vec2f(const Vec2F& v1, const Vec2F& v2) noexcept
{
    return v1.x * v2.x + v1.y * v2.y;
}

ROMANORENDER_FORCE_INLINE float length_vec2f(const Vec2F& a) noexcept
{
    return maths::sqrtf(dot_vec2f(a, a));
}

ROMANORENDER_FORCE_INLINE float length2_vec2f(const Vec2F& a) noexcept { return dot_vec2f(a, a); }

ROMANORENDER_FORCE_INLINE Vec2F normalize_safe_vec2f(const Vec2F& v) noexcept
{
    float t = 1.0f / length_vec2f(v);
    return Vec2F(v.x * t, v.y * t);
}

ROMANORENDER_FORCE_INLINE Vec2F normalize_vec2f(const Vec2F& v) noexcept
{
    float t = maths::rsqrtf(dot_vec2f(v, v));
    return Vec2F(v.x * t, v.y * t);
}

ROMANORENDER_FORCE_INLINE float dist_vec2f(const Vec2F& a, const Vec2F& b) noexcept
{
    return maths::sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
}

ROMANORENDER_FORCE_INLINE Vec2F abs_vec2f(const Vec2F& a) noexcept
{
    return Vec2F(maths::absf(a.x), maths::absf(a.y));
}

ROMANORENDER_FORCE_INLINE Vec2F sum_vec2f(const Vec2F& v1, const Vec2F& v2, const Vec2F& v3) noexcept
{
    return Vec2F((v1.x + v2.x + v3.x) / 3, (v1.y + v2.y + v3.y) / 3);
}

ROMANORENDER_FORCE_INLINE Vec2F pow_vec2f(const Vec2F& v, const float p) noexcept
{
    return Vec2F(maths::powf(v.x, p), maths::powf(v.y, p));
}

ROMANORENDER_FORCE_INLINE Vec2F min_vec2f(const Vec2F& a, const Vec2F& b) noexcept
{
    return Vec2F(maths::minf(a.x, b.x), maths::minf(a.y, b.y));
}

ROMANORENDER_FORCE_INLINE Vec2F max_vec2f(const Vec2F& a, const Vec2F& b) noexcept
{
    return Vec2F(maths::maxf(a.x, b.x), maths::maxf(a.y, b.y));
}

ROMANORENDER_FORCE_INLINE Vec2F lerp_vec2f(const Vec2F& a, const Vec2F& b, const float t) noexcept
{
    return Vec2F(maths::lerpf(a.x, b.x, t), maths::lerpf(a.y, b.y, t));
}

ROMANORENDER_FORCE_INLINE Vec2F lerp_vec2fv(const Vec2F& a, const Vec2F& b, const Vec2F& t) noexcept
{
    return Vec2F(maths::lerpf(a.x, b.x, t.x), maths::lerpf(a.y, b.y, t.y));
}

ROMANORENDER_FORCE_INLINE Vec2F rcp_vec2f(const Vec2F& v) noexcept
{
    const __m128 a = _mm_set_ps(0.0f, 0.0f, v.y, v.x);
    const __m128 r = _mm_rcp_ps(a);

#if defined(__AVX2__)
    const __m128 result = _mm_mul_ps(r, _mm_fnmadd_ps(r, a, _mm_set1_ps(2.0f)));
#else
    const __m128 temp = _mm_sub_ps(_mm_set1_ps(2.0f), _mm_mul_ps(r, a));
    const __m128 result = _mm_mul_ps(r, temp);
#endif

    alignas(16) float components[4];
    _mm_store_ps(components, result);
    return Vec2F(components[0], components[1]);
}

ROMANORENDER_NAMESPACE_END

template <>
struct fmt::formatter<romanorender::Vec2F>
{
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    auto format(romanorender::Vec2F& v, format_context& ctx) const
    {
        return format_to(ctx.out(), "{}, {}", v.x, v.y);
    }

    auto format(const romanorender::Vec2F& v, format_context& ctx) const
    {
        return format_to(ctx.out(), "{}, {}", v.x, v.y);
    }
};

#endif /* !defined(__ROMANORENDER_VEC2) */
#pragma once

#if !defined(__ROMANORENDER_VEC4)
#define __ROMANORENDER_VEC4

#include "romanorender/maths.h"

#include "spdlog/fmt/fmt.h"

ROMANORENDER_NAMESPACE_BEGIN

struct Vec4F
{
    float x, y, z, w;

    Vec4F() : x(0.0f), y(0.0f), z(0.0f), w(0.0f) {}

    Vec4F(float t) : x(t), y(t), z(t), w(t) {}

    Vec4F(float X, float Y, float Z, float W) : x(X), y(Y), z(Z), w(W) {}

    const Vec4F operator-() const { return Vec4F(-x, -y, -z, -w); }

    const float& operator[](int i) const { return (&x)[i]; }

    float& operator[](int i) { return (&x)[i]; }

    ROMANORENDER_FORCE_INLINE Vec4F operator+(const Vec4F& other) const noexcept
    {
        return Vec4F(x + other.x, y + other.y, z + other.z, w + other.w);
    }

    ROMANORENDER_FORCE_INLINE Vec4F operator+(const float t) const noexcept
    {
        return Vec4F(x + t, y + t, z + t, w + t);
    }

    ROMANORENDER_FORCE_INLINE Vec4F operator-(const Vec4F& other) const noexcept
    {
        return Vec4F(x - other.x, y - other.y, z - other.z, w - other.w);
    }

    ROMANORENDER_FORCE_INLINE Vec4F operator-(const float t) const noexcept
    {
        return Vec4F(x - t, y - t, z - t, w - t);
    }

    ROMANORENDER_FORCE_INLINE Vec4F operator*(const Vec4F& other) const noexcept
    {
        return Vec4F(x * other.x, y * other.y, z * other.z, w * other.w);
    }

    ROMANORENDER_FORCE_INLINE Vec4F operator*(const float t) const noexcept
    {
        return Vec4F(x * t, y * t, z * t, w * t);
    }

    ROMANORENDER_FORCE_INLINE Vec4F operator/(const Vec4F& other) const noexcept
    {
        return Vec4F(x / other.x, y / other.y, z / other.z, w / other.w);
    }

    ROMANORENDER_FORCE_INLINE Vec4F operator/(const float t) const noexcept
    {
        return Vec4F(x / t, y / t, z / t, w / t);
    }

    ROMANORENDER_FORCE_INLINE bool operator==(const Vec4F& v1) const noexcept
    {
        return (x == v1.x && y == v1.y && z == v1.z && w == v1.w);
    }

    ROMANORENDER_FORCE_INLINE bool operator>(const Vec4F& v1) const noexcept
    {
        return (x > v1.x && y > v1.y && z > v1.z && w > v1.w);
    }

    ROMANORENDER_FORCE_INLINE bool operator>=(const Vec4F& v1) const noexcept
    {
        return (x >= v1.x && y >= v1.y && z >= v1.z && w >= v1.w);
    }

    ROMANORENDER_FORCE_INLINE bool operator<(const Vec4F& v1) const noexcept
    {
        return (x < v1.x && y < v1.y && z < v1.z && w < v1.w);
    }

    ROMANORENDER_FORCE_INLINE bool operator<=(const Vec4F& v1) const noexcept
    {
        return (x <= v1.x && y <= v1.y && z <= v1.z && w <= v1.w);
    }

    ROMANORENDER_FORCE_INLINE Vec4F& operator+=(const Vec4F& v1) noexcept
    {
        x += v1.x;
        y += v1.y;
        z += v1.z;
        w += v1.w;
        return *this;
    }

    ROMANORENDER_FORCE_INLINE Vec4F& operator-=(const Vec4F& v1) noexcept
    {
        x -= v1.x;
        y -= v1.y;
        z -= v1.z;
        w -= v1.w;
        return *this;
    }

    ROMANORENDER_FORCE_INLINE Vec4F& operator*=(const Vec4F& v1) noexcept
    {
        x *= v1.x;
        y *= v1.y;
        z *= v1.z;
        w *= v1.w;
        return *this;
    }

    ROMANORENDER_FORCE_INLINE Vec4F& operator*=(float t) noexcept
    {
        x *= t;
        y *= t;
        z *= t;
        w *= t;
        return *this;
    }

    ROMANORENDER_FORCE_INLINE Vec4F& operator/=(const Vec4F& v1) noexcept
    {
        x /= v1.x;
        y /= v1.y;
        z /= v1.z;
        w /= v1.w;
        return *this;
    }

    ROMANORENDER_FORCE_INLINE Vec4F& operator/=(float t) noexcept
    {
        const float inv = maths::rcp_safef(t);
        x *= inv;
        y *= inv;
        z *= inv;
        w *= inv;
        return *this;
    }


    uint32_t as_uint32() const noexcept
    {
        uint32_t res;

        for(uint32_t i = 0; i < 4; i++)
        {
            res |= ((uint32_t)(maths::clampf((*this)[i]) * 255.0f)) << (i * 8);
        }

        return res;
    }
};

ROMANORENDER_FORCE_INLINE Vec4F operator*(const float t, const Vec4F& vec) noexcept
{
    return vec * t;
}

ROMANORENDER_FORCE_INLINE Vec4F operator/(const float t, const Vec4F& vec) noexcept
{
    return Vec4F(t / vec.x, t / vec.y, t / vec.z, t / vec.w);
}

ROMANORENDER_FORCE_INLINE float dot_vec4f(const Vec4F& v1, const Vec4F& v2) noexcept
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z + v1.w * v2.w;
}

ROMANORENDER_FORCE_INLINE float length_vec4f(const Vec4F& a) noexcept
{
    return maths::sqrtf(dot_vec4f(a, a));
}

ROMANORENDER_FORCE_INLINE float length2_vec4f(const Vec4F& a) noexcept { return dot_vec4f(a, a); }

ROMANORENDER_FORCE_INLINE Vec4F normalize_safe_vec4f(const Vec4F& v) noexcept
{
    float t = 1.0f / length_vec4f(v);
    return Vec4F(v.x * t, v.y * t, v.z * t, v.w * t);
}

ROMANORENDER_FORCE_INLINE Vec4F normalize_vec4f(const Vec4F& v) noexcept
{
    float t = maths::rsqrtf(dot_vec4f(v, v));
    return Vec4F(v.x * t, v.y * t, v.z * t, v.w * t);
}

ROMANORENDER_FORCE_INLINE float dist_vec4f(const Vec4F& a, const Vec4F& b) noexcept
{
    return maths::sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y)
                        + (a.z - b.z) * (a.z - b.z) + (a.w - b.w) * (a.w - b.w));
}

ROMANORENDER_FORCE_INLINE Vec4F abs_vec4f(const Vec4F& a) noexcept
{
    return Vec4F(maths::absf(a.x), maths::absf(a.y), maths::absf(a.z), maths::absf(a.w));
}

ROMANORENDER_FORCE_INLINE Vec4F sum_vec4f(const Vec4F& v1, const Vec4F& v2, const Vec4F& v3) noexcept
{
    return Vec4F((v1.x + v2.x + v3.x) * 0.3333f,
                 (v1.y + v2.y + v3.y) * 0.3333f,
                 (v1.z + v2.z + v3.z) * 0.3333f,
                 (v1.w + v2.w + v3.w) * 0.3333f);
}

ROMANORENDER_FORCE_INLINE Vec4F pow_vec4f(const Vec4F& v, const float p) noexcept
{
    return Vec4F(maths::powf(v.x, p), maths::powf(v.y, p), maths::powf(v.z, p), maths::powf(v.w, p));
}

ROMANORENDER_FORCE_INLINE Vec4F min_vec4f(const Vec4F& a, const Vec4F& b) noexcept
{
    return Vec4F(maths::minf(a.x, b.x), maths::minf(a.y, b.y), maths::minf(a.z, b.z), maths::minf(a.w, b.w));
}

ROMANORENDER_FORCE_INLINE Vec4F max_vec4f(const Vec4F& a, const Vec4F& b) noexcept
{
    return Vec4F(maths::maxf(a.x, b.x), maths::maxf(a.y, b.y), maths::maxf(a.z, b.z), maths::maxf(a.w, b.w));
}

ROMANORENDER_FORCE_INLINE Vec4F lerp_vec4ff(const Vec4F& a, const Vec4F& b, const float t) noexcept
{
    return Vec4F(maths::lerpf(a.x, b.x, t),
                 maths::lerpf(a.y, b.y, t),
                 maths::lerpf(a.z, b.z, t),
                 maths::lerpf(a.w, b.w, t));
}

ROMANORENDER_FORCE_INLINE Vec4F lerp_vec4fv(const Vec4F& a, const Vec4F& b, const Vec4F& t) noexcept
{
    return Vec4F(maths::lerpf(a.x, b.x, t.x),
                 maths::lerpf(a.y, b.y, t.y),
                 maths::lerpf(a.z, b.z, t.z),
                 maths::lerpf(a.w, b.w, t.w));
}

ROMANORENDER_FORCE_INLINE Vec4F rcp_vec4f(const Vec4F& v) noexcept
{
    const __m128 a = _mm_set_ps(v.w, v.z, v.y, v.x);
    const __m128 r = _mm_rcp_ps(a);

#if defined(__AVX2__)
    const __m128 result = _mm_mul_ps(r, _mm_fnmadd_ps(r, a, _mm_set1_ps(2.0f)));
#else
    const __m128 temp = _mm_sub_ps(_mm_set1_ps(2.0f), _mm_mul_ps(r, a));
    const __m128 result = _mm_mul_ps(r, temp);
#endif

    alignas(16) float components[4];
    _mm_store_ps(components, result);
    return Vec4F(components[0], components[1], components[2], components[3]);
}

ROMANORENDER_FORCE_INLINE bool isnan_vec4f(const Vec4F& v) noexcept
{
    const __m128 data = _mm_loadu_ps(std::addressof(v[0]));
    const __m128 cmp = _mm_cmpeq_ps(data, data);
    const int mask = _mm_movemask_ps(cmp);

    return mask != 0xF;
}

ROMANORENDER_FORCE_INLINE bool isinf_vec4f(const Vec4F& v) noexcept
{
    return maths::isinff(v.x) | maths::isinff(v.y) | maths::isinff(v.z) | maths::isinff(v.w);
}

#define ROMANORENDER_ABORT_IF_VEC4F_NAN(v) ROMANORENDER_ASSERT(!isnan_vec4f(v), "Vec4F contains nans")
#define ROMANORENDER_ABORT_IF_VEC4F_INF(v) ROMANORENDER_ASSERT(!isinf_vec4f(v), "Vec4F contains infs")

ROMANORENDER_FORCE_INLINE Vec4F default_if_nan_vec4f(const Vec4F& v, const Vec4F& v_if_nan) noexcept
{
    return isnan_vec4f(v) ? v_if_nan : v;
}

ROMANORENDER_NAMESPACE_END

template <>
struct fmt::formatter<romanorender::Vec4F>
{
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    auto format(romanorender::Vec4F& v, format_context& ctx) const
    {
        return format_to(ctx.out(), "{}, {}, {}, {}", v.x, v.y, v.z, v.w);
    }

    auto format(const romanorender::Vec4F& v, format_context& ctx) const
    {
        return format_to(ctx.out(), "{}, {}, {}, {}", v.x, v.y, v.z, v.w);
    }
};

#endif /* !defined(__ROMANORENDER_VEC4) */
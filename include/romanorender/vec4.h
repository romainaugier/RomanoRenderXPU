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

	const float& operator [] (int i) const { return (&x)[i]; }
	float& operator [] (int i) { return (&x)[i]; }

    uint32_t as_uint32() const noexcept 
    { 
        uint32_t res; 

#pragma omp simd
        for(uint32_t i = 0; i < 4; i++) 
        { 
            res |= ((uint32_t)(maths::clampf((*this)[i]) * 255.0f)) << (i * 8); 
        }
        
        return res; 
    }
};

ROMANORENDER_FORCE_INLINE Vec4F operator+(const Vec4F& vec, const Vec4F& other) noexcept { return Vec4F(vec.x + other.x, vec.y + other.y, vec.z + other.z, vec.w + other.w); }
ROMANORENDER_FORCE_INLINE Vec4F operator+(const Vec4F& vec, const float t) noexcept { return Vec4F(vec.x + t, vec.y + t, vec.z + t, vec.w + t); }
ROMANORENDER_FORCE_INLINE Vec4F operator-(const Vec4F& vec, const Vec4F& other) noexcept { return Vec4F(vec.x - other.x, vec.y - other.y, vec.z - other.z, vec.w - other.w); }
ROMANORENDER_FORCE_INLINE Vec4F operator-(const Vec4F& vec, const float t) noexcept { return Vec4F(vec.x - t, vec.y - t, vec.z - t, vec.w - t); }
ROMANORENDER_FORCE_INLINE Vec4F operator*(const Vec4F& vec, const Vec4F& other) noexcept { return Vec4F(vec.x * other.x, vec.y * other.y, vec.z * other.z, vec.w * other.w); }
ROMANORENDER_FORCE_INLINE Vec4F operator*(const Vec4F& vec, const float t) noexcept { return Vec4F(vec.x * t, vec.y * t, vec.z * t, vec.w * t); }
ROMANORENDER_FORCE_INLINE Vec4F operator*(const float t, const Vec4F& vec) noexcept { return Vec4F(vec.x * t, vec.y * t, vec.z * t, vec.w * t); }
ROMANORENDER_FORCE_INLINE Vec4F operator/(const Vec4F& vec, const Vec4F& other) noexcept { return Vec4F(vec.x / other.x, vec.y / other.y, vec.z / other.z, vec.w / other.w); }
ROMANORENDER_FORCE_INLINE Vec4F operator/(const Vec4F& vec, const float t) noexcept { return Vec4F(vec.x / t, vec.y / t, vec.z / t, vec.w / t); }
ROMANORENDER_FORCE_INLINE Vec4F operator/(const float t, const Vec4F& vec) noexcept { return Vec4F(t / vec.x, t / vec.y, t / vec.z, t / vec.w); }

ROMANORENDER_FORCE_INLINE bool operator==(const Vec4F& v0, const Vec4F& v1) noexcept { if (v0.x == v1.x && v0.y == v1.y && v0.z == v1.z && v0.w == v1.w) return true; else return false; }
ROMANORENDER_FORCE_INLINE bool operator>(const Vec4F& v0, const Vec4F& v1) noexcept { if (v0.x > v1.x && v0.y > v1.y && v0.z > v1.z && v0.w > v1.w) return true; else return false; }
ROMANORENDER_FORCE_INLINE bool operator>=(const Vec4F& v0, const Vec4F& v1) noexcept { if (v0.x >= v1.x && v0.y >= v1.y && v0.z >= v1.z && v0.w > v1.w) return true; else return false; }
ROMANORENDER_FORCE_INLINE bool operator<(const Vec4F& v0, const Vec4F& v1) noexcept { if (v0.x < v1.x && v0.y < v1.y && v0.z < v1.z && v0.w < v1.w) return true; else return false; }
ROMANORENDER_FORCE_INLINE bool operator<=(const Vec4F& v0, const Vec4F& v1) noexcept { if (v0.x <= v1.x && v0.y <= v1.y && v0.z <= v1.z && v0.w <= v1.w) return true; else return false; }

ROMANORENDER_FORCE_INLINE Vec4F operator+=(Vec4F& v0, Vec4F v1) noexcept { v0 = v0 + v1; return v0; }
ROMANORENDER_FORCE_INLINE Vec4F operator-=(Vec4F& v0, Vec4F v1) noexcept { v0 = v0 - v1; return v0; }
ROMANORENDER_FORCE_INLINE Vec4F operator*=(Vec4F& v0, Vec4F v1) noexcept { v0 = v0 * v1; return v0; }
ROMANORENDER_FORCE_INLINE Vec4F operator*=(Vec4F& v0, float t) noexcept { v0 = v0 * t; return v0; }
ROMANORENDER_FORCE_INLINE Vec4F operator/=(Vec4F& v0, Vec4F v1) noexcept { v0 = v0 / v1; return v0; }
ROMANORENDER_FORCE_INLINE Vec4F operator/=(Vec4F& v0, float t) noexcept { float inv = maths::rcpf(t); return Vec4F(v0.x * inv, v0.y * inv, v0.z * inv, v0.w * inv); }

ROMANORENDER_FORCE_INLINE float dot_vec4f(const Vec4F& v1, const Vec4F& v2) noexcept { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }
ROMANORENDER_FORCE_INLINE float length_vec4f(const Vec4F& a) noexcept { return maths::sqrtf(dot_vec4f(a, a)); }
ROMANORENDER_FORCE_INLINE float length2_vec4f(const Vec4F& a) noexcept { return dot_vec4f(a, a); }
ROMANORENDER_FORCE_INLINE Vec4F normalize_safe_vec4f(const Vec4F& v) noexcept { float t = 1.0f / length_vec4f(v); return Vec4F(v.x * t, v.y * t, v.z * t, v.w * t); }
ROMANORENDER_FORCE_INLINE Vec4F normalize_vec4f(const Vec4F& v) noexcept { float t = maths::rsqrtf(dot_vec4f(v, v)); return Vec4F(v.x * t, v.y * t, v.z * t, v.w * t); }
ROMANORENDER_FORCE_INLINE float dist_vec4f(const Vec4F& a, const Vec4F& b) noexcept { return maths::sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z) + (a.w - b.w) * (a.w - b.w)); }
ROMANORENDER_FORCE_INLINE Vec4F abs_vec4f(const Vec4F& a) noexcept { return Vec4F(maths::absf(a.x), maths::absf(a.y), maths::absf(a.z), maths::absf(a.w)); }
ROMANORENDER_FORCE_INLINE Vec4F sum_vec4f(const Vec4F& v1, const Vec4F& v2, const Vec4F& v3) noexcept { return Vec4F((v1.x + v2.x + v3.x) / 3, (v1.y + v2.y + v3.y) / 3, (v1.z + v2.z + v3.z) / 3, (v1.w + v2.w + v3.w) / 3); }
ROMANORENDER_FORCE_INLINE Vec4F pow_vec4f(const Vec4F& v, const float p) noexcept { return Vec4F(maths::powf(v.x, p), maths::powf(v.y, p), maths::powf(v.z, p), maths::powf(v.w, p)); }
ROMANORENDER_FORCE_INLINE Vec4F min_vec4f(const Vec4F& a, const Vec4F& b) noexcept { return Vec4F(maths::minf(a.x, b.x), maths::minf(a.y, b.y), maths::minf(a.z, b.z), maths::minf(a.w, b.w)); }
ROMANORENDER_FORCE_INLINE Vec4F max_vec4f(const Vec4F& a, const Vec4F& b) noexcept { return Vec4F(maths::maxf(a.x, b.x), maths::maxf(a.y, b.y), maths::maxf(a.z, b.z), maths::maxf(a.w, b.w)); }
ROMANORENDER_FORCE_INLINE Vec4F lerp_vec4ff(const Vec4F& a, const Vec4F& b, const float t) noexcept { return Vec4F(maths::lerpf(a.x, b.x, t), maths::lerpf(a.y, b.y, t), maths::lerpf(a.z, b.z, t), maths::lerpf(a.w, b.w, t)); }
ROMANORENDER_FORCE_INLINE Vec4F lerp_vec4fv(const Vec4F& a, const Vec4F& b, const Vec4F& t) noexcept { return Vec4F(maths::lerpf(a.x, b.x, t.x), maths::lerpf(a.y, b.y, t.y), maths::lerpf(a.z, b.z, t.z), maths::lerpf(a.w, b.w, t.w)); }
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

ROMANORENDER_NAMESPACE_END

template <>
struct fmt::formatter<romanorender::Vec4F>
{
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }
    auto format(romanorender::Vec4F& v, format_context& ctx) const { return format_to(ctx.out(), "{}, {}, {}, {}", v.x, v.y, v.z, v.w); }
    auto format(const romanorender::Vec4F& v, format_context& ctx) const { return format_to(ctx.out(), "{}, {}, {}, {}", v.x, v.y, v.z, v.w); }
};

#endif /* !defined(__ROMANORENDER_VEC4) */
#pragma once

// Simple simd library using sse, avx and avx2 intrinsics

#include "immintrin.h"
#include "intrin.h"

#include "decl.h"

// SIMD Float 4

struct vfloat4
{
    __m128 v; // data

    FORCEINLINE vfloat4() {}
    FORCEINLINE vfloat4(const vfloat4& other) { v = other.v; }
    FORCEINLINE vfloat4& operator =(const vfloat4& other) { v = other.v; return *this; }

    FORCEINLINE vfloat4(__m128 a) : v(a) {}
    FORCEINLINE vfloat4(float a) : v(_mm_set1_ps(a)) {}
    FORCEINLINE operator const __m128&() const { return v; } 
    FORCEINLINE operator __m128&() { return v; } 

    FORCEINLINE static vfloat4 set1(float a) { return _mm_set1_ps(a); }
    FORCEINLINE static vfloat4 set(float a, float b, float c, float d) { return _mm_set_ps(a, b, c, d); }

    FORCEINLINE static vfloat4 load(const float* ptr) { return _mm_load_ps(ptr); }
    FORCEINLINE static vfloat4 loadu(const float* ptr) { return _mm_loadu_ps(ptr); }

    FORCEINLINE static void store(float* ptr, const vfloat4& v) { return _mm_store_ps(ptr, v); }
    FORCEINLINE static void storeu(float* ptr, const vfloat4& v) { return _mm_storeu_ps(ptr, v); }

    FORCEINLINE static vfloat4 broadcast(const float* ptr) { return _mm_broadcast_ss(ptr); }

    FORCEINLINE static vfloat4 abs(const vfloat4& a) { return _mm_and_ps(a, _mm_castsi128_ps(_mm_set1_epi32(0x7fffffff))); }

    FORCEINLINE static vfloat4 rcp(const vfloat4& a) { const vfloat4 r = _mm_rcp_ps(a); return _mm_fmadd_ps(r, _mm_fnmadd_ps(a, r, vfloat4(1.0f)), r); } // AVX2

    FORCEINLINE static vfloat4 sqr(const vfloat4& a) { return _mm_mul_ps(a, a); }

    FORCEINLINE static vfloat4 sqrt(const vfloat4& a) { return _mm_sqrt_ps(a); }

    FORCEINLINE static vfloat4 rsqrt(const vfloat4& a) { vfloat4 r = _mm_rsqrt_ps(a); return _mm_fmadd_ps(vfloat4(1.5f), r, _mm_mul_ps(_mm_mul_ps(_mm_mul_ps(a, vfloat4(-0.5f)), r), sqr(r))); } // AVX2

    FORCEINLINE vfloat4 operator +(const vfloat4& a, const vfloat4& b) { return _mm_add_ps(a, b); }
    FORCEINLINE vfloat4 operator +(const vfloat4& a, float b) { return a + vfloat4(b); }
    FORCEINLINE vfloat4 operator +(float a, const vfloat4& b) { return vfloat4(a) + b; }

    FORCEINLINE vfloat4 operator -(const vfloat4& a, const vfloat4& b) { return _mm_sub_ps(a, b); }
    FORCEINLINE vfloat4 operator -(const vfloat4& a, float b) { return a - vfloat4(b); }
    FORCEINLINE vfloat4 operator -(const float a, const vfloat4& b) { return vfloat4(a) - b; }

    FORCEINLINE vfloat4 operator *(const vfloat4& a, const vfloat4& b) { return _mm_mul_ps(a, b); }
    FORCEINLINE vfloat4 operator *(const vfloat4& a, float b) { return a * vfloat4(b); }
    FORCEINLINE vfloat4 operator *(float a, const vfloat4& b) { return vfloat4(a) * b; }

    FORCEINLINE vfloat4 operator /(const vfloat4& a, const vfloat4& b) { return _mm_div_ps(a, b); }
    FORCEINLINE vfloat4 operator /(const vfloat4& a, float b) { return a / vfloat4(b); }
    FORCEINLINE vfloat4 operator /(float a, const vfloat4& b) { return vfloat4(a) / b; }

    FORCEINLINE vfloat4 operator &(const vfloat4& a, const vfloat4& b) { return _mm_and_ps(a, b); }
    FORCEINLINE vfloat4 operator |(const vfloat4& a, const vfloat4& b) { return _mm_or_ps(a, b); }
    FORCEINLINE vfloat4 operator ^(const vfloat4& a, const vfloat4& b) { return _mm_xor_ps(a, b); }

    FORCEINLINE static vfloat4 min(const vfloat4& a, const vfloat4& b) { return _mm_min_ps(a, b); }
    FORCEINLINE static vfloat4 min(const vfloat4& a, float b) { return vfloat4::min(a, vfloat4(b)); }
    FORCEINLINE static vfloat4 min(float a, const vfloat4& b) { return vfloat4::min(vfloat4(a), b); }
    
    FORCEINLINE static vfloat4 max(const vfloat4& a, const vfloat4& b) { return _mm_max_ps(a, b); }
    FORCEINLINE static vfloat4 max(const vfloat4& a, float b) { return vfloat4::max(a, vfloat4(b)); }
    FORCEINLINE static vfloat4 max(float a, const vfloat4& b) { return vfloat4::max(vfloat4(a), b); }
};

// SIMD Float 8

struct vfloat8
{
    __m256 v; // data

    FORCEINLINE vfloat8() {}
    FORCEINLINE vfloat8(const vfloat8& other) { v = other.v; }
    FORCEINLINE vfloat8& operator =(const vfloat8& other) { v = other.v; return *this; }

    FORCEINLINE vfloat8(__m256 a) : v(a) {}
    FORCEINLINE operator const __m256&() const { return v; } 
    FORCEINLINE operator __m256&() { return v; } 

    FORCEINLINE vfloat8 set1(float a) { return _mm256_set1_ps(a); }
    FORCEINLINE vfloat8 set(float a, float b, float c, float d, float e, float f, float g, float h) { return _mm256_set_ps(a, b, c, d, e, f, g, h); }

    FORCEINLINE vfloat8 load(const float* ptr) { return _mm256_load_ps(ptr); }
    FORCEINLINE vfloat8 loadu(const float* ptr) { return _mm256_loadu_ps(ptr); }

    FORCEINLINE void store(float* ptr, const vfloat8& v) { return _mm256_store_ps(ptr, v); }
    FORCEINLINE void storeu(float* ptr, const vfloat8& v) { return _mm256_storeu_ps(ptr, v); }

    FORCEINLINE vfloat8 broadcast(const float* ptr) { return _mm256_broadcast_ss(ptr); }
};
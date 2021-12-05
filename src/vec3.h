#pragma once

#include <cmath>
#include "decl.h"

#undef min
#undef max

struct vec3
{
	float x, y, z;

	vec3() : x(0.0f), y(0.0f), z(0.0f) {}
	vec3(float t) : x(t), y(t), z(t) {}
	vec3(float X, float Y, float Z) : x(X), y(Y), z(Z) {}

	const vec3 operator-() const { return vec3(-x, -y, -z); }

	const float& operator [] (int i) const { return (&x)[i]; }
	float& operator [] (int i) { return (&x)[i]; }
};

// maths operators
FORCEINLINE vec3 operator+(const vec3& vec, const vec3& other) noexcept { return vec3(vec.x + other.x, vec.y + other.y, vec.z + other.z); }
FORCEINLINE vec3 operator+(const vec3& vec, const float t) noexcept { return vec3(vec.x + t, vec.y + t, vec.z + t); }
FORCEINLINE vec3 operator-(const vec3& vec, const vec3& other) noexcept { return vec3(vec.x - other.x, vec.y - other.y, vec.z - other.z); }
FORCEINLINE vec3 operator-(const vec3& vec, const float t) noexcept { return vec3(vec.x - t, vec.y - t, vec.z - t); }
FORCEINLINE vec3 operator*(const vec3& vec, const vec3& other) noexcept { return vec3(vec.x * other.x, vec.y * other.y, vec.z * other.z); }
FORCEINLINE vec3 operator*(const vec3& vec, const float t) noexcept { return vec3(vec.x * t, vec.y * t, vec.z * t); }
FORCEINLINE vec3 operator*(const float t, const vec3& vec) noexcept { return vec3(vec.x * t, vec.y * t, vec.z * t); }
FORCEINLINE vec3 operator/(const vec3& vec, const vec3& other) noexcept { return vec3(vec.x / other.x, vec.y / other.y, vec.z / other.z); }
FORCEINLINE vec3 operator/(const vec3& vec, const float t) noexcept { return vec3(vec.x / t, vec.y / t, vec.z / t); }
FORCEINLINE vec3 operator/(const float t, const vec3& vec) noexcept { return vec3(t / vec.x, t / vec.y, t / vec.z); }

FORCEINLINE bool operator==(const vec3& v0, const vec3& v1) noexcept { if (v0.x == v1.x && v0.y == v1.y && v0.z == v1.z) return true; else return false; }
FORCEINLINE bool operator>(const vec3& v0, const vec3& v1) noexcept { if (v0.x > v1.x && v0.y > v1.y && v0.z > v1.z) return true; else return false; }
FORCEINLINE bool operator>=(const vec3& v0, const vec3& v1) noexcept { if (v0.x >= v1.x && v0.y >= v1.y && v0.z >= v1.z) return true; else return false; }
FORCEINLINE bool operator<(const vec3& v0, const vec3& v1) noexcept { if (v0.x < v1.x && v0.y < v1.y && v0.z < v1.z) return true; else return false; }
FORCEINLINE bool operator<=(const vec3& v0, const vec3& v1) noexcept { if (v0.x <= v1.x && v0.y <= v1.y && v0.z <= v1.z) return true; else return false; }

FORCEINLINE vec3 operator+=(vec3& v0, vec3 v1) noexcept { v0 = v0 + v1; return v0; }
FORCEINLINE vec3 operator-=(vec3& v0, vec3 v1) noexcept { v0 = v0 - v1; return v0; }
FORCEINLINE vec3 operator*=(vec3& v0, vec3 v1) noexcept { v0 = v0 * v1; return v0; }
FORCEINLINE vec3 operator*=(vec3& v0, float t) noexcept { v0 = v0 * t; return v0; }
FORCEINLINE vec3 operator/=(vec3& v0, vec3 v1) noexcept { v0 = v0 / v1; return v0; }
FORCEINLINE vec3 operator/=(vec3& v0, float t) noexcept { float inv = 1.0f / t; return vec3(v0.x *= inv, v0.y *= inv, v0.z *= inv); }

// utility functions
FORCEINLINE float dot(const vec3& v1, const vec3& v2) noexcept { return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z; }
FORCEINLINE float length(const vec3& a) noexcept { return sqrtf(dot(a, a)); }
FORCEINLINE float length2(const vec3& a) noexcept { return dot(a, a); }
FORCEINLINE vec3 normalize(const vec3& v) noexcept { float t = 1.0f / length(v); return vec3(v.x * t, v.y * t, v.z * t); }
FORCEINLINE vec3  cross(const vec3& v1, const vec3& v2) noexcept { return vec3(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x); }
FORCEINLINE float dist(const vec3& a, const vec3& b) noexcept { return sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z)); }
FORCEINLINE vec3  abs(const vec3& a) noexcept { return vec3(abs(a.x), abs(a.y), abs(a.z)); }
FORCEINLINE vec3  sum(const vec3& v1, const vec3& v2, const vec3& v3) noexcept { return vec3((v1.x + v2.x + v3.x) / 3, (v1.y + v2.y + v3.y) / 3, (v1.z + v2.z + v3.z) / 3); }
FORCEINLINE vec3  powvec3(const vec3& v, const float p) noexcept { return vec3(powf(v.x, p), powf(v.y, p), powf(v.z, p)); }
FORCEINLINE vec3  min(const vec3& a, const vec3& b) noexcept { return vec3(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z); }
FORCEINLINE vec3  max(const vec3& a, const vec3& b) noexcept { return vec3(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z); }
#pragma once

#include "vec3.h"
#include <immintrin.h>
#include <limits>
#include <cstring>

#define PI 3.14159265358979323846f
#define INVPI 1.0f / 3.14159265358979323846f

#define INF std::numeric_limits<float>::infinity()

FORCEINLINE float deg2rad(const float deg) { return deg * PI / 180; }
FORCEINLINE float rad2deg(const float rad) { return rad * 180 / PI; }

template <typename T>
FORCEINLINE T fit(T s, T a1, T a2, T b1, T b2) { return b1 + ((s - a1) * (b2 - b1)) / (a2 - a1); }

template <typename T>
FORCEINLINE T fit01(T x, T a, T b) { return x * (b - a) + a; }

template <typename T>
FORCEINLINE T lerp(T a, T b, float t) { return (1 - t) * a + t * b; }

template <typename T>
FORCEINLINE T clamp(T n, float lower, float upper) { return fmax(lower, fmin(n, upper)); }

FORCEINLINE float min(float a, float b) { return a < b ? a : b; }
FORCEINLINE float max(float a, float b) { return a > b ? a : b; }
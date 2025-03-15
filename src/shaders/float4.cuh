
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
    return make_float4(vec.x / other.x, vec.y / other.y, vec.z / other.z, vec.w / other.w);
}

__device__ __forceinline__ float4 operator/(const float4& vec, const float t) noexcept
{
    return make_float4(vec.x / t, vec.y / t, vec.z / t, vec.w / t);
}

__device__ __forceinline__ float4 operator/(const float t, const float4& vec) noexcept
{
    return make_float4(t / vec.x, t / vec.y, t / vec.z, t / vec.w);
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
    if(v0.x >= v1.x && v0.y >= v1.y && v0.z >= v1.z && v0.w > v1.w)
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
    return make_float4(v0.x / t, v0.y / t, v0.z / t, v0.w / t);
}

__device__ __forceinline__ float dot_vec4f(const float4& v1, const float4& v2) noexcept
{
    return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
}

__device__ __forceinline__ float4 cross_vec4f(const float4& v1, const float4& v2) noexcept
{
    return make_float4(v1.y * v2.z - v1.z * v2.y, v1.z * v2.x - v1.x * v2.z, v1.x * v2.y - v1.y * v2.x, 0.0f);
}

__device__ __forceinline__ float length_vec4f(const float4& a) noexcept { return __sqrtf(dot_vec4f(a, a)); }

__device__ __forceinline__ float length2_vec4f(const float4& a) noexcept { return dot_vec4f(a, a); }

__device__ __forceinline__ float4 normalize_safe_vec4f(const float4& v) noexcept
{
    float t = 1.0f / length_vec4f(v);
    return make_float4(v.x * t, v.y * t, v.z * t, v.w * t);
}

__device__ __forceinline__ float4 normalize_vec4f(const float4& v) noexcept
{
    float t = __frsqrt_rn(dot_vec4f(v, v));
    return make_float4(v.x * t, v.y * t, v.z * t, v.w * t);
}

__device__ __forceinline__ float dist_vec4f(const float4& a, const float4& b) noexcept
{
    return __sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y) + (a.z - b.z) * (a.z - b.z)
                   + (a.w - b.w) * (a.w - b.w));
}

__device__ __forceinline__ float4 sum_vec4f(const float4& v1, const float4& v2, const float4& v3) noexcept
{
    return make_float4(
        (v1.x + v2.x + v3.x) / 3, (v1.y + v2.y + v3.y) / 3, (v1.z + v2.z + v3.z) / 3, (v1.w + v2.w + v3.w) / 3);
}

__device__ __forceinline__ float4 pow_vec4f(const float4& v, const float p) noexcept
{
    return make_float4(__powf(v.x, p), __powf(v.y, p), __powf(v.z, p), __powf(v.w, p));
}

#define __minf(x, y) (x > y ? y : x)

__device__ __forceinline__ float4 min_vec4f(const float4& a, const float4& b) noexcept
{
    return make_float4(__minf(a.x, b.x), __minf(a.y, b.y), __minf(a.z, b.z), __minf(a.w, b.w));
}

#define __maxf(x, y) (x < y ? y : x)

__device__ __forceinline__ float4 max_vec4f(const float4& a, const float4& b) noexcept
{
    return make_float4(__maxf(a.x, b.x), __maxf(a.y, b.y), __maxf(a.z, b.z), __maxf(a.w, b.w));
}

#define __lerpf(x, y, t) ((1.0f - t) * x + t * y)

__device__ __forceinline__ float4 lerp_vec4ff(const float4& a, const float4& b, const float t) noexcept
{
    return make_float4(__lerpf(a.x, b.x, t), __lerpf(a.y, b.y, t), __lerpf(a.z, b.z, t), __lerpf(a.w, b.w, t));
}

__device__ __forceinline__ float4 lerp_vec4fv(const float4& a, const float4& b, const float4& t) noexcept
{
    return make_float4(__lerpf(a.x, b.x, t.x), __lerpf(a.y, b.y, t.y), __lerpf(a.z, b.z, t.z), __lerpf(a.w, b.w, t.w));
}

__device__ __forceinline__ float4 rcp_vec4f(const float4& v) noexcept
{
    return make_float4(__frcp_rn(v.x), __frcp_rn(v.y), __frcp_rn(v.z), __frcp_rn(v.w));
}
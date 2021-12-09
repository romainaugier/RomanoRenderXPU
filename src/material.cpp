#include "material.h"

vec3 MaterialDiffuse::Sample(const vec3& N, const vec3& wi, const float rx, const float ry, const float rz) const
{
    return SampleHemisphere(N, rx, ry);
}

vec3 MaterialDiffuse::Eval(const vec3& N, const vec3& wo, const float rx, const float ry, const float rz) const 
{
    return m_Color * maths::max(0.0f, dot(N, wo)) * maths::constants::one_over_pi;
}

vec3 MaterialReflective::Sample(const vec3& N, const vec3& wi, const float rx, const float ry, const float rz) const
{
    // const vec3 randomDir = SampleHemisphere(N, rx, ry);
    // const vec3 newNormal = lerp(N, randomDir, m_Roughness);
    return wi - 2 * dot(wi, N) * N; 
}

vec3 MaterialReflective::Eval(const vec3& N, const vec3& wo, const float rx, const float ry, const float rz) const
{
    return m_Color;
}

vec3 MaterialDielectric::Sample(const vec3& N, const vec3& wi, const float rx, const float ry, const float rz) const
{
    const vec3 nWi = normalize(wi);
    const bool inside = rz < 0.0f;
    const float eta = inside ? 1.0f / m_Ior : m_Ior / 1.0f;
    const vec3 nT = inside ? N : -N;
    const float c1 = dot(nT, nWi);
    const float w = c1 * eta;
    const float c2m = (w - eta) * (w + eta);
    if(c2m < -1.0f) return wi;
    else return eta * nWi + (w - maths::sqrt(1.0f + c2m)) * nT;
}

vec3 MaterialDielectric::Eval(const vec3& N, const vec3& wi, const float rx, const float ry, const float rz) const
{
    return m_Color;
}
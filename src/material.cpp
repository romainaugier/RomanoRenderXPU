#include "material.h"

vec3 MaterialDiffuse::Sample(const vec3& N, const vec3& wi, const float rx, const float ry, const float rz) const
{
    return SampleHemisphere(N, rx, ry);
}

vec3 MaterialDiffuse::Eval(const vec3& N, const vec3& wo, const float rx, const float ry, const float rz) const 
{
    return m_Color * max(0.0f, dot(N, wo)) * INVPI;
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
    const float NdotWI = dot(N, wi);
    const vec3 normal = NdotWI > 0.0f ? -N : N;
    const float eta = NdotWI > 0.0f ? 1.0f / m_Ior : m_Ior / 1.0f;
    const float c1 = dot(normal, wi);
    const float w = c1 * eta;
    const float c2m = (w - eta) * (w + eta);
    if(c2m < -1.0f) return wi;
    else return eta * wi + (w - sqrtf(1.0f + c2m)) * normal;
}

vec3 MaterialDielectric::Eval(const vec3& N, const vec3& wi, const float rx, const float ry, const float rz) const
{
    return m_Color;
}
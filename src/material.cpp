#include "material.h"

vec3 MaterialDiffuse::Sample(const vec3& N, const vec3& wi, const float rx, const float ry, const float rz) const
{
    const float randoms[2] = { rx, ry };
    return sample_ray_in_hemisphere(N, randoms);
}

vec3 MaterialDiffuse::Eval(const vec3& N, const vec3& wo, const float rx, const float ry, const float rz) const 
{
    return m_Color * max(0.0f, dot(N, wo)) * INVPI;
}

vec3 MaterialReflective::Sample(const vec3& N, const vec3& wi, const float rx, const float ry, const float rz) const
{
    // const float randoms[2] = { rx, ry };
    // const vec3 randomDir = sample_ray_in_hemisphere(N, randoms);
    // const vec3 newNormal = lerp(N, randomDir, m_Roughness);
    return wi - 2 * dot(wi, N) * N; 
}

vec3 MaterialReflective::Eval(const vec3& N, const vec3& wo, const float rx, const float ry, const float rz) const
{
    return m_Color;
}

vec3 MaterialDielectric::Sample(const vec3& N, const vec3& wi, const float rx, const float ry, const float rz) const
{

}

vec3 MaterialDielectric::Eval(const vec3& N, const vec3& wi, const float rx, const float ry, const float rz) const
{

}
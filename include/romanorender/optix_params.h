#pragma once

#if !defined(__ROMANORENDER_OPTIX_PARAMS)
#define __ROMANORENDER_OPTIX_PARAMS

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

#define NUM_PMJ02_SEQUENCES 128

struct GeometryData
{
    uint32_t id;
    const float4* vertices;
    const uint3* indices;
    const float3* normals;

    GeometryData() = default;

    GeometryData(const uint32_t id, 
                 const CUdeviceptr vertices,
                 const CUdeviceptr indices,
                 const CUdeviceptr normals)
        : id(id), 
          vertices(reinterpret_cast<const float4*>(vertices)),
          indices(reinterpret_cast<const uint3*>(indices)),
          normals(reinterpret_cast<const float3*>(normals))
    {
    }
};

struct OptixParams
{
    float camera_transform[16];
    float camera_fov;
    float camera_aspect;
    float4* pixels;

    const float2* pmj_samples[NUM_PMJ02_SEQUENCES];

    OptixTraversableHandle handle;

    size_t current_sample;
    uint64_t seed;
};

#endif // !defined(__ROMANORENDER_OPTIX_PARAMS)
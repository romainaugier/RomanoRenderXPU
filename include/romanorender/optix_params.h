#pragma once

#if !defined(__ROMANORENDER_OPTIX_PARAMS)
#define __ROMANORENDER_OPTIX_PARAMS

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

struct GeometryData
{
    uint32_t id;
    const float4* vertices;
    const uint3* indices;

    GeometryData() = default;

    GeometryData(const uint32_t id, const CUdeviceptr vertices, const CUdeviceptr indices)
        : id(id), vertices(reinterpret_cast<const float4*>(vertices)), indices(reinterpret_cast<const uint3*>(indices))
    {
    }
};

struct OptixParams
{
    float camera_transform[16];
    float camera_fov;
    float camera_aspect;
    float4* pixels;
    OptixTraversableHandle handle;

    size_t current_sample;
};

#endif // !defined(__ROMANORENDER_OPTIX_PARAMS)
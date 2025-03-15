#pragma once

#if !defined(__ROMANORENDER_OPTIX_PARAMS)
#define __ROMANORENDER_OPTIX_PARAMS

#include <cuda_runtime.h>
#include <optix.h>
#include <optix_stubs.h>

struct OptixParams
{
    float camera_transform[16];
    float fov;
    float aspect;
    float4* pixels;
    OptixTraversableHandle handle;
};

#endif // !defined(__ROMANORENDER_OPTIX_PARAMS)
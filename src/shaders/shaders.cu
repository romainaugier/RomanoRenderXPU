#include "romanorender/optix_params.h"

#include "payload.cuh"
#include "random.cuh"
#include "mat44.cuh"
#include "float4.cuh"
#include "float3.cuh"

#include <curand_kernel.h>
#include <optix.h>
#include <vector_types.h>

extern "C" __constant__ OptixParams params;

__device__ float3 get_ray_dir(const float aspect, 
                              const float fov,
                              const Mat44F& transform, 
                              const float rx, 
                              const float ry)
{
    const uint3 launch_index = optixGetLaunchIndex();
    const uint3 launch_dims = optixGetLaunchDimensions();

    const float ndc_x = (2.0f * ((float)launch_index.x + rx) / (float)launch_dims.x - 1.0f) * aspect;
    const float ndc_y = 1.0f - 2.0f * ((float)launch_index.y + ry) / (float)launch_dims.y;

    const float tan_half_fov = __tanf(deg2radf(fov * 0.5f));
    const float px = ndc_x * tan_half_fov;
    const float py = ndc_y * tan_half_fov;

    const float3 direction = make_float3(px, py, -1.0f);

    return normalize_safe_float3(transform.transform_dir(direction));
}

extern "C" __global__ void __raygen__rg()
{
    const uint3 launch_index = optixGetLaunchIndex();
    const uint3 launch_dims = optixGetLaunchDimensions();

    const unsigned long long seed = params.seed;
    const unsigned long long sequence = launch_index.x + launch_index.y * launch_dims.x;
    const unsigned long long offset = params.current_sample + launch_index.z;

    const float rand_x = random_float_01(seed + sequence + offset);
    const float rand_y = random_float_01(seed + sequence + offset + 1);

    RayData ray_data;

    const Mat44F transform(params.camera_transform);

    const float3 ray_dir = get_ray_dir(params.camera_aspect, params.camera_fov, transform, rand_x, rand_y);
    const float3 ray_pos = make_float3(params.camera_transform[12], params.camera_transform[13], params.camera_transform[14]);

    uint2 payload = split_ptr(&ray_data);

    optixTrace(params.handle, // Scene acceleration structure
               ray_pos,       // Ray origin
               ray_dir,       // Ray direction
               0.0f,          // tMin
               1e16f,         // tMax
               0.0f,          // Ray time (for motion blur, unused here)
               OptixVisibilityMask(255),
               OPTIX_RAY_FLAG_NONE,
               0,
               1,
               0,
               payload.x,
               payload.y);

    const unsigned int pixel_idx = launch_index.x + launch_index.y * launch_dims.x;
    params.pixels[pixel_idx] = lerp_float4f(params.pixels[pixel_idx],
                                            ray_data.color,
                                            1.0f / (float)(params.current_sample + launch_index.z));
}

extern "C" __global__ void __miss__ms()
{
    RayData* ray_data = merge_ptr(optixGetPayload_0(), optixGetPayload_1());

    ray_data->color = make_float4(0.0f);
}

__device__ float3 get_normal(const GeometryData* geom_data, 
                             const unsigned int primitive,
                             const float2 uv)
{
    const uint3 indices = geom_data->indices[primitive];

    if(geom_data->normals == 0)
    {
        const float4 v0 = geom_data->vertices[indices.x];
        const float4 v1 = geom_data->vertices[indices.y];
        const float4 v2 = geom_data->vertices[indices.z];

        const float4 edge0 = v1 - v0;
        const float4 edge1 = v2 - v0;
        const float4 object_normal = normalize_safe_float4(cross_float4(edge0, edge1));

        return make_float3(object_normal);
    }
    else
    {
        const float3 n0 = geom_data->normals[indices.x];
        const float3 n1 = geom_data->normals[indices.y];
        const float3 n2 = geom_data->normals[indices.z];

        const float w = 1.0f - uv.x - uv.y;

        return n0 * w + n1 * uv.x + n2 * uv.y;
    }
}

extern "C" __global__ void __closesthit__ch()
{
    RayData* ray_data = merge_ptr(optixGetPayload_0(), optixGetPayload_1());

    const GeometryData* geom_data = reinterpret_cast<const GeometryData*>(optixGetSbtDataPointer());

    const float3 normal = optixTransformNormalFromObjectToWorldSpace(get_normal(geom_data, 
                                                                                optixGetPrimitiveIndex(),
                                                                                optixGetTriangleBarycentrics()));

    const float3 color = (normalize_safe_float3(normal) + 0.5f) / 2.0f;

    ray_data->color = make_float4(color, 1.0f);
}
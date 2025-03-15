#include "romanorender/optix_params.h"

#include "float4.cuh"
#include "payload.cuh"

#include <optix.h>
#include <vector_types.h>

extern "C" __constant__ OptixParams params;

inline __device__ float deg2radf(float degrees) { return degrees / M_PI * 180.0f; }

inline __device__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }

inline __device__ float3 operator+(float3 a, float b) { return make_float3(a.x + b, a.y + b, a.z + b); }

inline __device__ float3 operator*(float s, float3 v) { return make_float3(s * v.x, s * v.y, s * v.z); }

inline __device__ float3 operator/(float3 v, float s) { return make_float3(v.x / s, v.y / s, v.z / s); }

inline __device__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

inline __device__ float3 normalize(const float3 v)
{
    const float norm = __frsqrt_rn(v.x * v.x + v.y * v.y + v.z * v.z);

    return make_float3(v.x * norm, v.y * norm, v.z * norm);
}

__device__ float3 mat44f_mul_dir(const float* M, const float3& v) noexcept
{
    return make_float3(v.x * M[0] + v.y * M[4] + v.z * M[8],
                       v.x * M[1] + v.y * M[5] + v.z * M[9],
                       v.x * M[2] + v.y * M[6] + v.z * M[10]);
}

__device__ float3 get_ray_dir(float aspect, float fov, float* transform)
{
    uint3 launch_index = optixGetLaunchIndex();
    uint3 launch_dims = optixGetLaunchDimensions();

    const float ndc_x = (2.0f * ((float)launch_index.x + 0.5f) / (float)launch_dims.x - 1.0f) * aspect;
    const float ndc_y = 1.0f - 2.0f * ((float)launch_index.y + 0.5f) / (float)launch_dims.y;

    const float tan_half_fov = __tanf(deg2radf(fov * 0.5f));
    const float px = ndc_x * tan_half_fov;
    const float py = ndc_y * tan_half_fov;

    float3 direction = make_float3(px, py, -1.0f);

    return mat44f_mul_dir(transform, direction);
}

extern "C" __global__ void __raygen__rg()
{
    uint3 launch_index = optixGetLaunchIndex();
    uint3 launch_dims = optixGetLaunchDimensions();

    RayData ray_data;

    float3 ray_dir = get_ray_dir(params.camera_aspect, params.camera_fov, params.camera_transform);
    float3 ray_pos = make_float3(params.camera_transform[3], params.camera_transform[7], params.camera_transform[11]);

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
               0, // SBT offsets and stride
               payload.x,
               payload.y);

    unsigned int pixel_idx = launch_index.x + launch_index.y * launch_dims.x;
    params.pixels[pixel_idx] = ray_data.color;
}

extern "C" __global__ void __miss__ms()
{
    RayData* ray_data = merge_ptr(optixGetPayload_0(), optixGetPayload_1());

    ray_data->color = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
}

extern "C" __global__ void __closesthit__ch()
{
    RayData* ray_data = merge_ptr(optixGetPayload_0(), optixGetPayload_1());
    const GeometryData* geomData = reinterpret_cast<const GeometryData*>(optixGetSbtDataPointer());

    const uint3 indices = geomData->indices[optixGetPrimitiveIndex()];

    const float4 v0 = geomData->vertices[indices.x];
    const float4 v1 = geomData->vertices[indices.y];
    const float4 v2 = geomData->vertices[indices.z];

    const float4 edge0 = v1 - v0;
    const float4 edge1 = v2 - v0;
    const float4 objectNormal = normalize_vec4f(cross_vec4f(edge0, edge1));

    const float3 normal
        = optixTransformNormalFromObjectToWorldSpace(make_float3(objectNormal.x, objectNormal.y, objectNormal.z));

    const float3 color = (normal + 0.5f) / 2.0f;

    ray_data->color = lerp_vec4ff(
        ray_data->color, make_float4(color.x, color.y, color.z, 0.0f), 1.0f / (float)params.current_sample);
}
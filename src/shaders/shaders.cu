#include "romanorender/optix_params.h"

#include <optix.h>
#include <vector_types.h>

extern "C" __constant__ OptixParams params;

inline __device__ float deg2radf(float degrees) { return degrees / M_PI * 180.0f; }

inline __device__ float3 operator+(float3 a, float3 b) { return make_float3(a.x + b.x, a.y + b.y, a.z + b.z); }

inline __device__ float3 operator*(float s, float3 v) { return make_float3(s * v.x, s * v.y, s * v.z); }

inline __device__ float3 cross(float3 a, float3 b)
{
    return make_float3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

inline __device__ float3 normalize(float3 v)
{
    float len = sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    if(len > 0.0f)
    {
        float invLen = 1.0f / len;
        return make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
    }

    return v;
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

    const float tan_half_fov = tanf(deg2radf(fov * 0.5f));
    const float px = ndc_x * tan_half_fov;
    const float py = ndc_y * tan_half_fov;

    float3 direction = make_float3(px, py, -1.0f);

    return mat44f_mul_dir(transform, direction);
}

extern "C" __global__ void __raygen__rg()
{
    uint3 launch_index = optixGetLaunchIndex();
    uint3 launch_dims = optixGetLaunchDimensions();

    float3 ray_dir = get_ray_dir(params.aspect, params.fov, params.camera_transform);
    float3 ray_pos = make_float3(params.camera_transform[3], params.camera_transform[7], params.camera_transform[11]);

    unsigned int p0, p1;

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
               p0,
               p1 // Payload
    );

    // Write result to pixel buffer (assuming payload encodes color)
    float4 color = make_float4(__int_as_float(p0), __int_as_float(p1), 0.0f, 1.0f);
    unsigned int pixel_idx = launch_index.x + launch_index.y * launch_dims.x;
    params.pixels[pixel_idx] = color;
}

extern "C" __global__ void __miss__ms()
{
    // Set payload to a background color (e.g., blue)
    optixSetPayload_0(__float_as_int(0.0f)); // R
    optixSetPayload_1(__float_as_int(0.0f)); // G
}

extern "C" __global__ void __closesthit__ch()
{
    // Set payload to a hit color (e.g., red)
    optixSetPayload_0(__float_as_int(1.0f)); // R
    optixSetPayload_1(__float_as_int(0.0f)); // G
}
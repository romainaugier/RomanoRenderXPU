#include "romanorender/integrator.h"
#include "romanorender/sampling.h"

#include "stdromano/logger.h"
#include "stdromano/random.h"

ROMANORENDER_NAMESPACE_BEGIN

#define CLAY Vec3F(0.4f, 0.4f, 0.4f)

Vec4F integrator_pathtrace(const Scene* scene, const uint16_t x, const uint16_t y, const uint32_t sample, const uint16_t max_bounces) noexcept
{
    Vec3F color(0.0f);

    float alpha = 0.0f;
    const uint32_t pixel_id = scene->get_camera()->get_xres() * y + x;
    const uint32_t random_offset = pcg_random_uint32(pixel_id);

    const Vec2F pixel_sample = sampler().get_pmj02_sample(pixel_id * random_offset * 0x4738, sample);
    Vec3F ray_origin = scene->get_camera()->get_ray_origin();
    Vec3F ray_dir = scene->get_camera()->get_ray_direction(x, y, xoshiro_next_float(), xoshiro_next_float());
    tinybvh::Ray ray(ray_origin, ray_dir);

    Vec3F throughput(1.0f, 1.0f, 1.0f);
    Vec3F hit_p, world_n;

    for(uint16_t bounce = 0; bounce < max_bounces; ++bounce)
    {
        const int32_t cost = scene->intersect(ray);

        if(ray.hit.t >= BVH_FAR)
        {
            break;
        }

        if(bounce == 0)
        {
            alpha = 1.0f;
        }

        hit_p = ray_origin + ray_dir * ray.hit.t;
        const tinybvh::BLASInstance* inst = static_cast<const tinybvh::BLASInstance*>(scene->get_instance(ray.hit
                                                                                                              .inst));
        const ObjectMesh* obj = scene->get_object_mesh(ray.hit.inst);
        const Vec3F hit_n = normalize_vec3f(obj->get_normal(ray.hit.prim, ray.hit.u, ray.hit.v));
        world_n = normalize_vec3f(mat44_rowmajor_vec_mul_dir(inst->transform, hit_n));

        const LightBase* random_light = scene->get_random_light();
        const Vec2F light_sample = sampler().get_pmj02_sample(pixel_id * random_offset * 0x4738
                                                                  + random_light->get_id(),
                                                              sample + bounce);
        float light_pdf = 0.0f;
        const Vec3F nee_dir = random_light->sample_direction(hit_p, light_sample, world_n, light_pdf);

        tinybvh::Ray shadow_ray(hit_p + world_n * maths::constants::flt_large_epsilon, nee_dir);
        const bool occluded = scene->occlude(shadow_ray);

        if(!occluded && light_pdf > maths::constants::flt_large_epsilon)
        {
            const float cos_theta = maths::maxf(0.0f, dot_vec3f(world_n, nee_dir));
            const Vec3F radiance = random_light->sample_intensity();

            const float brdf_pdf = cos_theta / maths::constants::pi;

            const float mis_weight = light_pdf / (light_pdf + brdf_pdf);

            color += throughput * CLAY * radiance * cos_theta * mis_weight / light_pdf;
        }

        const Vec3F next_dir = sample_hemisphere(world_n, xoshiro_next_float(), xoshiro_next_float());

        const float cos_theta = maths::maxf(0.0f, dot_vec3f(world_n, next_dir));

        throughput *= CLAY;

        float light_dir_pdf = 0.0f;

        if(bounce > 2)
        {
            const float max_component = maths::maxf(maths::maxf(throughput.x, throughput.y),
                                                    throughput.z);
            const float q = maths::maxf(0.05f, 1.0f - max_component);

            const float rr_sample = xoshiro_next_float();

            if(rr_sample < q)
            {
                break;
            }

            throughput /= (1.0f - q);
        }

        ray_origin = hit_p + world_n * maths::constants::flt_large_epsilon;
        ray_dir = next_dir;
        ray = tinybvh::Ray(ray_origin, ray_dir);
    }

    return default_if_nan_vec4f(Vec4F(color.x, color.y, color.z, alpha), Vec4F(0.5f, 0.5f, 0.5f, alpha));
}

ROMANORENDER_NAMESPACE_END
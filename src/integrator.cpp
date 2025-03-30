#include "romanorender/integrator.h"
#include "romanorender/sampling.h"

#include "stdromano/logger.h"
#include "stdromano/random.h"

ROMANORENDER_NAMESPACE_BEGIN

#define CLAY Vec3F(0.4f, 0.4f, 0.4f)

Vec4F integrator_pathtrace(const Scene* scene, 
                           const uint16_t x,
                           const uint16_t y,
                           const uint32_t sample,
                           const uint16_t max_bounces) noexcept
{
    Vec3F color(0.0f);
    float alpha = 0.0f;

    const uint32_t pixel_id = scene->get_camera()->get_xres() * y + x;
    const uint32_t random_offset = pcg_random_uint32(pixel_id);
    const Vec2F pixel_sample = sampler().get_pmj02_sample(pixel_id * random_offset * 0x4738, sample);
    const Vec3F ray_origin = scene->get_camera()->get_ray_origin();
    const Vec3F ray_dir = scene->get_camera()->get_ray_direction(x, y);

    tinybvh::Ray ray(ray_origin, ray_dir);

    const int32_t cost = scene->intersect(ray);

    if(ray.hit.t < BVH_FAR)
    {
        alpha = 1.0f;

        const Vec3F hit_p = ray_origin + ray_dir * ray.hit.t;
        const tinybvh::BLASInstance* inst = static_cast<const tinybvh::BLASInstance*>(scene->get_instance(ray.hit.inst));
        const ObjectMesh* obj = scene->get_object_mesh(ray.hit.inst);
        const Vec3F hit_n = normalize_vec3f(obj->get_normal(ray.hit.prim, ray.hit.u, ray.hit.v));
        const Vec3F world_n = normalize_vec3f(mat44_rowmajor_vec_mul_dir(inst->transform, hit_n));

        const LightBase* random_light = scene->get_random_light();
        const Vec2F light_sample = sampler().get_pmj02_sample(pixel_id * random_offset * 0x4738 + random_light->get_id(), sample);
        float light_pdf = 0.0f;
        const Vec3F nee_dir = random_light->sample_direction(hit_p, light_sample, world_n, light_pdf);

        tinybvh::Ray shadow_ray(hit_p + world_n * maths::constants::flt_large_epsilon, nee_dir);

        const bool occlude = scene->occlude(shadow_ray);

        if(!occlude && light_pdf > maths::constants::flt_large_epsilon)
        {
            const float cos_theta = maths::maxf(0.0f, dot_vec3f(world_n, nee_dir));

            const Vec3F radiance = random_light->sample_intensity();

            color += CLAY * radiance * cos_theta / light_pdf;
        }
    }

    return default_if_nan_vec4f(Vec4F(color.x, color.y, color.z, alpha), Vec4F(0.5f, 0.5f, 0.5f, alpha));
}

ROMANORENDER_NAMESPACE_END
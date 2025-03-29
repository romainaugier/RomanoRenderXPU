#include "romanorender/integrator.h"
#include "romanorender/sampling.h"

#include "stdromano/logger.h"
#include "stdromano/random.h"

ROMANORENDER_NAMESPACE_BEGIN

Vec4F integrator_pathtrace(const Scene* scene, 
                           const uint16_t x,
                           const uint16_t y,
                           const uint32_t sample,
                           const uint16_t max_bounces) noexcept
{
    const uint32_t pixel_id = scene->get_camera()->get_xres() * y + x;
    const Vec2F pixel_sample = sampler().get_pmj02_sample(pixel_id, sample);
    const Vec3F ray_origin = scene->get_camera()->get_ray_origin();
    const Vec3F ray_dir = scene->get_camera()->get_ray_direction(x, y, pixel_sample.x, pixel_sample.y);

    tinybvh::Ray ray(ray_origin, ray_dir);

    const int32_t cost = scene->intersect(ray);

    if(ray.hit.t < BVH_FAR)
    {
        const tinybvh::BLASInstance* inst = static_cast<const tinybvh::BLASInstance*>(scene->get_instance(ray.hit.inst));
        const ObjectMesh* obj = scene->get_object_mesh(ray.hit.inst);
        const Vec3F hit_n = normalize_vec3f(obj->get_normal(ray.hit.prim, ray.hit.u, ray.hit.v));
        const Vec3F world_n = normalize_vec3f(mat44_rowmajor_vec_mul_dir(inst->transform, hit_n));

        const Vec3F color = (world_n + 0.5f) / 2.0f;

        return Vec4F(color.x, color.y, color.z, 1.0f);
    }
    else
    {
        return Vec4F(0.0f);
    }
}

ROMANORENDER_NAMESPACE_END
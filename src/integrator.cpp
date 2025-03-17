#include "romanorender/integrator.h"

#include "stdromano/logger.h"
#include "stdromano/random.h"

ROMANORENDER_NAMESPACE_BEGIN

Vec4F integrator_debug(Scene* scene, uint16_t x, uint16_t y, uint32_t sample) noexcept
{
    const Vec3F ray_origin = scene->get_camera()->get_ray_origin();
    const Vec3F ray_dir = scene->get_camera()->get_ray_direction(x, y);

    tinybvh::Ray ray(ray_origin, ray_dir);

    const int32_t cost = scene->intersect(ray);

    if(ray.hit.t < BVH_FAR)
    {
        const tinybvh::BLASInstance* inst = static_cast<const tinybvh::BLASInstance*>(scene->get_instance(ray.hit
                                                                                                              .inst));
        const ObjectMesh* obj = scene->get_object_mesh(ray.hit.inst);
        const Vec3F hit_n = normalize_vec3f(obj->get_primitive_normal(ray.hit.prim));
        const Vec3F world_n = normalize_vec3f(mat44f_mul_dir(inst->transform, hit_n));

        const Vec3F color = (world_n + 0.5f) / 2.0f;

        return Vec4F(color.x, color.y, color.z, 1.0f);
    }
    else
    {
        return Vec4F(0.0f);
    }
}

Vec4F integrator_mask(Scene* scene, uint16_t x, uint16_t y, uint32_t sample) noexcept
{
    const Vec3F ray_origin = scene->get_camera()->get_ray_origin();
    const Vec3F ray_dir = scene->get_camera()->get_ray_direction(x, y);

    tinybvh::Ray ray(ray_origin, ray_dir);

    const int32_t cost = scene->intersect(ray);

    if(ray.hit.t < BVH_FAR)
    {
        return Vec4F(stdromano::wang_hash_float(ray.hit.prim ^ 0x192FF),
                     stdromano::wang_hash_float(ray.hit.prim ^ 0x481AF),
                     stdromano::wang_hash_float(ray.hit.prim ^ 0x918EF),
                     1.0f);
    }
    else
    {
        return Vec4F(0.0f);
    }
}

Vec4F integrator_ambient_occlusion(Scene* scene, uint16_t x, uint16_t y, uint32_t sample) noexcept
{
    return Vec4F(0.0f);
}

ROMANORENDER_NAMESPACE_END
#include "romanorender/integrator.h"

#include "stdromano/random.h"
#include "stdromano/logger.h"

ROMANORENDER_NAMESPACE_BEGIN

Vec4F integrator_mask(Scene* scene, uint16_t x, uint16_t y, uint32_t sample) noexcept
{
    const Vec3F ray_origin = scene->get_camera().get_ray_origin();
    const Vec3F ray_dir = scene->get_camera().get_ray_direction(x, y);

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
#include "romanorender/integrator.h"

ROMANORENDER_NAMESPACE_BEGIN

Vec4F integrator_mask(Scene* scene, uint16_t x, uint16_t y, uint32_t sample) noexcept
{
    const Vec3F ray_origin = scene->get_camera().get_ray_origin();
    const Vec3F ray_dir = scene->get_camera().get_ray_direction(x, y);

    tinybvh::Ray ray(tinybvh::bvhvec3(ray_origin.x, ray_origin.y, ray_origin.z), tinybvh::bvhvec3(ray_dir.x, ray_dir.y, ray_dir.z));

    return scene->intersect(ray) > 0 ? Vec4F(1.0f) : Vec4F(0.0f);
}

Vec4F integrator_ambient_occlusion(Scene* scene, uint16_t x, uint16_t y, uint32_t sample) noexcept
{
    return Vec4F(0.0f);
}

ROMANORENDER_NAMESPACE_END
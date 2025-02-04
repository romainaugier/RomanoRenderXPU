#include "romanorender/scene.h"

ROMANORENDER_NAMESPACE_BEGIN

uint32_t Scene::attach_geometry(Geometry&& geometry) noexcept
{
    this->geometries.emplace_back(std::move(geometry));

    const uint32_t new_geom_id = this->geometries.size() - 1;

    this->geometries.at(new_geom_id)->set_id(new_geom_id);

    return new_geom_id;
}

bool Scene::build() noexcept
{
    return true;
}

bool Scene::intersect(RayHit& rayhit) const noexcept
{
    return true;
}

bool Scene::occlude(Ray& ray) const noexcept
{
    return true;
}

ROMANORENDER_NAMESPACE_END
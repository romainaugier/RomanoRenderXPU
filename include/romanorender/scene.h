#pragma once

#if !defined(__ROMANORENDER_SCENE)
#define __ROMANORENDER_SCENE

#include "romanorender/romanorender.h"
#include "romanorender/geometry.h"
#include "romanorender/ray.h"

ROMANORENDER_NAMESPACE_BEGIN

enum SceneBuildMode_ : uint32_t
{
    SceneBuildMode_Low = 0,
    SceneBuildMode_Mid = 1,
    SceneBuildMode_High = 2,
};

using Geometries = stdromano::Vector<Geometry>;

class ROMANORENDER_API Scene 
{
    Geometries geometries;

public:
    Scene() {}

    uint32_t attach_geometry(Geometry&& geometry) noexcept;

    bool build() noexcept;

    bool intersect(RayHit& rayhit) const noexcept;

    bool occlude(Ray& ray) const noexcept;
};

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_SCENE) */
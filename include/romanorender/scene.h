#pragma once

#if !defined(__ROMANORENDER_SCENE)
#define __ROMANORENDER_SCENE

#include "romanorender/object.h"
#include "romanorender/camera.h"

#include "stdromano/vector.h"

ROMANORENDER_NAMESPACE_BEGIN

class ROMANORENDER_API Scene 
{
    stdromano::Vector<const tinybvh::BVHBase*> _blasses;
    stdromano::Vector<tinybvh::BLASInstance> _instances;

    tinybvh::BVH _tlas;

    Camera _camera;

public:
    void set_camera(const Camera& camera) noexcept 
    {
        this->_camera = camera;
    }

    const Camera& get_camera() const noexcept
    {
        return this->_camera;
    }

    void set_camera_transform(const Mat44F& transform) noexcept
    {
        this->_camera.set_transform(transform);
    }

    void add_object(Object& obj) noexcept;

    void build_tlas() noexcept;

    int32_t intersect(tinybvh::Ray& ray) const noexcept
    {
        return this->_tlas.Intersect(ray);
    }

    bool occlude(const tinybvh::Ray& ray) const noexcept
    {
        return this->_tlas.IsOccluded(ray);
    }
};

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_SCENE) */
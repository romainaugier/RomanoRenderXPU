#pragma once

#if !defined(__ROMANORENDER_SCENE)
#define __ROMANORENDER_SCENE

#include "romanorender/camera.h"
#include "romanorender/object.h"

#include "stdromano/vector.h"

ROMANORENDER_NAMESPACE_BEGIN

class ROMANORENDER_API Scene
{
    stdromano::Vector<const tinybvh::BVHBase*> _blasses;
    stdromano::Vector<tinybvh::BLASInstance> _instances;
    stdromano::Vector<const Object*> _objects;
    stdromano::Vector<uint32_t> _objects_lookup;

    tinybvh::BVH _tlas;

    Camera _camera;

public:
    void set_camera(const Camera& camera) noexcept { this->_camera = camera; }

    const Camera& get_camera() const noexcept { return this->_camera; }

    void set_camera_transform(const Mat44F& transform) noexcept { this->_camera.set_transform(transform); }

    void add_object(Object& obj) noexcept;

    const Object* get_object(const uint32_t instance_id) noexcept;

    void add_instance(const Object* obj, const Mat44F& transform) noexcept;

    const tinybvh::BLASInstance* get_instance(const uint32_t instance_id) noexcept
    {
        return instance_id >= this->_instances.size() ? nullptr : this->_instances.at(instance_id);
    }

    void build_tlas() noexcept;

    int32_t intersect(tinybvh::Ray& ray) const noexcept { return this->_tlas.Intersect(ray); }

    bool occlude(const tinybvh::Ray& ray) const noexcept { return this->_tlas.IsOccluded(ray); }
};

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_SCENE) */
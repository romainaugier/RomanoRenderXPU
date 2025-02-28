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

    void add_object(const Object& obj) 
    {
        tinybvh::BLASInstance instance;
        instance.blasIdx = this->_blasses.size();
        std::memcpy(instance.transform, obj.get_transform().data(), 16 * sizeof(float));
        
        this->_blasses.push_back(&obj.get_blas());
        this->_instances.push_back(instance);
    }

    void build_tlas() noexcept
    {
        this->_tlas.Build(this->_instances.data(), 
                          this->_instances.size(),
                          const_cast<tinybvh::BVHBase**>(this->_blasses.data()),
                          this->_blasses.size());
    }

    int32_t intersect(tinybvh::Ray& ray) const noexcept
    {
        return this->_tlas.IntersectTLAS(ray);
    }

    bool occlude(const tinybvh::Ray& ray) const noexcept
    {
        return this->_tlas.IsOccludedTLAS(ray);
    }
};

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_SCENE) */
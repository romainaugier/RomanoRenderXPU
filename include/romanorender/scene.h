#pragma once

#if !defined(__ROMANORENDER_SCENE)
#define __ROMANORENDER_SCENE

#include "romanorender/scenegraph.h"

#include "stdromano/vector.h"

ROMANORENDER_NAMESPACE_BEGIN

class ROMANORENDER_API Scene
{
    stdromano::Vector<tinybvh::BVH8_CPU> _blasses;
    stdromano::Vector<tinybvh::BVHBase*> _blasses_ptr;
    stdromano::Vector<tinybvh::BLASInstance> _instances;
    stdromano::Vector<const ObjectMesh*> _meshes;
    stdromano::Vector<uint32_t> _objects_lookup;

    tinybvh::BVH _tlas;

    Camera* _camera;

public:
    void build_from_scenegraph(const SceneGraph& scenegraph) noexcept;

    void set_camera(Camera* camera) noexcept { this->_camera = camera; }

    Camera* get_camera() const noexcept { return this->_camera; }

    void add_object_mesh(ObjectMesh* obj) noexcept;

    const ObjectMesh* get_object_mesh(const uint32_t instance_id) noexcept;

    void add_instance(const ObjectMesh* obj, const Mat44F& transform) noexcept;

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
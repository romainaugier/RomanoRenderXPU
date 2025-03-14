#pragma once

#if !defined(__ROMANORENDER_SCENE)
#define __ROMANORENDER_SCENE

#include "romanorender/scenegraph.h"

#include "stdromano/vector.h"

#include "optix.h"
#include "optix_stubs.h"

ROMANORENDER_NAMESPACE_BEGIN

enum SceneBackend : uint32_t
{
    SceneBackend_CPU = 1,
    SceneBackend_GPU,
};

class AccelerationStructure 
{
public:
    virtual void add_object(const Vertices& vertices, const Indices& indices, const size_t num_triangles, const Mat44F& transform, const uint32_t id) noexcept = 0;

    virtual void add_instance(const size_t object_id, const Mat44F& transform) noexcept = 0;

    virtual void clear() noexcept = 0;

    virtual void build() noexcept = 0;
  
    virtual int32_t intersect(tinybvh::Ray& ray) const noexcept = 0;

    virtual bool occlude(const tinybvh::Ray& ray) const noexcept = 0;

    virtual ~AccelerationStructure() = default;
};

class CPUAccelerationStructure : public AccelerationStructure
{
    stdromano::Vector<tinybvh::BVH8_CPU> _blasses;
    stdromano::Vector<tinybvh::BVHBase*> _blasses_ptr;
    stdromano::Vector<tinybvh::BLASInstance> _instances;

    tinybvh::BVH _tlas;

public:
    virtual void add_object(const Vertices& vertices, 
                            const Indices& indices,
                            const size_t num_triangles,
                            const Mat44F& transform,
                            const uint32_t id) noexcept override;

    virtual void add_instance(const size_t id, const Mat44F& transform) noexcept override;

    virtual void clear() noexcept override;

    virtual void build() noexcept override;

    virtual int32_t intersect(tinybvh::Ray& ray) const noexcept override { return this->_tlas.Intersect(ray); }

    virtual bool occlude(const tinybvh::Ray& ray) const noexcept override { return this->_tlas.IsOccluded(ray); }

    virtual ~CPUAccelerationStructure() override;
};

class GPUAccelerationStructure : public AccelerationStructure
{
public:
    virtual void add_object(const Vertices& vertices, 
                            const Indices& indices,
                            const size_t num_triangles,
                            const Mat44F& transform,
                            const uint32_t id) noexcept override;

    virtual void add_instance(const size_t id, const Mat44F& transform) noexcept override;

    virtual void clear() noexcept override;

    virtual void build() noexcept override;

    virtual int32_t intersect(tinybvh::Ray& ray) const noexcept override { return 0; }

    virtual bool occlude(const tinybvh::Ray& ray) const noexcept override { return false; }

    virtual ~GPUAccelerationStructure() override;
};

using Instance = std::pair<uint32_t, Mat44F>;

class ROMANORENDER_API Scene
{
    AccelerationStructure* _as = nullptr;

    stdromano::Vector<const ObjectMesh*> _meshes;
    stdromano::Vector<uint32_t> _objects_lookup;

    stdromano::Vector<Instance> _instances;

    Camera* _camera = nullptr;

    SceneBackend _backend;

public:
    Scene(SceneBackend backend = SceneBackend_CPU);

    ~Scene();

    void set_backend(SceneBackend backend) noexcept;

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

    void build() noexcept { this->_as->build(); }

    int32_t intersect(tinybvh::Ray& ray) const noexcept { return this->_as->intersect(ray); }

    bool occlude(const tinybvh::Ray& ray) const noexcept { return this->_as->occlude(ray); }
};

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_SCENE) */
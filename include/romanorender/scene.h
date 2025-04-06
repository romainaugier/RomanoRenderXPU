#pragma once

#if !defined(__ROMANORENDER_SCENE)
#define __ROMANORENDER_SCENE

#include "romanorender/scenegraph.h"
#include "romanorender/ray.h"
#include "romanorender/random.h"

#include "stdromano/vector.h"

#include "optix.h"
#include "optix_stubs.h"

ROMANORENDER_NAMESPACE_BEGIN

#define INVALID_INSTANCE_ID 0xFFFFFFFFUL

enum SceneBackend : uint32_t
{
    SceneBackend_CPU = 1,
    SceneBackend_GPU,
};

class AccelerationStructure
{
    friend class Scene;

public:
    virtual uint32_t add_object(ObjectMesh* object,
                                const Mat44F& transform,
                                const uint8_t visibility_flags = VisibilityFlag_VisibleAllRays) noexcept = 0;

    virtual uint32_t add_instance(ObjectMesh* object, 
                                  const Mat44F& transform,
                                  const uint8_t visibility_flags = VisibilityFlag_VisibleAllRays) noexcept = 0;

    virtual void clear() noexcept = 0;

    virtual void clear_cache() noexcept = 0;

    virtual bool build() noexcept = 0;

    virtual int32_t intersect(tinybvh::Ray& ray) const noexcept = 0;

    virtual bool occlude(const tinybvh::Ray& ray) const noexcept = 0;

    virtual ~AccelerationStructure() = default;
};

class CPUAccelerationStructure : public AccelerationStructure
{
    friend class Scene;

    stdromano::HashMap<uint64_t, uint32_t> _uuid_to_blas_id;

    stdromano::Vector<tinybvh::BVH8_CPU> _blasses;
    stdromano::Vector<tinybvh::BVHBase*> _blasses_ptr;
    stdromano::Vector<tinybvh::BLASInstance> _instances;

    tinybvh::BVH _tlas;

public:
    virtual uint32_t add_object(ObjectMesh* object,
                                const Mat44F& transform,
                                const uint8_t visibility_flags = VisibilityFlag_VisibleAllRays) noexcept override;

    virtual uint32_t add_instance(ObjectMesh* object, 
                                  const Mat44F& transform,
                                  const uint8_t visibility_flags = VisibilityFlag_VisibleAllRays) noexcept override;

    virtual void clear() noexcept override;

    virtual void clear_cache() noexcept override;

    virtual bool build() noexcept override;

    virtual int32_t intersect(tinybvh::Ray& ray) const noexcept override
    {
        return this->_tlas.Intersect(ray);
    }

    virtual bool occlude(const tinybvh::Ray& ray) const noexcept override
    {
        return this->_tlas.IsOccluded(ray);
    }

    virtual ~CPUAccelerationStructure() override;
};

class GPUAccelerationStructure : public AccelerationStructure
{
    friend class Scene;

    struct BLASData
    {
        BLASData() = default;

        BLASData(const uint32_t id,
                 const Vertices& vertices, 
                 const Indices& indices,
                 const size_t num_triangles,
                 const Vec3F* normals);

        BLASData(const BLASData& other) = delete;
        BLASData& operator=(const BLASData& other) = delete;

        BLASData(BLASData&& other);
        BLASData& operator=(BLASData&& other);

        ~BLASData();

        uint32_t _id = 0;
        CUdeviceptr _vertices = 0;
        CUdeviceptr _indices = 0;
        CUdeviceptr _normals = 0;
        OptixTraversableHandle _handle = 0;
        CUdeviceptr _as = 0;
    };

    CudaVector<BLASData> _blasses;
    stdromano::HashMap<uint64_t, BLASData*> _blasses_map;
    CudaVector<OptixInstance> _instances;
    OptixTraversableHandle _tlas_handle = 0;
    CUdeviceptr _tlas_buffer = 0;
    CUdeviceptr _instances_buffer = 0;

public:
    virtual uint32_t add_object(ObjectMesh* object,
                                const Mat44F& transform,
                                const uint8_t visibility_flags = VisibilityFlag_VisibleAllRays) noexcept override;

    virtual uint32_t add_instance(ObjectMesh* object, 
                                  const Mat44F& transform,
                                  const uint8_t visibility_flags = VisibilityFlag_VisibleAllRays) noexcept override;

    virtual void clear() noexcept override;

    virtual void clear_cache() noexcept override;

    virtual bool build() noexcept override;

    virtual int32_t intersect(tinybvh::Ray& ray) const noexcept override { return 0; }

    virtual bool occlude(const tinybvh::Ray& ray) const noexcept override { return false; }

    virtual ~GPUAccelerationStructure() override;
};

class ROMANORENDER_API Scene
{
    AccelerationStructure* _as = nullptr;

    stdromano::HashMap<uint32_t, ObjectMesh*> _instances_to_meshes;

    stdromano::HashMap<uint32_t, ObjectMesh*> _light_meshes;

    stdromano::HashMap<uint32_t, LightBase*> _instances_to_lights;

    CudaVector<LightBase*> _lights;

    Camera* _camera = nullptr;

    SceneBackend _backend;

    uint32_t _id_counter = 0;

public:
    Scene(SceneBackend backend = SceneBackend_CPU);

    ~Scene();

    void clear() noexcept;

    void set_backend(SceneBackend backend) noexcept;

    bool build_from_scenegraph(const SceneGraph& scenegraph) noexcept;

    void set_camera(Camera* camera) noexcept { this->_camera = camera; }

    Camera* get_camera() const noexcept { return this->_camera; }

    void add_object_mesh(ObjectMesh* obj) noexcept;

    const ObjectMesh* get_object_mesh(const uint32_t instance_id) const noexcept;

    void add_instance(ObjectMesh* obj, 
                      const Mat44F& transform,
                      const uint8_t visibility_flags = VisibilityFlag_VisibleAllRays) noexcept;

    void add_light(ObjectLight* obj, const Mat44F& transform) noexcept;

    const void* get_instance(const uint32_t instance_id) const noexcept
    {
        if(CPUAccelerationStructure* as = dynamic_cast<CPUAccelerationStructure*>(this->_as))
        {
            return as->_instances.at(instance_id);
        }

        return nullptr;
    }

    void* get_as_handle() noexcept
    {
        if(GPUAccelerationStructure* as = dynamic_cast<GPUAccelerationStructure*>(this->_as))
        {
            return &as->_tlas_handle;
        }

        return nullptr;
    }

    void* get_blasses() noexcept
    {
        if(GPUAccelerationStructure* as = dynamic_cast<GPUAccelerationStructure*>(this->_as))
        {
            return as->_blasses.data();
        }

        return nullptr;
    }

    ROMANORENDER_FORCE_INLINE LightBase** get_lights() noexcept
    {
        return this->_lights.data();
    }

    ROMANORENDER_FORCE_INLINE const LightBase* get_light_for_mesh(const uint32_t instance_id) const
    {
        const auto& it = this->_instances_to_lights.find(instance_id);

        return it == this->_instances_to_lights.end() ? nullptr : it.value();
    }

    ROMANORENDER_FORCE_INLINE const LightBase* get_random_light() const noexcept
    {
        ROMANORENDER_ASSERT(this->_lights.size() > 0, "No lights in the scene");

        const uint32_t index = static_cast<int32_t>(maths::rintf(pcg_next_float() * static_cast<float>(this->_lights.size() - 1)));

        return this->_lights[index];
    }

    ROMANORENDER_FORCE_INLINE uint32_t get_num_lights() const noexcept
    {
        return this->_lights.size();
    }

    void build() noexcept { this->_as->build(); }

    int32_t intersect(tinybvh::Ray& ray) const noexcept { return this->_as->intersect(ray); }

    bool occlude(const tinybvh::Ray& ray) const noexcept { return this->_as->occlude(ray); }
};

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_SCENE) */
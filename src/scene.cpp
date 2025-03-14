#include "romanorender/scene.h"

#include "stdromano/logger.h"

#define STDROMANO_ENABLE_PROFILING
#include "stdromano/profiling.h"

ROMANORENDER_NAMESPACE_BEGIN

void CPUAccelerationStructure::add_object(const Vertices& vertices,
                                          const Indices& indices,
                                          const size_t num_triangles,
                                          const Mat44F& transform,
                                          const uint32_t id) noexcept
{
    this->_instances.push_back(id);
    std::memcpy(this->_instances.back().transform, transform.data(), 16 * sizeof(float));

    this->_blasses.emplace_back((tbvh::Vec4F*)vertices.data(), indices.data(), num_triangles);

    this->_blasses_ptr.push_back((tinybvh::BVHBase*)&this->_blasses.back());
}

void CPUAccelerationStructure::add_instance(const size_t id, const Mat44F& transform) noexcept
{
    this->_instances.emplace_back(id);
    std::memcpy(this->_instances.back().transform, transform.data(), 16 * sizeof(float));
}

void CPUAccelerationStructure::clear() noexcept
{
    this->_blasses.clear();
    this->_blasses_ptr.clear();
    this->_instances.clear();
}

void CPUAccelerationStructure::build() noexcept
{
    SCOPED_PROFILE_START(stdromano::ProfileUnit::MilliSeconds, tlas_build);

    this->_tlas.Build(
        this->_instances.data(), this->_instances.size(), this->_blasses_ptr.data(), this->_blasses_ptr.size());

    stdromano::log_debug("Built scene CPU TLAS. Bounds:\nmin({})\nmax({})",
                         this->_tlas.aabbMin,
                         this->_tlas.aabbMax);
}

CPUAccelerationStructure::~CPUAccelerationStructure()
{

}

void GPUAccelerationStructure::add_object(const Vertices& vertices,
                                          const Indices& indices,
                                          const size_t num_triangles,
                                          const Mat44F& transform,
                                          const uint32_t id) noexcept
{
}

void GPUAccelerationStructure::add_instance(const size_t id, const Mat44F& transform) noexcept
{
}

void GPUAccelerationStructure::clear() noexcept
{
}

void GPUAccelerationStructure::build() noexcept
{
    SCOPED_PROFILE_START(stdromano::ProfileUnit::MilliSeconds, tlas_build);

    stdromano::log_debug("Built scene GPU TLAS. Bounds:\nmin({})\nmax({})");
}

GPUAccelerationStructure::~GPUAccelerationStructure()
{

}

Scene::Scene(SceneBackend backend)
{
    this->_backend = backend;

    this->_as = backend == SceneBackend_CPU ? new CPUAccelerationStructure : new GPUAccelerationStructure;
}

Scene::~Scene() 
{
    if(this->_as != nullptr)
    {
        delete this->_as;
    }
}

void Scene::set_backend(SceneBackend backend) noexcept
{
    this->_backend = backend;

    if(this->_as != nullptr)
    {
        delete this->_as;
    }

    this->_as = backend == SceneBackend_CPU ? new CPUAccelerationStructure : new GPUAccelerationStructure;

    for(ObjectMesh* mesh : this->_meshes)
    {
        this->_as->add_object(mesh->get_vertices(), mesh->get_indices(), mesh->get_indices().size() / 3, mesh->get_transform(), mesh->get_id());
    }

    for(const Instance& instance : this->_instances)
    {
        this->_as->add_instance(instance.first, instance.second);
    }
}

void Scene::build_from_scenegraph(const SceneGraph& scenegraph) noexcept
{
    this->_meshes.clear();
    this->_objects_lookup.clear();

    bool camera_set = false;

    if(scenegraph.get_result() == nullptr)
    {
        stdromano::log_debug("Cannot build scene from errored scenegraph");
        return;
    }

    for(Object* obj : *scenegraph.get_result())
    {
        if(ObjectMesh* objmesh = dynamic_cast<ObjectMesh*>(obj))
        {
            this->add_object_mesh(objmesh);
        }
        else if(ObjectCamera* cam = dynamic_cast<ObjectCamera*>(obj))
        {
            if(camera_set)
            {
                continue;
            }

            this->set_camera(cam->get_camera());

            camera_set = true;
        }
        else if(ObjectInstance* inst = dynamic_cast<ObjectInstance*>(obj))
        {
            this->add_instance(inst->get_instanced(), inst->get_transform());
        }
    }

    this->build_tlas();
}

void Scene::add_object_mesh(ObjectMesh* obj) noexcept
{
    const uint32_t id = this->_blasses.size();

    obj->set_id(id);

    if(obj->get_name().empty())
    {
        obj->set_name(std::move(stdromano::String<>("object{}", id)));
    }

    this->_objects_lookup.emplace_back(id);

    this->_as->add_object(obj->get_vertices(), obj->get_indices(), obj->get_indices().size() / 3, obj->get_transform(), id);

    this->_meshes.push_back(obj);

    stdromano::log_debug("Added a new object to the scene: {} (id: {})", obj->get_name(), obj->get_id());
}

const ObjectMesh* Scene::get_object_mesh(const uint32_t instance_id) noexcept
{
    return instance_id >= this->_objects_lookup.size() ? nullptr : this->_meshes[this->_objects_lookup[instance_id]];
}

void Scene::add_instance(const ObjectMesh* obj, const Mat44F& transform) noexcept
{
    if(obj->get_id() == INVALID_OBJECT_ID)
    {
        return;
    }

    this->_objects_lookup.emplace_back(obj->get_id());

    this->_as->add_instance(obj->get_id(), transform);

    this->_instances.emplace_back(obj->get_id(), transform);

    stdromano::log_debug("Added a new instance to the scene: {} (id: {})", obj->get_name(), obj->get_id());
}

ROMANORENDER_NAMESPACE_END
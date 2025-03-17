#include "romanorender/scene.h"
#include "romanorender/optix_utils.h"

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

    this->_tlas.Build(this->_instances.data(),
                      this->_instances.size(),
                      this->_blasses_ptr.data(),
                      this->_blasses_ptr.size());

    stdromano::log_debug("Built scene CPU TLAS. Bounds:\nmin({})\nmax({})",
                         this->_tlas.aabbMin,
                         this->_tlas.aabbMax);
}

CPUAccelerationStructure::~CPUAccelerationStructure() {}

GPUAccelerationStructure::BLASData::BLASData(const Vertices& vertices,
                                             const Indices& indices,
                                             const size_t numTriangles)
{
    SCOPED_PROFILE_START(stdromano::ProfileUnit::MilliSeconds, gpu_blas_data_build);

    uint32_t geom_flags = OPTIX_GEOMETRY_FLAG_NONE;

    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&this->_vertices),
                               vertices.size() * sizeof(Vec4F),
                               optix_manager().get_stream()));

    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&this->_indices),
                               indices.size() * sizeof(uint32_t),
                               optix_manager().get_stream()));

    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(this->_vertices),
                               vertices.data(),
                               vertices.size() * sizeof(Vec4F),
                               cudaMemcpyHostToDevice,
                               optix_manager().get_stream()));

    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(this->_indices),
                               indices.data(),
                               indices.size() * sizeof(uint32_t),
                               cudaMemcpyHostToDevice,
                               optix_manager().get_stream()));

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;
    build_input.triangleArray.vertexBuffers = &this->_vertices;
    build_input.triangleArray.numVertices = static_cast<uint32_t>(vertices.size());
    build_input.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    build_input.triangleArray.vertexStrideInBytes = sizeof(Vec4F);
    build_input.triangleArray.indexBuffer = this->_indices;
    build_input.triangleArray.numIndexTriplets = static_cast<uint32_t>(numTriangles);
    build_input.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_UNSIGNED_INT3;
    build_input.triangleArray.indexStrideInBytes = sizeof(uint3);
    build_input.triangleArray.flags = &geom_flags;
    build_input.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION
                               | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blas_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_manager().get_context(),
                                             &accel_options,
                                             &build_input,
                                             1,
                                             &blas_sizes));

    CUdeviceptr tmp_buffer, output_buffer, compacted_size_buffer;
    CUDA_CHECK(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&tmp_buffer),
                                       blas_sizes.tempSizeInBytes,
                                       optix_manager().get_mem_pool(),
                                       optix_manager().get_stream()));
    CUDA_CHECK(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&output_buffer),
                                       blas_sizes.outputSizeInBytes,
                                       optix_manager().get_mem_pool(),
                                       optix_manager().get_stream()));
    CUDA_CHECK(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&compacted_size_buffer),
                                       sizeof(uint64_t),
                                       optix_manager().get_mem_pool(),
                                       optix_manager().get_stream()));

    OptixAccelEmitDesc emit_desc;
    emit_desc.type = OPTIX_PROPERTY_TYPE_COMPACTED_SIZE;
    emit_desc.result = compacted_size_buffer;

    OPTIX_CHECK(optixAccelBuild(optix_manager().get_context(),
                                optix_manager().get_stream(),
                                &accel_options,
                                &build_input,
                                1,
                                tmp_buffer,
                                blas_sizes.tempSizeInBytes,
                                output_buffer,
                                blas_sizes.outputSizeInBytes,
                                &this->_handle,
                                &emit_desc,
                                1));

    uint64_t compacted_size;
    CUDA_CHECK(cudaMemcpyAsync(&compacted_size,
                               reinterpret_cast<void*>(compacted_size_buffer),
                               sizeof(uint64_t),
                               cudaMemcpyDeviceToHost,
                               optix_manager().get_stream()));

    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&this->_as),
                               compacted_size,
                               optix_manager().get_stream()));

    OptixTraversableHandle compacted_handle;
    OPTIX_CHECK(optixAccelCompact(optix_manager().get_context(),
                                  optix_manager().get_stream(),
                                  this->_handle,
                                  this->_as,
                                  compacted_size,
                                  &compacted_handle));

    this->_handle = compacted_handle;

    CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(output_buffer), optix_manager().get_stream()));
    CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(tmp_buffer), optix_manager().get_stream()));
    CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(compacted_size_buffer),
                             optix_manager().get_stream()));
}

GPUAccelerationStructure::BLASData::~BLASData()
{
    if(this->_as != 0)
    {
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(this->_as), optix_manager().get_stream()));
    }

    if(this->_vertices != 0)
    {
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(this->_vertices),
                                 optix_manager().get_stream()));
    }

    if(this->_indices != 0)
    {
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(this->_indices),
                                 optix_manager().get_stream()));
    }
}

void GPUAccelerationStructure::add_object(const Vertices& vertices,
                                          const Indices& indices,
                                          const size_t num_triangles,
                                          const Mat44F& transform,
                                          const uint32_t id) noexcept
{
    this->_blasses.emplace_back(vertices, indices, num_triangles);
    this->_blasses_map.insert(std::make_pair(id, &this->_blasses.back()));

    this->add_instance(id, transform);
}

void GPUAccelerationStructure::add_instance(const size_t id, const Mat44F& transform) noexcept
{
    if(this->_blasses_map.find(id) == this->_blasses_map.end())
    {
        return;
    }

    OptixInstance instance = {};
    instance.traversableHandle = this->_blasses_map[id]->_handle;
    instance.visibilityMask = 0xFF;
    instance.sbtOffset = 0;

    for(int i = 0; i < 3; ++i)
    {
        for(int j = 0; j < 4; ++j)
        {
            instance.transform[j + i * 4] = transform[j + i * 4];
        }
    }

    this->_instances.push_back(instance);
}

void GPUAccelerationStructure::clear() noexcept
{
    this->_blasses.clear();
    this->_blasses_map.clear();

    if(this->_tlas_buffer != 0)
    {
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(this->_tlas_buffer),
                                 optix_manager().get_stream()));
    }

    if(this->_instances_buffer != 0)
    {
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(this->_instances_buffer),
                                 optix_manager().get_stream()));
    }

    this->_instances.clear();

    this->_tlas_handle = 0;
}

void GPUAccelerationStructure::build() noexcept
{
    SCOPED_PROFILE_START(stdromano::ProfileUnit::MilliSeconds, tlas_build);

    if(this->_instances.empty())
    {
        return;
    }

    if(this->_instances_buffer != 0)
    {
        CUDA_CHECK(cudaFreeAsync((void*)this->_instances_buffer, optix_manager().get_stream()));
    }

    if(this->_tlas_buffer != 0)
    {
        CUDA_CHECK(cudaFreeAsync((void*)this->_tlas_buffer, optix_manager().get_stream()));
    }

    CUDA_CHECK(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&this->_instances_buffer),
                                       this->_instances.size() * sizeof(OptixInstance),
                                       optix_manager().get_mem_pool(),
                                       optix_manager().get_stream()));
    CUDA_CHECK(cudaMemcpyAsync((void*)this->_instances_buffer,
                               this->_instances.data(),
                               this->_instances.size() * sizeof(OptixInstance),
                               cudaMemcpyHostToDevice,
                               optix_manager().get_stream()));

    OptixBuildInput build_input = {};
    build_input.type = OPTIX_BUILD_INPUT_TYPE_INSTANCES;
    build_input.instanceArray.instances = this->_instances_buffer;
    build_input.instanceArray.numInstances = static_cast<uint32_t>(this->_instances.size());

    OptixAccelBuildOptions accel_options = {};
    accel_options.buildFlags = OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes tlas_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_manager().get_context(),
                                             &accel_options,
                                             &build_input,
                                             1,
                                             &tlas_sizes));

    CUdeviceptr temp_buffer;
    CUDA_CHECK(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&temp_buffer),
                                       tlas_sizes.tempSizeInBytes,
                                       optix_manager().get_mem_pool(),
                                       optix_manager().get_stream()));
    CUDA_CHECK(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&this->_tlas_buffer),
                                       tlas_sizes.outputSizeInBytes,
                                       optix_manager().get_mem_pool(),
                                       optix_manager().get_stream()));

    OptixTraversableHandle new_tlas;
    OPTIX_CHECK(optixAccelBuild(optix_manager().get_context(),
                                optix_manager().get_stream(),
                                &accel_options,
                                &build_input,
                                1,
                                temp_buffer,
                                tlas_sizes.tempSizeInBytes,
                                this->_tlas_buffer,
                                tlas_sizes.outputSizeInBytes,
                                &new_tlas,
                                nullptr,
                                0));

    this->_tlas_handle = new_tlas;

    CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(temp_buffer), optix_manager().get_stream()));

    stdromano::Vector<GeometryData> geom_data;

    for(const auto& it : this->_blasses_map)
    {
        geom_data.emplace_back(it.first, it.second->_vertices, it.second->_indices);
    }

    optix_manager().create_sbt(geom_data);

    stdromano::log_debug("Built scene GPU TLAS");
}

GPUAccelerationStructure::~GPUAccelerationStructure() { this->clear(); }

Scene::Scene(SceneBackend backend)
{
    this->_backend = backend;

    if(backend == SceneBackend_CPU)
    {
        this->_as = new CPUAccelerationStructure;
    }
    else
    {
        this->_as = new GPUAccelerationStructure;
    }
}

Scene::~Scene()
{
    if(this->_as != nullptr)
    {
        delete this->_as;
    }
}

void Scene::clear() noexcept
{
    this->_meshes.clear();
    this->_objects_lookup.clear();
    this->_id_counter = 0;
}

void Scene::set_backend(SceneBackend backend) noexcept
{
    if(backend == this->_backend)
    {
        return;
    }

    this->_backend = backend;

    if(this->_as != nullptr)
    {
        delete this->_as;
    }

    if(backend == SceneBackend_CPU)
    {
        this->_as = new CPUAccelerationStructure;
    }
    else
    {
        this->_as = new GPUAccelerationStructure;
    }

    for(const ObjectMesh* mesh : this->_meshes)
    {
        ObjectMesh* _mesh = const_cast<ObjectMesh*>(mesh);
        this->_as->add_object(_mesh->get_vertices(),
                              _mesh->get_indices(),
                              _mesh->get_indices().size() / 3,
                              _mesh->get_transform(),
                              _mesh->get_id());
    }

    for(const Instance& instance : this->_instances)
    {
        this->_as->add_instance(instance.first, instance.second);
    }
}

void Scene::build_from_scenegraph(const SceneGraph& scenegraph) noexcept
{
    this->clear();

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

    this->_as->build();
}

void Scene::add_object_mesh(ObjectMesh* obj) noexcept
{
    const uint32_t id = this->_id_counter++;

    obj->set_id(id);

    if(obj->get_name().empty())
    {
        obj->set_name(std::move(stdromano::String<>("object{}", id)));
    }

    this->_objects_lookup.emplace_back(id);

    this->_as->add_object(obj->get_vertices(),
                          obj->get_indices(),
                          obj->get_indices().size() / 3,
                          obj->get_transform(),
                          id);

    this->_meshes.push_back(obj);

    stdromano::log_debug("Added a new object to the scene: {} (id: {})",
                         obj->get_name(),
                         obj->get_id());
}

const ObjectMesh* Scene::get_object_mesh(const uint32_t instance_id) noexcept
{
    return instance_id >= this->_objects_lookup.size()
               ? nullptr
               : this->_meshes[this->_objects_lookup[instance_id]];
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

    stdromano::log_debug("Added a new instance to the scene: {} (id: {})",
                         obj->get_name(),
                         obj->get_id());
}

ROMANORENDER_NAMESPACE_END
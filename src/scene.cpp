#include "romanorender/scene.h"
#include "romanorender/optix_utils.h"

#include "stdromano/logger.hpp"

#define STDROMANO_ENABLE_PROFILING
#include "stdromano/profiling.hpp"

ROMANORENDER_NAMESPACE_BEGIN

uint32_t CPUAccelerationStructure::add_object(ObjectMesh* object,
                                              const Mat44F& transform,
                                              const uint8_t visibility_flags) noexcept
{
    const uint64_t uuid = object->get_uuid();
    const auto it = this->_uuid_to_blas_id.find(uuid);

    if(it != this->_uuid_to_blas_id.end())
    {
        return this->add_instance(object, transform, visibility_flags);
    }
    else
    {
        const uint32_t blas_id = this->_blasses.size();

        this->_blasses.emplace_back(std::move(tinybvh::BVH8_CPU()));

        this->_blasses.back().BuildHQ((tbvh::Vec4F*)object->get_vertices().data(),
                                      object->get_indices().data(),
                                      object->get_indices().size() / 3);

        if(isinf_vec3f(this->_blasses.back().aabbMin) || isinf_vec3f(this->_blasses.back().aabbMax) ||
           isnan_vec3f(this->_blasses.back().aabbMin) || isnan_vec3f(this->_blasses.back().aabbMax))
        {
            stdromano::log_error("Infty/Nan found when building CPU BLAS for object: {}", object->get_path());

            this->_blasses.pop_back();

            return INVALID_INSTANCE_ID;
        }

        this->_blasses_ptr.push_back((tinybvh::BVHBase*)&this->_blasses.back());

        this->_uuid_to_blas_id.insert(std::make_pair(uuid, blas_id));

        stdromano::log_debug("Added a new object to the BLAS cache. BLAS ID: {}", blas_id);

        return this->add_instance(object, transform, visibility_flags);
    }
}

uint32_t CPUAccelerationStructure::add_instance(ObjectMesh* object, 
                                                const Mat44F& transform,
                                                const uint8_t visibility_flags) noexcept
{
    const uint64_t uuid = object->get_uuid();
    const auto it = this->_uuid_to_blas_id.find(uuid);

    if(transform.has_zero_scale())
    {
        stdromano::log_error("Discarding instance of object {}. One the scale axis is zeroed", object->get_path());

        return INVALID_INSTANCE_ID;
    }

    if(it == this->_uuid_to_blas_id.end())
    {
        return this->add_object(object, transform, visibility_flags);
    }

    this->_instances.emplace_back(it.value());
    this->_instances.back().mask = (uint32_t)visibility_flags;

    const Mat44F transform_transposed = transform.transpose();

    std::memcpy(std::addressof(this->_instances.back().transform[0]), transform_transposed.data(), 16 * sizeof(float));

    return this->_instances.size() - 1;
}

void CPUAccelerationStructure::clear() noexcept
{
    this->_instances.clear();
}

void CPUAccelerationStructure::clear_cache() noexcept
{
    this->_uuid_to_blas_id.clear();
    this->_blasses.clear();
    this->_blasses_ptr.clear();

    this->clear();
}

bool CPUAccelerationStructure::build() noexcept
{
    SCOPED_PROFILE_START(stdromano::ProfileUnit::MilliSeconds, cpu_tlas_build);

    if(this->_instances.empty())
    {
        stdromano::log_error("No geometry found to build the scene");
        return false;
    }

    this->_tlas.Build(this->_instances.data(),
                      this->_instances.size(),
                      this->_blasses_ptr.data(),
                      this->_blasses_ptr.size());

    if(isinf_vec3f(this->_tlas.aabbMin) || isinf_vec3f(this->_tlas.aabbMax) ||
       isnan_vec3f(this->_tlas.aabbMin) || isnan_vec3f(this->_tlas.aabbMax))
    {
        stdromano::log_error("Infty/Nan found when building CPU TLAS");
        return false;
    }

    stdromano::log_debug("Built scene CPU TLAS.\nBounds: min({}) max({})",
                         this->_tlas.aabbMin,
                         this->_tlas.aabbMax);

    return true;
}

CPUAccelerationStructure::~CPUAccelerationStructure() { this->clear_cache(); }

GPUAccelerationStructure::BLASData::BLASData(const uint32_t id,
                                             const Vertices& vertices,
                                             const Indices& indices,
                                             const size_t numTriangles,
                                             const Vec3F* normals)
{
    SCOPED_PROFILE_START(stdromano::ProfileUnit::MilliSeconds, gpu_blas_data_build);

    this->_id = id;

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
    accel_options.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accel_options.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes blas_sizes;
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_manager().get_context(), &accel_options, &build_input, 1, &blas_sizes));

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

    CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&this->_as), compacted_size, optix_manager().get_stream()));

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
    CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(compacted_size_buffer), optix_manager().get_stream()));

    if(normals != nullptr)
    {
        CUDA_CHECK(cudaMallocAsync((void**)&this->_normals,
                                   vertices.size() * sizeof(Vec3F),
                                   optix_manager().get_stream()));
        CUDA_CHECK(cudaMemcpyAsync((void*)this->_normals,
                                   normals,
                                   vertices.size() * sizeof(Vec3F),
                                   cudaMemcpyHostToDevice,
                                   optix_manager().get_stream()));
    }
}

GPUAccelerationStructure::BLASData::BLASData(BLASData&& other) : 
    _id(other._id),
    _vertices(other._vertices),
    _indices(other._indices),
    _normals(other._normals),
    _handle(other._handle),
    _as(other._as)
{
    other._id = INVALID_OBJECT_ID;
    other._vertices = 0;
    other._indices = 0;
    other._normals = 0;
    other._handle = 0;
    other._as = 0;
}

GPUAccelerationStructure::BLASData& GPUAccelerationStructure::BLASData::operator=(BLASData&& other)
{
    this->_id = other._id;
    this->_vertices = other._vertices;
    this->_indices = other._indices;
    this->_normals = other._normals;
    this->_handle = other._handle;
    this->_as = other._as;

    other._id = INVALID_OBJECT_ID;
    other._vertices = 0;
    other._indices = 0;
    other._normals = 0;
    other._handle = 0;
    other._as = 0;

    return *this;
}

GPUAccelerationStructure::BLASData::~BLASData()
{
    if(this->_as != 0)
    {
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(this->_as), optix_manager().get_stream()));
    }

    if(this->_vertices != 0)
    {
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(this->_vertices), optix_manager().get_stream()));
    }

    if(this->_indices != 0)
    {
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(this->_indices), optix_manager().get_stream()));
    }

    if(this->_normals != 0)
    {
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(this->_normals), optix_manager().get_stream()));
    }
}

uint32_t GPUAccelerationStructure::add_object(ObjectMesh* object,
                                              const Mat44F& transform,
                                              const uint8_t visibility_flags) noexcept
{
    const uint64_t uuid = object->get_uuid();

    if(this->_blasses_map.find(uuid) != this->_blasses_map.end())
    {
        return this->add_instance(object, transform, visibility_flags);
    }
    else
    {
        const AttributeBuffer* N_buffer = object->get_vertex_attribute_buffer("N");

        const uint32_t blas_id = this->_blasses.size();

        this->_blasses.emplace_back(blas_id,
                                    object->get_vertices(),
                                    object->get_indices(),
                                    object->get_indices().size() / 3,
                                    N_buffer != nullptr ? N_buffer->get_data_ptr<Vec3F>() : nullptr);

        this->_blasses_map.insert(std::make_pair(uuid, &this->_blasses.back()));

        return this->add_instance(object, transform, visibility_flags);
    }
}

uint32_t GPUAccelerationStructure::add_instance(ObjectMesh* object, 
                                                const Mat44F& transform,
                                                const uint8_t visibility_flags) noexcept
{
    const uint64_t uuid = object->get_uuid();

    if(this->_blasses_map.find(uuid) == this->_blasses_map.end())
    {
        return this->add_object(object, transform, visibility_flags);
    }

    const uint32_t blas_id = this->_blasses_map[uuid]->_id;

    OptixInstance instance = {};
    instance.traversableHandle = this->_blasses_map[uuid]->_handle;
    instance.visibilityMask = visibility_flags;
    instance.sbtOffset = blas_id;
    instance.instanceId = this->_instances.size();

    const Mat44F transform_transposed = transform.transpose();

    std::memcpy(instance.transform, transform_transposed.data(), 12 * sizeof(float));

    this->_instances.push_back(instance);

    return this->_instances.size() - 1;
}

void GPUAccelerationStructure::clear() noexcept
{
    cudaStreamSynchronize(optix_manager().get_stream());

    if(this->_tlas_buffer != 0)
    {
        CUDA_CHECK(cudaFreeAsync((void*)this->_tlas_buffer, optix_manager().get_stream()));
    }

    this->_tlas_buffer = 0;

    if(this->_instances_buffer != 0)
    {
        CUDA_CHECK(cudaFreeAsync((void*)this->_instances_buffer, optix_manager().get_stream()));
    }

    this->_instances_buffer = 0;

    this->_instances.clear();

    this->_tlas_handle = 0;
}

void GPUAccelerationStructure::clear_cache() noexcept
{
    cudaStreamSynchronize(optix_manager().get_stream());

    this->_blasses.clear();
    this->_blasses_map.clear();

    this->clear();
}

int cmp_geom_data(const void* left, const void* right)
{
    const GeometryData* l = reinterpret_cast<const GeometryData*>(left);
    const GeometryData* r = reinterpret_cast<const GeometryData*>(right);

    return l->id - r->id;
}

bool GPUAccelerationStructure::build() noexcept
{
    SCOPED_PROFILE_START(stdromano::ProfileUnit::MilliSeconds, gpu_tlas_build);

    if(this->_instances.empty())
    {
        return false;
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
    OPTIX_CHECK(optixAccelComputeMemoryUsage(optix_manager().get_context(), &accel_options, &build_input, 1, &tlas_sizes));

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
        geom_data.emplace_back(it.second->_id, it.second->_vertices, it.second->_indices, it.second->_normals);
    }

    geom_data.sort(cmp_geom_data);

    optix_manager().create_sbt(geom_data);

    stdromano::log_debug("Built scene GPU TLAS");

    return true;
}

GPUAccelerationStructure::~GPUAccelerationStructure() { this->clear_cache(); }

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
    this->clear();

    for(auto& it : this->_light_meshes)
    {
        delete it.second;
    }

    if(this->_as != nullptr)
    {
        delete this->_as;
    }
}

void Scene::clear() noexcept
{
    this->_instances_to_meshes.clear();
    this->_instances_to_lights.clear();
    this->_id_counter = 0;
    this->_lights.clear();

    if(this->_as != nullptr)
    {
        this->_as->clear();
    }
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
}

bool Scene::build_from_scenegraph(const SceneGraph& scenegraph) noexcept
{
    this->clear();

    bool camera_set = false;

    if(scenegraph.get_result() == nullptr)
    {
        stdromano::log_debug("Cannot build scene from errored scenegraph");
        return false;
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
        else if(ObjectLight* light = dynamic_cast<ObjectLight*>(obj))
        {
            this->add_light(light, light->get_transform());
        }
    }

    for(Object* obj : *scenegraph.get_result())
    {
        if(ObjectInstance* inst = dynamic_cast<ObjectInstance*>(obj))
        {
            this->add_instance(inst->get_instanced(), inst->get_transform(), inst->get_visibility_flags());
        }
    }

    return this->_as->build();
}

void Scene::add_object_mesh(ObjectMesh* obj) noexcept
{
    const uint32_t instance_id = this->_as->add_object(obj, obj->get_transform(), obj->get_visibility_flags());

    this->_instances_to_meshes.insert(std::make_pair(instance_id, obj));

    stdromano::log_debug("Added a new object to the scene: {}", obj->get_name());
}

const ObjectMesh* Scene::get_object_mesh(const uint32_t instance_id) const noexcept
{
    const auto it = this->_instances_to_meshes.find(instance_id);

    return it == this->_instances_to_meshes.end() ? nullptr : it.value();
}

void Scene::add_instance(ObjectMesh* obj, 
                         const Mat44F& transform,
                         const uint8_t visibility_flags) noexcept
{
    const uint64_t uuid = obj->get_uuid();

    const uint32_t instance_id = this->_as->add_instance(obj, transform, visibility_flags);

    this->_instances_to_meshes.insert(std::make_pair(instance_id, obj));

    stdromano::log_debug("Added a new instance to the scene: {}", obj->get_path());
}

void Scene::add_light(ObjectLight* obj, const Mat44F& transform) noexcept
{
    obj->get_light()->set_transform(transform);
    this->_lights.push_back(obj->get_light());

    if(!obj->get_light()->get_is_visible())
    {
        return;
    }

    auto it = this->_light_meshes.find(obj->get_uuid_32());

    if(it != this->_light_meshes.end())
    {
        Mat44F light_transform;

        switch(obj->get_light()->get_type())
        {
            case LightType_Square:
            {
                LightSquare* light_ptr = reinterpret_cast<LightSquare*>(obj->get_light());
                light_transform = Mat44F::from_scale(Vec3F(light_ptr->get_size_x(), light_ptr->get_size_y(), 1.0f)) * transform;
                break;
            }
            default:
            {
                light_transform = transform;
                break;
            }
        }

        const uint32_t instance_id = this->_as->add_instance(it.value(),
                                                             light_transform,
                                                             VisibilityFlag_VisiblePrimaryRays);

        this->_instances_to_lights[instance_id] = obj->get_light();
        this->_instances_to_meshes[instance_id] = it.value();
    }
    else
    {
        switch(obj->get_light()->get_type())
        {
            case LightType_Square:
            {
                LightSquare* light_ptr = reinterpret_cast<LightSquare*>(obj->get_light());
                ObjectMesh* plane = ObjectMesh::plane(Vec3F(0.0f), Vec3F(1.0f,
                                                                         1.0f,
                                                                         0.0f));

                plane->_uuid = obj->get_uuid_32();
                plane->set_is_light_mesh(true);

                this->_light_meshes.insert(std::make_pair(obj->get_uuid_32(), plane));

                const Mat44F light_transform = Mat44F::from_scale(Vec3F(light_ptr->get_size_x(),
                                                                        light_ptr->get_size_y(),
                                                                        1.0f)) * transform;

                const uint32_t instance_id = this->_as->add_object(plane,
                                                                   light_transform,
                                                                   VisibilityFlag_VisiblePrimaryRays);

                this->_instances_to_lights[instance_id] = obj->get_light();
                this->_instances_to_meshes[instance_id] = plane;

                stdromano::log_debug("Added a new mesh in meshlight cache for light: {}", obj->get_path());

                break;
            }
        }
    }

    stdromano::log_debug("Added a new light to the scene: {}", obj->get_path());
    stdromano::log_debug("Light transform: {}");
}

ROMANORENDER_NAMESPACE_END
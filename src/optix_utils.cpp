#include "romanorender/optix_utils.h"
#include "romanorender/cuda_vector.h"

#include "stdromano/filesystem.hpp"

#define STDROMANO_ENABLE_PROFILING
#include "stdromano/profiling.hpp"

#include <optix_function_table_definition.h>

ROMANORENDER_NAMESPACE_BEGIN

void optix_log_callback(unsigned int level, const char* tag, const char* message, void* cbdata)
{
    switch(level)
    {
    case 1:
        stdromano::log_critical("[OPTIX]: {}", message);
        break;
    case 2:
        stdromano::log_error("[OPTIX]: {}", message);
        break;
    case 3:
        stdromano::log_warn("[OPTIX]: {}", message);
        break;
    case 4:
        stdromano::log_info("[OPTIX]: {}", message);
        break;
    }
}

void OptixManager::setup_cuda() noexcept
{
    SCOPED_PROFILE_START(stdromano::ProfileUnit::MilliSeconds, cuda_initialization);

    cudaFree(0);

    CUDA_CHECK(cudaGetDevice(&this->_cuda_device));
    CUDA_CHECK(cudaStreamCreate(&this->_cuda_stream));

    cudaMemPoolProps props = {};
    props.allocType = cudaMemAllocationTypePinned;
    props.location.id = this->_cuda_device;
    props.location.type = cudaMemLocationTypeDevice;

    cuuint64_t threshold = 0;

    CUDA_CHECK(cudaMemPoolCreate(&this->_cuda_mem_pool, &props));
    CUDA_CHECK(cudaMemPoolSetAttribute(this->_cuda_mem_pool, cudaMemPoolAttrReleaseThreshold, &threshold));
}

void OptixManager::setup_optix() noexcept
{
    SCOPED_PROFILE_START(stdromano::ProfileUnit::MilliSeconds, optix_initialization);

    CUcontext current_ctx = nullptr;

    OptixDeviceContextOptions options = {};
    options.validationMode = OPTIX_DEVICE_CONTEXT_VALIDATION_MODE_ALL;
    options.logCallbackLevel = 3;
    options.logCallbackFunction = optix_log_callback;
    OPTIX_CHECK(optixInit());
    OPTIX_CHECK(optixDeviceContextCreate(current_ctx, &options, &this->_context));
}

void OptixManager::create_pipeline() noexcept
{
    SCOPED_PROFILE_START(stdromano::ProfileUnit::MilliSeconds, optix_pipeline_initialization);

    OptixModuleCompileOptions module_compile_options = {};
    module_compile_options.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    module_compile_options.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    module_compile_options.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    OptixPipelineCompileOptions pipeline_compiles_options = {};
    pipeline_compiles_options.usesMotionBlur = false;
    pipeline_compiles_options.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipeline_compiles_options.numPayloadValues = 2;
    pipeline_compiles_options.numAttributeValues = 2;
    pipeline_compiles_options.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipeline_compiles_options.pipelineLaunchParamsVariableName = "params";

    stdromano::StringD
        shaders_ptx_path = stdromano::fs_expand_from_executable_dir("shaders/shaders.ptx");
    stdromano::StringD shaders_ptx = std::move(stdromano::load_file_content(shaders_ptx_path.data()));

    if(shaders_ptx.empty())
    {
        stdromano::log_error("Error while loading shaders ptx code, cannot find file: {}", shaders_ptx_path);

        std::exit(1);
    }

    char log[4096];
    size_t log_size = sizeof(log);

    OPTIX_CHECK(optixModuleCreate(this->_context,
                                  &module_compile_options,
                                  &pipeline_compiles_options,
                                  shaders_ptx.data(),
                                  shaders_ptx.size(),
                                  log,
                                  &log_size,
                                  &this->_module));

    // Raygen

    {
        OptixProgramGroupOptions options = {};
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = this->_module;
        desc.raygen.entryFunctionName = "__raygen__rg";
        OPTIX_CHECK(optixProgramGroupCreate(this->_context, &desc, 1, &options, log, &log_size, &this->_raygen_group));
    }

    // Miss

    {
        OptixProgramGroupOptions options = {};
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module = this->_module;
        desc.miss.entryFunctionName = "__miss__ms";
        OPTIX_CHECK(optixProgramGroupCreate(this->_context, &desc, 1, &options, log, &log_size, &this->_miss_group));
    }

    // Hit

    {
        OptixProgramGroupOptions options = {};
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH = this->_module;
        desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        OPTIX_CHECK(optixProgramGroupCreate(this->_context, &desc, 1, &options, log, &log_size, &this->_hit_group));
    }


    OptixProgramGroup program_groups[] = {this->_raygen_group, this->_miss_group, this->_hit_group};

    OptixPipelineLinkOptions pipeline_link_options = {};
    pipeline_link_options.maxTraceDepth = 2;

    OPTIX_CHECK(optixPipelineCreate(this->_context,
                                    &pipeline_compiles_options,
                                    &pipeline_link_options,
                                    program_groups,
                                    sizeof(program_groups) / sizeof(program_groups[0]),
                                    log,
                                    &log_size,
                                    &this->_pipeline));
}

void OptixManager::create_sbt(const stdromano::Vector<GeometryData>& geometries) noexcept
{
    SCOPED_PROFILE_START(stdromano::ProfileUnit::MilliSeconds, optix_sbt_initialization);

    if(this->_sbt.raygenRecord != 0)
    {
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(this->_sbt.raygenRecord), this->_cuda_stream));
    }

    if(this->_sbt.missRecordBase != 0)
    {
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(this->_sbt.missRecordBase), this->_cuda_stream));
    }

    if(this->_sbt.hitgroupRecordBase != 0)
    {
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(this->_sbt.hitgroupRecordBase), this->_cuda_stream));
    }

    // Raygen
    RaygenRecord rg_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(this->_raygen_group, &rg_record));

    CUdeviceptr d_raygen_record;
    CUDA_CHECK(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&d_raygen_record),
                                       sizeof(RaygenRecord),
                                       this->_cuda_mem_pool,
                                       this->_cuda_stream));
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_raygen_record),
                               &rg_record,
                               sizeof(RaygenRecord),
                               cudaMemcpyHostToDevice,
                               this->_cuda_stream));

    // Miss
    MissRecord ms_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(this->_miss_group, &ms_record));

    CUdeviceptr d_miss_record;
    CUDA_CHECK(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&d_miss_record),
                                       sizeof(MissRecord),
                                       this->_cuda_mem_pool,
                                       this->_cuda_stream));
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_miss_record),
                               &ms_record,
                               sizeof(MissRecord),
                               cudaMemcpyHostToDevice,
                               this->_cuda_stream));

    // Hit
    CudaVector<HitGroupRecord> hg_records;

    for(const GeometryData& geom_data : geometries)
    {
        HitGroupRecord rec;
        OPTIX_CHECK(optixSbtRecordPackHeader(this->_hit_group, &rec));
        rec.data = geom_data;

        hg_records.push_back(rec);
    }

    CUdeviceptr d_hitgroup_record;
    CUDA_CHECK(cudaMallocFromPoolAsync(reinterpret_cast<void**>(&d_hitgroup_record),
                                       hg_records.size() * sizeof(HitGroupRecord),
                                       this->_cuda_mem_pool,
                                       this->_cuda_stream));
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(d_hitgroup_record),
                               hg_records.data(),
                               hg_records.size() * sizeof(HitGroupRecord),
                               cudaMemcpyHostToDevice,
                               this->_cuda_stream));

    this->_sbt.raygenRecord = d_raygen_record;
    this->_sbt.missRecordBase = d_miss_record;
    this->_sbt.missRecordStrideInBytes = sizeof(MissRecord);
    this->_sbt.missRecordCount = 1;
    this->_sbt.hitgroupRecordBase = d_hitgroup_record;
    this->_sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    this->_sbt.hitgroupRecordCount = hg_records.size();
}

void OptixManager::update_params(const OptixParams* params) noexcept
{
    CUDA_CHECK(cudaMemcpyAsync(reinterpret_cast<void*>(this->_params),
                               params,
                               sizeof(OptixParams),
                               cudaMemcpyHostToDevice,
                               this->_cuda_stream));
}

void OptixManager::launch(const size_t width, const size_t height, const size_t num_samples) noexcept
{
    OPTIX_CHECK(optixLaunch(this->_pipeline, this->_cuda_stream, this->_params, sizeof(OptixParams), &this->_sbt, width, height, num_samples));

    CUDA_CHECK(cudaStreamSynchronize(this->_cuda_stream));
}

void OptixManager::initialize() noexcept
{
    if(!this->_initialized)
    {
        this->setup_cuda();
        this->setup_optix();
        this->create_pipeline();
        this->_initialized = true;

        CUDA_CHECK(cudaMallocAsync(reinterpret_cast<void**>(&this->_params), sizeof(OptixParams), this->_cuda_stream));
    }
}

void OptixManager::cleanup() noexcept
{
#if 1
    CUDA_SYNC_CHECK();
    CUDA_CHECK(cudaStreamSynchronize(this->_cuda_stream));

    if(this->_params != 0)
    {
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(this->_params), this->_cuda_stream));
        this->_params = 0;
    }

    if(this->_sbt.raygenRecord != 0)
    {
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(this->_sbt.raygenRecord), this->_cuda_stream));
        this->_sbt.raygenRecord = 0;
    }

    if(this->_sbt.missRecordBase != 0)
    {
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(this->_sbt.missRecordBase), this->_cuda_stream));
        this->_sbt.missRecordBase = 0;
    }

    if(this->_sbt.hitgroupRecordBase != 0)
    {
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(this->_sbt.hitgroupRecordBase), this->_cuda_stream));
        this->_sbt.hitgroupRecordBase = 0;
    }

    for(CUdeviceptr ptr : this->_ptrs_to_free)
    {
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(ptr), this->_cuda_stream));
    }

    this->_ptrs_to_free.clear();

    CUDA_CHECK(cudaStreamSynchronize(this->_cuda_stream));

    if(this->_pipeline != nullptr)
    {
        optixPipelineDestroy(this->_pipeline);
        this->_pipeline = nullptr;
    }

    if(this->_raygen_group != nullptr)
    {
        optixProgramGroupDestroy(this->_raygen_group);
        this->_raygen_group = nullptr;
    }

    if(this->_miss_group != nullptr)
    {
        optixProgramGroupDestroy(this->_miss_group);
        this->_miss_group = nullptr;
    }

    if(this->_hit_group != nullptr)
    {
        optixProgramGroupDestroy(this->_hit_group);
        this->_hit_group = nullptr;
    }

    if(this->_module != nullptr)
    {
        optixModuleDestroy(this->_module);
        this->_module = nullptr;
    }

    if(this->_context != nullptr)
    {
        optixDeviceContextDestroy(this->_context);
        this->_context = nullptr;
    }

    // Clean up CUDA resources
    if(this->_cuda_mem_pool != nullptr)
    {
        CUDA_CHECK(cudaMemPoolDestroy(this->_cuda_mem_pool));
        this->_cuda_mem_pool = nullptr;
    }

    if(this->_cuda_stream != nullptr)
    {
        CUDA_CHECK(cudaStreamDestroy(this->_cuda_stream));
        this->_cuda_stream = nullptr;
    }

    this->_initialized = false;
#else
    CUDA_SYNC_CHECK();
    CUDA_CHECK(cudaStreamSynchronize(this->_cuda_stream));

    if(this->_pipeline != nullptr)
    {
        optixPipelineDestroy(this->_pipeline);
    }

    if(this->_context != nullptr)
    {
        optixDeviceContextDestroy(this->_context);
    }

    if(this->_module != nullptr)
    {
        optixModuleDestroy(this->_module);
    }

    CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(this->_sbt.raygenRecord), this->_cuda_stream));
    CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(this->_sbt.missRecordBase), this->_cuda_stream));
    CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(this->_sbt.hitgroupRecordBase), this->_cuda_stream));

    for(CUdeviceptr ptr : this->_ptrs_to_free)
    {
        CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(ptr), this->_cuda_stream));
    }

    CUDA_CHECK(cudaFreeAsync(reinterpret_cast<void*>(this->_params), this->_cuda_stream));

    CUDA_CHECK(cudaMemPoolDestroy(this->_cuda_mem_pool));

    CUDA_CHECK(cudaStreamSynchronize(this->_cuda_stream));
    CUDA_CHECK(cudaStreamDestroy(this->_cuda_stream));
#endif
}

ROMANORENDER_NAMESPACE_END
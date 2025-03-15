#include "romanorender/optix_utils.h"
#include "romanorender/cuda_vector.h"

#include "stdromano/filesystem.h"

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
    cudaFree(0);

    CUDA_CHECK(cudaGetDevice(&this->_cuda_device));
    CUDA_CHECK(cudaStreamCreate(&this->_cuda_stream));
}

void OptixManager::setup_optix() noexcept
{
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

    stdromano::String<> shaders_ptx_path = stdromano::expand_from_executable_dir("shaders/shaders.ptx");
    stdromano::String<> shaders_ptx = std::move(stdromano::load_file_content(shaders_ptx_path.data()));

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
    if(this->_sbt.raygenRecord != 0)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(this->_sbt.raygenRecord)));
    }

    if(this->_sbt.missRecordBase != 0)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(this->_sbt.missRecordBase)));
    }

    if(this->_sbt.hitgroupRecordBase != 0)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(this->_sbt.hitgroupRecordBase)));
    }

    // Raygen
    RaygenRecord rg_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(this->_raygen_group, &rg_record));

    CUdeviceptr d_raygen_record;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), sizeof(RaygenRecord)));
    CUDA_CHECK(
        cudaMemcpy(reinterpret_cast<void*>(d_raygen_record), &rg_record, sizeof(RaygenRecord), cudaMemcpyHostToDevice));

    // Miss
    MissRecord ms_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(this->_miss_group, &ms_record));

    CUdeviceptr d_miss_record;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record), sizeof(MissRecord)));
    CUDA_CHECK(
        cudaMemcpy(reinterpret_cast<void*>(d_miss_record), &ms_record, sizeof(MissRecord), cudaMemcpyHostToDevice));

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
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_record), hg_records.size() * sizeof(HitGroupRecord)));
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(d_hitgroup_record),
                          hg_records.data(),
                          hg_records.size() * sizeof(HitGroupRecord),
                          cudaMemcpyHostToDevice));

    this->_sbt.raygenRecord = d_raygen_record;
    this->_sbt.missRecordBase = d_miss_record;
    this->_sbt.missRecordStrideInBytes = sizeof(MissRecord);
    this->_sbt.missRecordCount = 1;
    this->_sbt.hitgroupRecordBase = d_hitgroup_record;
    this->_sbt.hitgroupRecordStrideInBytes = hg_records.size() * sizeof(HitGroupRecord);
    this->_sbt.hitgroupRecordCount = hg_records.size();
}

void OptixManager::update_params(const OptixParams* params) noexcept
{
    CUDA_CHECK(cudaMemcpy(reinterpret_cast<void*>(this->_params), params, sizeof(OptixParams), cudaMemcpyHostToDevice));
}

void OptixManager::launch(const size_t width, const size_t height, const size_t num_samples) noexcept
{
    OPTIX_CHECK(optixLaunch(this->_pipeline,
                            this->_cuda_stream,
                            this->_params,
                            sizeof(OptixParams),
                            &this->_sbt,
                            width,
                            height,
                            num_samples));
}

void OptixManager::initialize() noexcept
{
    if(!this->_initialized)
    {
        this->setup_cuda();
        this->setup_optix();
        this->create_pipeline();
        this->_initialized = true;

        CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&this->_params), sizeof(OptixParams)));
    }
}

void OptixManager::cleanup() noexcept
{
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

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(this->_sbt.raygenRecord)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(this->_sbt.missRecordBase)));
    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(this->_sbt.hitgroupRecordBase)));

    for(CUdeviceptr ptr : this->_ptrs_to_free)
    {
        CUDA_CHECK(cudaFree(reinterpret_cast<void*>(ptr)));
    }

    CUDA_CHECK(cudaFree(reinterpret_cast<void*>(this->_params)));
}

ROMANORENDER_NAMESPACE_END
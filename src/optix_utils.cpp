#include "romanorender/optix_utils.h"

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
    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_LEVEL_INSTANCING;
    pipelineCompileOptions.numPayloadValues = 2;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    stdromano::String<> shaders_ptx_path = stdromano::expand_from_executable_dir("shaders/shaders.ptx");
    stdromano::String<> shaders_ptx = std::move(stdromano::load_file_content(shaders_ptx_path.data()));

    if(shaders_ptx.empty())
    {
        stdromano::log_error("Error while loading shaders ptx code, cannot find file: {}", shaders_ptx_path);

        std::exit(1);
    }

    char log[4096];
    size_t log_size = sizeof(log);

    OptixModule module = nullptr;
    OPTIX_CHECK(optixModuleCreate(this->_context,
                                  &moduleCompileOptions,
                                  &pipelineCompileOptions,
                                  shaders_ptx.data(),
                                  shaders_ptx.size(),
                                  log,
                                  &log_size,
                                  &module));

    // Raygen
    OptixProgramGroup raygen_group;

    {
        OptixProgramGroupOptions options = {};
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
        desc.raygen.module = module;
        desc.raygen.entryFunctionName = "__raygen__rg";
        OPTIX_CHECK(optixProgramGroupCreate(this->_context, &desc, 1, &options, log, &log_size, &raygen_group));
    }

    // Miss
    OptixProgramGroup miss_group;

    {
        OptixProgramGroupOptions options = {};
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
        desc.miss.module = module;
        desc.miss.entryFunctionName = "__miss__ms";
        OPTIX_CHECK(optixProgramGroupCreate(this->_context, &desc, 1, &options, log, &log_size, &miss_group));
    }

    // Hit
    OptixProgramGroup hit_group;

    {
        OptixProgramGroupOptions options = {};
        OptixProgramGroupDesc desc = {};
        desc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
        desc.hitgroup.moduleCH = module;
        desc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
        OPTIX_CHECK(optixProgramGroupCreate(this->_context, &desc, 1, &options, log, &log_size, &hit_group));
    }


    OptixProgramGroup programGroups[] = {raygen_group, miss_group, hit_group};

    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 2;

    OPTIX_CHECK(optixPipelineCreate(this->_context,
                                    &pipelineCompileOptions,
                                    &pipelineLinkOptions,
                                    programGroups,
                                    sizeof(programGroups) / sizeof(programGroups[0]),
                                    log,
                                    &log_size,
                                    &this->_pipeline));

    this->create_sbt(raygen_group, miss_group, hit_group);
}

void OptixManager::create_sbt(OptixProgramGroup raygen_group,
                              OptixProgramGroup miss_group,
                              OptixProgramGroup hit_group) noexcept
{
    RaygenRecord rg_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(raygen_group, &rg_record));

    MissRecord ms_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(miss_group, &ms_record));

    HitGroupRecord hg_record;
    OPTIX_CHECK(optixSbtRecordPackHeader(hit_group, &hg_record));

    CUdeviceptr d_raygen_record;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_raygen_record), sizeof(RaygenRecord)));
    CUDA_CHECK(
        cudaMemcpy(reinterpret_cast<void*>(d_raygen_record), &rg_record, sizeof(RaygenRecord), cudaMemcpyHostToDevice));
    this->_ptrs_to_free.push_back(d_raygen_record);

    CUdeviceptr d_miss_record;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_miss_record), sizeof(MissRecord)));
    CUDA_CHECK(
        cudaMemcpy(reinterpret_cast<void*>(d_miss_record), &ms_record, sizeof(MissRecord), cudaMemcpyHostToDevice));
    this->_ptrs_to_free.push_back(d_miss_record);

    CUdeviceptr d_hitgroup_record;
    CUDA_CHECK(cudaMalloc(reinterpret_cast<void**>(&d_hitgroup_record), sizeof(HitGroupRecord)));
    CUDA_CHECK(cudaMemcpy(
        reinterpret_cast<void*>(d_hitgroup_record), &hg_record, sizeof(HitGroupRecord), cudaMemcpyHostToDevice));
    this->_ptrs_to_free.push_back(d_hitgroup_record);

    this->_sbt.raygenRecord = d_raygen_record;
    this->_sbt.missRecordBase = d_miss_record;
    this->_sbt.missRecordStrideInBytes = sizeof(MissRecord);
    this->_sbt.missRecordCount = 1;
    this->_sbt.hitgroupRecordBase = d_hitgroup_record;
    this->_sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupRecord);
    this->_sbt.hitgroupRecordCount = 1;
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
    if(this->_pipeline)
    {
        optixPipelineDestroy(this->_pipeline);
    }

    if(this->_context)
    {
        optixDeviceContextDestroy(this->_context);
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
#pragma once

#if !defined(__ROMANORENDER_OPTIX_UTILS)
#define __ROMANORENDER_OPTIX_UTILS

#include "romanorender/cuda_utils.h"
#include "romanorender/optix_params.h"
#include "romanorender/romanorender.h"

#include "stdromano/logger.h"
#include "stdromano/vector.h"

#include <optix.h>
#include <optix_stubs.h>

ROMANORENDER_NAMESPACE_BEGIN

#define OPTIX_CHECK(call)                                                                                              \
    {                                                                                                                  \
        OptixResult res = call;                                                                                        \
        if(res != OPTIX_SUCCESS)                                                                                       \
        {                                                                                                              \
            stdromano::log_error("[OPTIX ERROR] OptiX call \"{}\" failed with error:"                                  \
                                 " {} ({}:{})\n",                                                                      \
                                 #call,                                                                                \
                                 optixGetErrorName(res),                                                               \
                                 __FILE__,                                                                             \
                                 __LINE__);                                                                            \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    }

template <typename T>
struct SbtRecord
{
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};

using RaygenRecord = SbtRecord<void*>;
using MissRecord = SbtRecord<void*>;
using HitGroupRecord = SbtRecord<void*>;

class ROMANORENDER_API OptixManager
{
public:
    static OptixManager& get_instance()
    {
        static OptixManager instance;
        return instance;
    }

    OptixDeviceContext get_context() const { return this->_context; }

    OptixPipeline get_pipeline() const { return this->_pipeline; }

    CUstream get_stream() const { return this->_cuda_stream; }

    void update_params(const OptixParams* params) noexcept;

    void launch(const size_t width, const size_t height, const size_t num_samples = 1) noexcept;

    OptixManager(const OptixManager&) = delete;
    void operator=(const OptixManager&) = delete;

private:
    OptixManager() { this->initialize(); };

    ~OptixManager() { this->cleanup(); }

    void setup_cuda() noexcept;
    void setup_optix() noexcept;
    void create_pipeline() noexcept;
    void create_sbt(OptixProgramGroup raygen_group, OptixProgramGroup miss_group, OptixProgramGroup hit_group) noexcept;

    void initialize() noexcept;
    void cleanup() noexcept;

    bool _initialized = false;

    OptixDeviceContext _context = nullptr;
    OptixPipeline _pipeline = nullptr;
    OptixShaderBindingTable _sbt;

    CUdeviceptr _params;

    CUstream _cuda_stream;

    stdromano::Vector<CUdeviceptr> _ptrs_to_free;

    int _cuda_device;
};

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_OPTIX_UTILS) */
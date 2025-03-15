#include "romanorender/cuda_utils.h"
#include "stdromano/logger.h"

ROMANORENDER_NAMESPACE_BEGIN

void log_cuda_version() noexcept
{
    int runtime_ver;
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime_ver));
    stdromano::log_info("CUDA Runtime version: {}", runtime_ver);

    int driver_ver;
    CUDA_CHECK(cudaDriverGetVersion(&driver_ver));
    stdromano::log_info("CUDA Driver version: {}", driver_ver);
}

ROMANORENDER_NAMESPACE_END
#pragma once

#if !defined(__ROMANORENDER_CUDA_UTILS)
#define __ROMANORENDER_CUDA_UTILS

#include "romanorender/romanorender.h"

#include "stdromano/logger.h"

#include <cstdio>
#include <cstdlib>

#include <cuda.h>
#include <cuda_runtime.h>

ROMANORENDER_NAMESPACE_BEGIN

#define CUDA_CHECK(call)                                                                                               \
    {                                                                                                                  \
        cudaError_t res = (cudaError_t)call;                                                                           \
        if(res != cudaSuccess)                                                                                         \
        {                                                                                                              \
            stdromano::log_error("[CUDA ERROR] Cuda call \"{}\" failed with error:"                                    \
                                 " {} ({}:{})\n",                                                                      \
                                 #call,                                                                                \
                                 cudaGetErrorString(res),                                                              \
                                 __FILE__,                                                                             \
                                 __LINE__);                                                                            \
            std::exit(1);                                                                                              \
        }                                                                                                              \
    }

#define CUDA_SYNC_CHECK() CUDA_CHECK(cudaDeviceSynchronize())

ROMANORENDER_API void log_cuda_version() noexcept;

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_CUDA_UTILS) */
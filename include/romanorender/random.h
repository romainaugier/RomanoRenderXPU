#pragma once

#if !defined(__ROMANORENDER_RANDOM)
#define __ROMANORENDER_RANDOM

#include "romanorender/romanorender.h"

ROMANORENDER_NAMESPACE_BEGIN

ROMANORENDER_API void seed_thread_rngs() noexcept;

ROMANORENDER_API uint32_t pcg_random_uint32(const uint32_t seed) noexcept;

ROMANORENDER_API uint32_t pcg_next_uint32() noexcept;

ROMANORENDER_API float pcg_next_float() noexcept;

ROMANORENDER_API uint64_t xoshiro_random_uint64(const uint64_t seed) noexcept;

ROMANORENDER_API uint64_t xoshiro_next_uint64() noexcept;

ROMANORENDER_API float xoshiro_next_float() noexcept;

ROMANORENDER_API void seed_pcg(uint64_t seed) noexcept;

ROMANORENDER_API void seed_xoshiro(uint64_t seed) noexcept;

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_RANDOM) */
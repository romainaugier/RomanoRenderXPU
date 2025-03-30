#pragma once

#if !defined(__ROMANORENDER_SAMPLING)
#define __ROMANORENDER_SAMPLING

#include "romanorender/maths.h"
#include "romanorender/vec3.h"
#include "romanorender/vec2.h"
#include "romanorender/cuda_vector.h"
#include "romanorender/random.h"

#include "stdromano/hash.h"
#include "stdromano/vector.h"

ROMANORENDER_NAMESPACE_BEGIN

ROMANORENDER_API Vec3F sample_hemisphere(const Vec3F& hit_normal, const float rx, const float ry) noexcept;

ROMANORENDER_API Vec3F sample_hemisphere_unsafe(const Vec3F& hit_normal, const float rx, const float ry) noexcept;

/* PMJ02 */

/* Adapted from here: https://github.com/Andrew-Helmer/stochastic-generation/blob/main/sampling/ssobol.cpp */

CudaVector<Vec2F> get_pmj02_samples(const uint32_t num_samples,
                                    const uint32_t n_candidates,
                                    const uint32_t seed) noexcept;

/* Sampler singleton */

#define NUM_PMJ02_SEQUENCES 128
#define NUM_PMJ02_SAMPLES 4096

class ROMANORENDER_API Sampler
{
public:
    static Sampler& get_instance()
    {
        static Sampler myInstance;

        return myInstance;
    }

    Sampler(Sampler const&) = delete;
    Sampler(Sampler&&) = delete;
    Sampler& operator=(Sampler const&) = delete;
    Sampler& operator=(Sampler&&) = delete;

    size_t get_memory_usage() const noexcept;

    void initialize() noexcept;

    Vec2F get_pmj02_sample(uint32_t sequence, uint32_t sample) const noexcept;

    const Vec2F* get_pmj02_sequence_ptr(uint32_t sequence) const noexcept;

private:
    Sampler() {}

    ~Sampler() {}

    CudaVector<CudaVector<Vec2F>> _pmjs;
};

#define sampler() Sampler::get_instance()

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_SAMPLING) */
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

ROMANORENDER_FORCE_INLINE Vec2F sample_gaussian(const Vec2F& uv) noexcept
{
    const float f = maths::sqrtf(-2.0f * maths::logf(uv.x));
    const float a = maths::constants::two_pi * uv.y;

    const float cos_a = maths::cosf(a);
    const float sin_a = maths::sinf(a);

    return Vec2F(cos_a, sin_a) * f;
}

ROMANORENDER_FORCE_INLINE Vec2F sample_triangle(const Vec2F& uv) noexcept
{
    float u = uv.x;
    float v = uv.y;

    if(v > u)
    {
        u *= 0.5f;
        v -= u;
    }
    else
    {
        v *= 0.5f;
        u -= v;
    }

    return Vec2F(u, v);
}

ROMANORENDER_API Vec3F sample_hemisphere(const Vec3F& hit_normal, const float rx, const float ry) noexcept;

ROMANORENDER_API Vec3F sample_hemisphere_unsafe(const Vec3F& hit_normal, const float rx, const float ry) noexcept;

ROMANORENDER_FORCE_INLINE Vec2F sample_disk(const Vec2F& uv) noexcept
{
    const float theta = maths::constants::two_pi * uv.x;
    const float r = maths::sqrtf(uv.y);
    return Vec2F(maths::cosf(theta), maths::sinf(theta)) * r;
}

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

    ROMANORENDER_FORCE_INLINE Vec2F get_pmj02_sample(uint32_t sequence, uint32_t sample) const noexcept
    {
        sequence = sequence % NUM_PMJ02_SEQUENCES;

        return this->_pmjs[sequence][sample];
    }

    ROMANORENDER_FORCE_INLINE const Vec2F* get_pmj02_sequence_ptr(uint32_t sequence) const noexcept
    {
        ROMANORENDER_ASSERT(sequence < NUM_PMJ02_SEQUENCES, "sequence should be lower than the total number of sequences");

        return this->_pmjs[sequence].data();
    }

private:
    Sampler() {}

    ~Sampler() {}

    CudaVector<CudaVector<Vec2F>> _pmjs;
};

#define sampler() Sampler::get_instance()

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_SAMPLING) */
#pragma once

#if !defined(__ROMANORENDER_SAMPLING)
#define __ROMANORENDER_SAMPLING

#include "romanorender/maths.h"
#include "romanorender/vec3.h"
#include "romanorender/vec2.h"
#include "romanorender/cuda_vector.h"
#include "romanorender/random.h"

#include "stdromano/hash.hpp"
#include "stdromano/vector.hpp"

ROMANORENDER_NAMESPACE_BEGIN

ROMANORENDER_FORCE_INLINE Vec2F sample_gaussian(const Vec2F& uv) noexcept
{
    const float f = maths::sqrtf(-2.0f * maths::logf(uv.x));
    const float a = maths::constants::two_pi * uv.y;

    float cos_a, sin_a;
    maths::sincosf(a, &sin_a, &cos_a);

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

ROMANORENDER_FORCE_INLINE Vec2F sample_disk_uniform(const Vec2F& uv) noexcept
{
    const float theta = maths::constants::two_pi * uv.x;
    const float r = maths::sqrtf(uv.y);
    return Vec2F(maths::cosf(theta), maths::sinf(theta)) * r;
}

ROMANORENDER_FORCE_INLINE Vec2F sample_disk_uniform_concentric(const Vec2F& uv) noexcept
{
    Vec2F u_offset = 2 * uv - Vec2F(1, 1);

    if(u_offset.x == 0 && u_offset.y == 0)
    {
        return Vec2F(0.0f, 0.0f);
    }

    float theta, r;

    if(maths::absf(u_offset.x) > maths::absf(u_offset.y)) 
    {
        r = u_offset.x;
        theta = maths::constants::pi_over_four * (u_offset.y / u_offset.x);
    } 
    else 
    {
        r = u_offset.y;
        theta = maths::constants::pi_over_two - maths::constants::pi_over_four * (u_offset.x / u_offset.y);
    }

    float sin, cos;
    maths::sincosf(theta, &sin, &cos);

    return r * Vec2F(cos, sin);
}

ROMANORENDER_FORCE_INLINE Vec3F sample_hemisphere_uniform(const Vec2F& uv) noexcept
{
    const float z = uv.x;
    const float r = maths::sqrtf(1.0f - maths::sqrf(z));
    const float phi = 2 * maths::constants::pi * uv.y;

    float cos, sin;
    maths::sincosf(phi, &sin, &cos);

    return Vec3F(r * cos, r * sin, z);
}

ROMANORENDER_FORCE_INLINE float sample_hemisphere_uniform_pdf() noexcept
{
    return maths::constants::inv_two_pi;
}

ROMANORENDER_FORCE_INLINE Vec3F sample_hemisphere_cosine(const Vec2F& uv) noexcept
{
    const Vec2F d = sample_disk_uniform_concentric(uv);
    const float z = maths::sqrtf(1.0f - maths::sqrf(d.x) - maths::sqrf(d.y));
    return Vec3F(d.x, d.y, z);
}

ROMANORENDER_FORCE_INLINE float sample_hemisphere_cosine_pdf(const float cos_theta) noexcept
{
    return cos_theta * maths::constants::inv_pi;
}

ROMANORENDER_FORCE_INLINE Vec3F spherical_direction(const float sin_theta, const float cos_theta, const float phi) noexcept
{
    float sin, cos;
    maths::sincosf(phi, &sin, &cos);

    return Vec3F(maths::clampf(sin_theta, -1.0f, 1.0f) * cos,
                 maths::clampf(sin_theta, -1.0f, 1.0f) * sin,
                 maths::clampf(cos_theta, -1.0f, 1.0f));
}

ROMANORENDER_FORCE_INLINE Vec3F sample_cone_uniform(const Vec2F& uv, const float cos_theta_max) noexcept
{
    const float cos_theta = (1.0f - uv.x) + uv.x * cos_theta_max;
    const float sin_theta = maths::sqrtf(1.0f - maths::sqrf(cos_theta));
    const float phi = uv.y * maths::constants::two_pi;

    return spherical_direction(sin_theta, cos_theta, phi);
}

ROMANORENDER_FORCE_INLINE float sample_cone_uniform_pdf(const float cos_theta_max) noexcept
{
    return maths::rcpf(maths::constants::two_pi * (1.0f - cos_theta_max));
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
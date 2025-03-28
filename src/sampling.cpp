#include "romanorender/sampling.h"

#include "stdromano/random.h"

#include <unordered_set>
#include <random>

ROMANORENDER_NAMESPACE_BEGIN

Vec3F sample_hemisphere(const Vec3F& hit_normal, const float rx, const float ry) noexcept
{
    float signZ = (hit_normal.z >= 0.0f) ? 1.0f : -1.0f;
    float a = -1.0f / (signZ + hit_normal.z);
    float b = hit_normal.x * hit_normal.y * a;
    Vec3F b1 = Vec3F(1.0f + signZ * hit_normal.x * hit_normal.x * a,
                     signZ * b,
                     -signZ * hit_normal.x);
    Vec3F b2 = Vec3F(b, signZ + hit_normal.y * hit_normal.y * a, -hit_normal.y);

    float phi = 2.0f * maths::constants::pi * rx;
    float cosTheta = maths::sqrtf(ry);
    float sinTheta = maths::sqrtf(1.0f - ry);

    return normalize_vec3f(((b1 * maths::cosf(phi) + b2 * maths::sinf(phi)) * cosTheta
                            + hit_normal * sinTheta));
}

Vec3F sample_hemisphere_unsafe(const Vec3F& hit_normal, const float rx, const float ry) noexcept
{
    float a = 1.0f - 2.0f * rx;
    float b = maths::sqrtf(1.0f - a * a);
    float phi = 2.0f * maths::constants::pi * ry;

    return Vec3F(hit_normal.x + b * maths::cosf(phi),
                 hit_normal.y + b * maths::sinf(phi),
                 hit_normal.z + a);
}

/* PMJ02 */

/* https://jcgt.org/published/0008/01/04/ */

struct PMJ02State
{
    std::unordered_set<PMJ02CellKey> occupied_cells;
    std::unordered_set<uint64_t> occupied_rows;
    std::unordered_set<uint64_t> occupied_cols;

    uint32_t current_max_level = 0;
};

Vec2F generate_pmj02_candidate(const uint32_t seed, 
                               const uint32_t p,
                               const PMJ02State& state,
                               const stdromano::Vector<Vec2F>& samples) noexcept
{
    std::mt19937 gen(seed + p);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);

    if(samples.empty())
    {
        return Vec2F(dist(gen), dist(gen));
    }

    const uint32_t level = state.current_max_level;
    const uint32_t grid_size = 1 << level;

    if(samples.size() < (grid_size * grid_size))
    {
        const uint32_t parent_grid = (grid_size > 1) ? (grid_size / 2) : 1;

        const uint32_t px = static_cast<uint32_t>(samples.back().x * parent_grid);
        const uint32_t py = static_cast<uint32_t>(samples.back().y * parent_grid);

        const uint32_t sx = (parent_grid - 1 - px) * 2;
        const uint32_t sy = (parent_grid - 1 - py) * 2;

        const float xmin = static_cast<float>(sx) / (2.0f * parent_grid);
        const float ymin = static_cast<float>(sy) / (2.0f * parent_grid);

        return Vec2F(maths::clampf((xmin + dist(gen)) / (2.0f * parent_grid), 0.0f, 1.0f - FLT_EPSILON),
                     maths::clampf((ymin + dist(gen)) / (2.0f * parent_grid), 0.0f, 1.0f - FLT_EPSILON));
    }

    std::uniform_real_distribution<float> candidate_dist(0.0f, 1.0f - FLT_EPSILON);

    for(uint32_t tries = 0; tries < 1000; tries++)
    {
        Vec2F candidate(candidate_dist(gen),
                        candidate_dist(gen));

        bool valid = true;

        for(uint32_t l = 0; l <= state.current_max_level; ++l)
        {
            const uint32_t g = 1 << l;
            const uint32_t i = static_cast<uint32_t>(candidate.x * g);
            const uint32_t j = static_cast<uint32_t>(candidate.y * g);

            if(state.occupied_cells.count({l, i, j}))
            {
                valid = false;
                break;
            }

            const uint64_t row_key = (static_cast<uint64_t>(l) << 32) | i;
            const uint64_t col_key = (static_cast<uint64_t>(l) << 32) | j;

            if(state.occupied_rows.count(row_key) || 
               state.occupied_cols.count(col_key))
            {
                valid = false;
                break;
            }
        }

        if(valid)
        {
            return candidate;
        }
    }

    return Vec2F(dist(gen), dist(gen));
}

stdromano::Vector<Vec2F> generate_pmj02_samples(const uint32_t num_samples, const uint32_t seed) noexcept
{
    stdromano::Vector<Vec2F> samples;
    PMJ02State state;

    uint32_t n = 0;

    while(samples.size() < num_samples)
    {
        Vec2F candidate = generate_pmj02_candidate(seed, n++, state, samples);

        for(uint32_t l = 0; l <= state.current_max_level; ++l)
        {
            const uint32_t g = 1 << l;
            const uint32_t i = static_cast<uint32_t>(candidate.x * g);
            const uint32_t j = static_cast<uint32_t>(candidate.y * g);

            state.occupied_cells.insert({l, i, j});

            state.occupied_rows.insert((static_cast<uint64_t>(l) << 32) | i);
            state.occupied_cols.insert((static_cast<uint64_t>(l) << 32) | j);
        }

        samples.push_back(candidate);
        
        if((samples.size() & (samples.size() - 1)) == 0)
        {
            state.current_max_level = static_cast<uint32_t>(maths::log2f(samples.size()));
        }
    }

    return std::move(samples);
}

ROMANORENDER_NAMESPACE_END
#include "romanorender/sampling.h"

#include "stdromano/logger.h"
#include "stdromano/random.h"

#define STDROMANO_ENABLE_PROFILING
#include "stdromano/profiling.h"

#include <deque>
#include <random>

ROMANORENDER_NAMESPACE_BEGIN

/* PMJ02 */

int get_grid_index(const Vec2F& point, const int32_t grid_width) noexcept
{
    const int32_t x_pos = static_cast<int32_t>(point.x * grid_width);
    const int32_t y_pos = static_cast<int32_t>(point.y * grid_width);

    return y_pos * grid_width + x_pos;
}

float get_toroidal_distance_sq(const Vec2F& p0, const Vec2F& p1) noexcept
{
    float x_diff = maths::absf(p1.x - p0.x);

    if(x_diff > 0.5f) 
    {
        x_diff = 1.0f - x_diff;
    }

    float y_diff = maths::absf(p1.y - p0.y);

    if(y_diff > 0.5f)
    {
        y_diff = 1.0f - y_diff;
    }

    return (x_diff * x_diff) + (y_diff * y_diff);
}

int32_t wrap_int(const int32_t index, const int32_t limit) 
{
    if(index < 0) 
    {
        return index + limit;
    }

    if(index >= limit) 
    {
        return index - limit;
    }

    return index;
}

class SampleGrid2D
{
public:
    SampleGrid2D() : _sample_grid(1, nullptr) {}

    void add_sample(const Vec2F& sample) noexcept
    {
        int grid_idx = get_grid_index(sample, this->_width);

        if(this->_sample_grid[grid_idx] != nullptr) 
        {
            Vec2F conflicting_point = *this->_sample_grid[grid_idx];
            int32_t subdivisions = 1;
            int32_t temp_width = this->_width * 2;

            while(static_cast<int32_t>(conflicting_point.x * temp_width) == static_cast<int32_t>(sample.x * temp_width) &&
                  static_cast<int32_t>(conflicting_point.y * temp_width) == static_cast<int32_t>(sample.y * temp_width)) 
            {
                subdivisions++;
                temp_width <<= 1;
            }

            this->subdivide_grid(subdivisions);
            grid_idx = get_grid_index(sample, this->_width);
        }

        this->_points.push_back(sample);
        this->_sample_grid[grid_idx] = &(this->_points.back());
    }

    int32_t get_best_candidate(const stdromano::Vector<Vec2F>& candidates) const
    {
        int32_t best_candidate = 0;
        Vec2F prev_nearest;

        float max_min_dist_sq = this->get_min_dist_sq_fast(candidates[0], 0.0, &prev_nearest);
        const int n_candidates = candidates.size();

        for(int i = 1; i < n_candidates; i++) 
        {
            if(get_toroidal_distance_sq(candidates[i], prev_nearest) < max_min_dist_sq)
            {
                continue;
            }

            const float min_dist_sq = this->get_min_dist_sq_fast(candidates[i], max_min_dist_sq, &prev_nearest);

            if(min_dist_sq > max_min_dist_sq) 
            {
                max_min_dist_sq = min_dist_sq;
                best_candidate = i;
            }
        }

        return best_candidate;
    }

private:
    float get_min_dist_sq_fast(const Vec2F& sample, const float max_min_dist_sq, Vec2F* nearest) const
    {
        float min_dist_sq = get_toroidal_distance_sq(sample, this->_points[0]);
        const Vec2F* nearest_ptr = &this->_points[0];

        const float cell_width = 1.0 / this->_width;

        int32_t outer_width = 1;
        int32_t num_cells_to_check = 1;
        int32_t starting_x_pos = sample.x * this->_width;
        int32_t starting_y_pos = sample.y * this->_width;

        while(outer_width <= (this->_width + 1)) 
        {
            int32_t x_pos = starting_x_pos, y_pos = starting_y_pos;
            int32_t x_dir = 1, y_dir = 0;

            for(int32_t i = 0; i < num_cells_to_check; i++) 
            {
                const int32_t lookup_x_pos = wrap_int(x_pos, this->_width);
                const int32_t lookup_y_pos = wrap_int(y_pos, this->_width);
                const int32_t grid_idx = lookup_y_pos * this->_width + lookup_x_pos;

                Vec2F* point = this->_sample_grid[grid_idx];

                if(point != nullptr) 
                {
                    const float dist_sq = get_toroidal_distance_sq(sample, *point);

                    if(dist_sq < min_dist_sq) 
                    {
                        min_dist_sq = dist_sq;
                        nearest_ptr = point;

                        if(min_dist_sq < max_min_dist_sq)
                        {
                            break;
                        }
                    }
                }

                if(i > 0 && (i % (outer_width-1) == 0)) 
                {
                    y_dir = -y_dir;
                    std::swap(x_dir, y_dir);
                }

                x_pos += x_dir;
                y_pos += y_dir;
            }

            num_cells_to_check = 2 * (2 * outer_width + 2);
            outer_width += 2;
            starting_x_pos--;
            starting_y_pos--;

            const float min_grid_dist = maths::maxf((outer_width / 2) - 1, 0) * cell_width;

            if (min_dist_sq < (min_grid_dist*min_grid_dist) ||
                min_dist_sq < max_min_dist_sq) 
            {
                break;
            }
        }

        if(nearest != nullptr) 
        {
            *nearest = *nearest_ptr;
        }

        return min_dist_sq;
    }

    void subdivide_grid(const int32_t subdivisions) noexcept
    {
        this->_width <<= subdivisions;
        this->_sample_grid = stdromano::Vector<Vec2F*>(this->_width * this->_width, nullptr);

        const int num_points = this->_points.size();
        
        for(int32_t i = 0; i < num_points; i++) 
        {
            const int32_t grid_idx = get_grid_index(this->_points[i], this->_width);
            this->_sample_grid[grid_idx] = &this->_points[i];
        }
    }

    stdromano::Vector<Vec2F*> _sample_grid;
    std::deque<Vec2F> _points;
    int32_t _width = 1;
};

Vec2F get_pmj02_point(const int32_t x_stratum, const int32_t y_stratum, const float i_strata, const uint32_t seed) noexcept
{
    return Vec2F((stdromano::wang_hash_float(seed + 0x9876) + x_stratum) * i_strata,
                 (stdromano::wang_hash_float(seed + 0x6789) + y_stratum) * i_strata);
}

static constexpr uint32_t pmj02_xors[2][32] = {
    {0x0, 0x0, 0x2, 0x6, 0x6, 0xe, 0x36, 0x4e, 0x16, 0x2e, 0x276, 0x6ce, 0x716, 0xc2e, 0x3076, 0x40ce, 0x116, 0x22e, 0x20676, 0x60ece, 0x61716, 0xe2c2e, 0x367076, 0x4ec0ce, 0x170116, 0x2c022e, 0x2700676, 0x6c00ece, 0x7001716, 0xc002c2e, 0x30007076, 0x4000c0ce},
    {0x0, 0x1, 0x3, 0x3, 0x7, 0x1b, 0x27, 0xb, 0x17, 0x13b, 0x367, 0x38b, 0x617, 0x183b, 0x2067, 0x8b, 0x117, 0x1033b, 0x30767, 0x30b8b, 0x71617, 0x1b383b, 0x276067, 0xb808b, 0x160117, 0x138033b, 0x3600767, 0x3800b8b, 0x6001617, 0x1800383b, 0x20006067, 0x808b}
};

CudaVector<Vec2F> get_pmj02_samples(const uint32_t num_samples,
                                    const uint32_t n_candidates,
                                    const uint32_t seed) noexcept
{
    CudaVector<Vec2F> samples;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> rand_float(0.0, 1.0);

    samples.emplace_back(rand_float(rng), rand_float(rng));

    SampleGrid2D sample_grid;

    if(n_candidates > 1)
    {
        sample_grid.add_sample(samples[0]);
    }

    stdromano::Vector<Vec2F> candidates(n_candidates);

    for(int32_t log_n = 0; (1 << log_n) < num_samples; log_n++)
    {
        const int32_t prev_len = 1 << log_n;
        const int32_t n_strata = prev_len * 2;
        const float i_strata = 1.0 / n_strata;

        for(int i = 0; i < prev_len && (prev_len + i) < num_samples; i++)
        {
            const int32_t prev_x_idx = i ^ pmj02_xors[0][log_n];
            const int32_t prev_x_stratum = static_cast<int32_t>(samples[prev_x_idx].x * n_strata);
            const int32_t x_stratum = prev_x_stratum ^ 1;

            const int32_t prev_y_idx = i ^ pmj02_xors[1][log_n];
            const int32_t prev_y_stratum = static_cast<int32_t>(samples[prev_y_idx].y * n_strata);
            const int32_t y_stratum = prev_y_stratum ^ 1;

            Vec2F sample((rand_float(rng) + x_stratum) * i_strata, (rand_float(rng) + y_stratum) * i_strata);

            if(n_candidates > 1)
            {
                candidates[0] = { sample };

                for(int cand_idx = 1; cand_idx < n_candidates; cand_idx++)
                {
                    const Vec2F candidate((rand_float(rng) + x_stratum) * i_strata, (rand_float(rng) + y_stratum) * i_strata);
                    candidates[cand_idx] = candidate;
                }

                int32_t best_candidate = sample_grid.get_best_candidate(candidates);
                sample_grid.add_sample(candidates[best_candidate]);

                samples.push_back(candidates[best_candidate]);
            }
        }
    }

    return samples;
}

/* Sampler singleton */

void Sampler::initialize() noexcept
{
    SCOPED_PROFILE_START(stdromano::ProfileUnit::Seconds, sampler_initialization);

    for(uint32_t i = 0; i < NUM_PMJ02_SEQUENCES; i++)
    {
        this->_pmjs.emplace_back(std::move(get_pmj02_samples(NUM_PMJ02_SAMPLES, 2, i)));
    }

    stdromano::log_debug("Initialized Sampler with {} sequences of {} pmj02 samples", 
                         NUM_PMJ02_SEQUENCES,
                         NUM_PMJ02_SAMPLES);
}

ROMANORENDER_NAMESPACE_END
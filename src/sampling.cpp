#include "romanorender/sampling.h"

#include "stdromano/random.h"

#include <random>
#include <unordered_set>


ROMANORENDER_NAMESPACE_BEGIN

Vec3F sample_hemisphere(const Vec3F& hit_normal, const float rx, const float ry) noexcept
{
    float signZ = (hit_normal.z >= 0.0f) ? 1.0f : -1.0f;
    float a = -1.0f / (signZ + hit_normal.z);
    float b = hit_normal.x * hit_normal.y * a;
    Vec3F b1 = Vec3F(1.0f + signZ * hit_normal.x * hit_normal.x * a, signZ * b, -signZ * hit_normal.x);
    Vec3F b2 = Vec3F(b, signZ + hit_normal.y * hit_normal.y * a, -hit_normal.y);

    float phi = 2.0f * maths::constants::pi * rx;
    float cosTheta = maths::sqrtf(ry);
    float sinTheta = maths::sqrtf(1.0f - ry);

    return normalize_vec3f(((b1 * maths::cosf(phi) + b2 * maths::sinf(phi)) * cosTheta + hit_normal * sinTheta));
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
    std::unordered_set<uint64_t> occupied;
    stdromano::Vector<Vec2F> samples;
    uint32_t current_level = 0;
};

struct TreeNode
{
    bool occupied = false;
    std::unique_ptr<TreeNode> children[2];
};

void build_invalid_tree(TreeNode& node,
                        const uint32_t depth,
                        const uint32_t max_depth,
                        const uint32_t nx,
                        const uint32_t ny,
                        const uint32_t x,
                        const uint32_t y,
                        const std::unordered_set<uint64_t>& occupied)
{
    if(depth > max_depth)
        return;

    const uint64_t key = (static_cast<uint64_t>(nx) << 48) | (static_cast<uint64_t>(ny) << 32) | (x << 16) | y;
    node.occupied = occupied.count(key);

    if(!node.occupied)
    {
        for(int i = 0; i < 2; i++)
        {
            node.children[i] = std::make_unique<TreeNode>();
            build_invalid_tree(*node.children[i], depth + 1, max_depth, nx * 2, ny / 2, x * 2 + i, y / 2, occupied);
        }
    }
}

void collect_valid(const TreeNode& node, uint32_t depth, uint32_t max_depth, uint32_t x, uint32_t y, stdromano::Vector<uint32_t>& offsets)
{
    if(node.occupied)
        return;

    if(depth == max_depth)
    {
        offsets.push_back(x);
        return;
    }

    for(int i = 0; i < 2; i++)
    {
        collect_valid(*node.children[i], depth + 1, max_depth, x * 2 + i, y / 2, offsets);
    }
}

Vec2F generate_candidate(PMJ02State& state, uint32_t seed)
{
    const uint32_t n = state.samples.size();
    std::mt19937 gen(seed + n);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f - std::numeric_limits<float>::epsilon());

    if(n == 0)
    {
        return Vec2F(dist(gen), dist(gen));
    }

    const uint32_t grid_size = 1 << state.current_level;
    const uint32_t stratum_x = n % grid_size;
    const uint32_t stratum_y = n / grid_size;
    const uint32_t max_depth = static_cast<uint32_t>(log2(n + 1));

    TreeNode x_root, y_root;
    build_invalid_tree(x_root, 0, max_depth, grid_size, grid_size, stratum_x, stratum_y, state.occupied);
    build_invalid_tree(y_root, 0, max_depth, grid_size, grid_size, stratum_y, stratum_x, state.occupied);

    stdromano::Vector<uint32_t> x_offsets, y_offsets;
    collect_valid(x_root, 0, max_depth, stratum_x, stratum_y, x_offsets);
    collect_valid(y_root, 0, max_depth, stratum_y, stratum_x, y_offsets);

    if (x_offsets.empty() || y_offsets.empty()) 
    {
        return Vec2F(dist(gen), dist(gen));
    }

    std::uniform_int_distribution<uint32_t> idx_dist(0, x_offsets.size() - 1);
    const uint32_t x_idx = x_offsets[idx_dist(gen)];
    const uint32_t y_idx = y_offsets[idx_dist(gen)];

    const float scale = 1.0f / (1 << max_depth);
    const float x_jitter = dist(gen) * scale;
    const float y_jitter = dist(gen) * scale;

    return Vec2F((x_idx * scale + x_jitter), (y_idx * scale + y_jitter));
}

stdromano::Vector<Vec2F> generate_pmj02_samples(const uint32_t num_samples, const uint32_t seed) noexcept
{
    PMJ02State state;

    while(state.samples.size() < num_samples)
    {
        Vec2F candidate = generate_candidate(state, seed);

        for(uint32_t l = 0; l <= state.current_level; l++)
        {
            const uint32_t nx = 1 << l;
            const uint32_t ny = 1 << (state.current_level - l);
            const uint32_t x = static_cast<uint32_t>(candidate.x * nx);
            const uint32_t y = static_cast<uint32_t>(candidate.y * ny);
            const uint64_t key = (static_cast<uint64_t>(nx) << 48) | (static_cast<uint64_t>(ny) << 32) | (x << 16) | y;
            state.occupied.insert(key);
        }

        state.samples.push_back(candidate);

        if((state.samples.size() & (state.samples.size() - 1)) == 0)
        {
            state.current_level = static_cast<uint32_t>(maths::log2f(static_cast<float>(state.samples.size())));
        }
    }

    return std::move(state.samples);
}

ROMANORENDER_NAMESPACE_END
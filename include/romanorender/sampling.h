#pragma once

#if !defined(__ROMANORENDER_SAMPLING)
#define __ROMANORENDER_SAMPLING

#include "romanorender/maths.h"
#include "romanorender/vec3.h"
#include "romanorender/vec2.h"

#include "stdromano/hash.h"
#include "stdromano/vector.h"

ROMANORENDER_NAMESPACE_BEGIN

ROMANORENDER_API Vec3F sample_hemisphere(const Vec3F& hit_normal, const float rx, const float ry) noexcept;

ROMANORENDER_API Vec3F sample_hemisphere_unsafe(const Vec3F& hit_normal, const float rx, const float ry) noexcept;

/* PMJ02 */

struct ROMANORENDER_API PMJ02CellKey
{
    uint32_t level, i, j;

    bool operator==(const PMJ02CellKey& other) const noexcept
    {
        return this->level == other.level && 
               this->i == other.i &&
               this->j == other.j;
    }
};

ROMANORENDER_API stdromano::Vector<Vec2F> generate_pmj02_samples(const uint32_t num_samples, const uint32_t seed) noexcept;

ROMANORENDER_NAMESPACE_END

template <>
struct std::hash<romanorender::PMJ02CellKey>
{
    std::size_t operator()(const romanorender::PMJ02CellKey& ck) const
    {
        return static_cast<std::size_t>((std::hash<uint32_t>()(ck.level)) ^
                                        (std::hash<uint32_t>()(ck.i) << 1) ^
                                        (std::hash<uint32_t>()(ck.j) << 2));
    }
};

#endif /* !defined(__ROMANORENDER_SAMPLING) */
#define TINYBVH_IMPLEMENTATION
#define TINY_BVH_USE_COPY_MOVE_SEMANTICS
#include "romanorender/tbvh.h"

namespace tbvh {
    Vec4F::Vec4F(const Vec3F& a) : romanorender::Vec4F(a.x, a.y, a.z, 0.0f) {}

    Vec4F::Vec4F(const Vec3F& a, float b) : romanorender::Vec4F(a.x, a.y, a.z, b) {}
}
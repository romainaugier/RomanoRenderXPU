#define TINYBVH_IMPLEMENTATION
#include "romanorender/tbvh.h"

namespace tbvh {
    Vec4F::Vec4F(const Vec3F& a) : romanorender::Vec4F(a.x, a.y, a.z, 0.0f) {}

    Vec4F::Vec4F(const Vec3F& a, float b) : romanorender::Vec4F(a.x, a.y, a.z, b) {}
}
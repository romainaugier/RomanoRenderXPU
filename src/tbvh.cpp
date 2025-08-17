#include "stdromano/memory.hpp"
#define TINYBVH_IMPLEMENTATION
#include "romanorender/tbvh.h"

namespace tbvh {

Vec4F::Vec4F(const Vec3F& a) : romanorender::Vec4F(a.x, a.y, a.z, 0.0f) {}

Vec4F::Vec4F(const Vec3F& a, float b) : romanorender::Vec4F(a.x, a.y, a.z, b) {}

tinybvh::BVHContext& get_context() noexcept
{
    static tinybvh::BVHContext ctx;
    ctx.malloc = [](std::size_t size, void* = nullptr) -> void* { return stdromano::mem_aligned_alloc(size, 64); };
    ctx.free = [](void* ptr, void* = nullptr) -> void { stdromano::mem_aligned_free(ptr); };

    return ctx;
}

}
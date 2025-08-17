#include "romanorender/vec2.h"
#include "romanorender/vec3.h"
#include "romanorender/vec4.h"

namespace tbvh {

struct Vec3F;

struct Vec4F : public romanorender::Vec4F
{
    Vec4F() = default;

    Vec4F(const float a, const float b, const float c, const float d) : romanorender::Vec4F(a, b, c, d) {}

    Vec4F(const float a) : romanorender::Vec4F(a) {}

    Vec4F(const Vec3F& a);

    Vec4F(const Vec3F& a, float b);

    Vec4F(const romanorender::Vec4F& v) : romanorender::Vec4F(v) {}
};

struct Vec2F : public romanorender::Vec2F
{
    Vec2F() = default;

    Vec2F(const float a, const float b) : romanorender::Vec2F(a, b) {}

    Vec2F(const float a) : romanorender::Vec2F(a) {}
};

struct Vec3F : public romanorender::Vec3F
{
    Vec3F() = default;

    Vec3F(const float a, const float b, const float c) : romanorender::Vec3F(a, b, c) {}

    Vec3F(const float a) : romanorender::Vec3F(a) {}

    Vec3F(const Vec4F& a) : romanorender::Vec3F(a.x, a.y, a.z) {}

    Vec3F(const romanorender::Vec3F& v) : romanorender::Vec3F(v) {}
};

struct Vec3I
{
    Vec3I() = default;

    Vec3I(const int32_t a, const int32_t b, const int32_t c) : x(a), y(b), z(c) {}

    Vec3I(const int32_t a) : x(a), y(a), z(a) {}

    Vec3I(const Vec3F& a) { x = (int32_t)a.x, y = (int32_t)a.y, z = (int32_t)a.z; }

    int32_t& operator[](const int32_t i) { return (&this->x)[i]; }

    int32_t x, y, z;
};

struct Vec2I
{
    Vec2I() = default;

    Vec2I(const int32_t a, const int32_t b) : x(a), y(b) {}

    Vec2I(const int32_t a) : x(a), y(a) {}

    int32_t x, y;
};

struct Vec2U
{
    Vec2U() = default;

    Vec2U(const uint32_t a, const uint32_t b) : x(a), y(b) {}

    Vec2U(const uint32_t a) : x(a), y(a) {}

    uint32_t x, y;
};

ROMANORENDER_FORCE_INLINE Vec2F operator-(const Vec2F& a) { return Vec2F(-a.x, -a.y); }

ROMANORENDER_FORCE_INLINE Vec3F operator-(const Vec3F& a) { return Vec3F(-a.x, -a.y, -a.z); }

ROMANORENDER_FORCE_INLINE Vec4F operator-(const Vec4F& a) { return Vec4F(-a.x, -a.y, -a.z, -a.w); }

ROMANORENDER_FORCE_INLINE Vec2F operator+(const Vec2F& a, const Vec2F& b)
{
    return Vec2F(a.x + b.x, a.y + b.y);
}

ROMANORENDER_FORCE_INLINE Vec3F operator+(const Vec3F& a, const Vec3F& b)
{
    return Vec3F(a.x + b.x, a.y + b.y, a.z + b.z);
}

ROMANORENDER_FORCE_INLINE Vec4F operator+(const Vec4F& a, const Vec4F& b)
{
    return Vec4F(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

ROMANORENDER_FORCE_INLINE Vec4F operator+(const Vec4F& a, const Vec3F& b)
{
    return Vec4F(a.x + b.x, a.y + b.y, a.z + b.z, a.w);
}

ROMANORENDER_FORCE_INLINE Vec2F operator-(const Vec2F& a, const Vec2F& b)
{
    return Vec2F(a.x - b.x, a.y - b.y);
}

ROMANORENDER_FORCE_INLINE Vec3F operator-(const Vec3F& a, const Vec3F& b)
{
    return Vec3F(a.x - b.x, a.y - b.y, a.z - b.z);
}

ROMANORENDER_FORCE_INLINE Vec4F operator-(const Vec4F& a, const Vec4F& b)
{
    return Vec4F(a.x - b.x, a.y - b.y, a.z - b.z, a.w - b.w);
}

ROMANORENDER_FORCE_INLINE void operator+=(Vec2F& a, const Vec2F& b)
{
    a.x += b.x;
    a.y += b.y;
}

ROMANORENDER_FORCE_INLINE void operator+=(Vec3F& a, const Vec3F& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
}

ROMANORENDER_FORCE_INLINE void operator+=(Vec4F& a, const Vec4F& b)
{
    a.x += b.x;
    a.y += b.y;
    a.z += b.z;
    a.w += b.w;
}

ROMANORENDER_FORCE_INLINE Vec2F operator*(const Vec2F& a, const Vec2F& b)
{
    return Vec2F(a.x * b.x, a.y * b.y);
}

ROMANORENDER_FORCE_INLINE Vec3F operator*(const Vec3F& a, const Vec3F& b)
{
    return Vec3F(a.x * b.x, a.y * b.y, a.z * b.z);
}

ROMANORENDER_FORCE_INLINE Vec4F operator*(const Vec4F& a, const Vec4F& b)
{
    return Vec4F(a.x * b.x, a.y * b.y, a.z * b.z, a.w * b.w);
}

ROMANORENDER_FORCE_INLINE Vec2F operator*(const Vec2F& a, float b)
{
    return Vec2F(a.x * b, a.y * b);
}

ROMANORENDER_FORCE_INLINE Vec3F operator*(const Vec3F& a, float b)
{
    return Vec3F(a.x * b, a.y * b, a.z * b);
}

ROMANORENDER_FORCE_INLINE Vec4F operator*(const Vec4F& a, float b)
{
    return Vec4F(a.x * b, a.y * b, a.z * b, a.w * b);
}

ROMANORENDER_FORCE_INLINE Vec2F operator*(float b, const Vec2F& a)
{
    return Vec2F(b * a.x, b * a.y);
}

ROMANORENDER_FORCE_INLINE Vec3F operator*(float b, const Vec3F& a)
{
    return Vec3F(b * a.x, b * a.y, b * a.z);
}

ROMANORENDER_FORCE_INLINE Vec4F operator*(float b, const Vec4F& a)
{
    return Vec4F(b * a.x, b * a.y, b * a.z, b * a.w);
}

ROMANORENDER_FORCE_INLINE Vec2F operator/(float b, const Vec2F& a)
{
    return Vec2F(b / a.x, b / a.y);
}

ROMANORENDER_FORCE_INLINE Vec3F operator/(float b, const Vec3F& a)
{
    return Vec3F(b / a.x, b / a.y, b / a.z);
}

ROMANORENDER_FORCE_INLINE Vec4F operator/(float b, const Vec4F& a)
{
    return Vec4F(b / a.x, b / a.y, b / a.z, b / a.w);
}

ROMANORENDER_FORCE_INLINE void operator*=(Vec3F& a, const float b)
{
    a.x *= b;
    a.y *= b;
    a.z *= b;
}

struct bvhmat4
{
	bvhmat4() = default;
	float& operator [] ( const int32_t i ) { return cell[i]; }
	const float& operator [] ( const int32_t i ) const { return cell[i]; }
	bvhmat4& operator += ( const bvhmat4& a ) { for (int i = 0; i < 16; i++) cell[i] += a.cell[i]; return *this; }
	float cell[16] = { 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f };
};

} /* end namespace tbvh */

template <>
struct fmt::formatter<tbvh::Vec3F>
{
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    auto format(tbvh::Vec3F& v, format_context& ctx) const
    {
        return format_to(ctx.out(), "{}, {}, {}", v.x, v.y, v.z);
    }

    auto format(const tbvh::Vec3F& v, format_context& ctx) const
    {
        return format_to(ctx.out(), "{}, {}, {}", v.x, v.y, v.z);
    }
};

namespace tinybvh
{
    using bvhint2 = tbvh::Vec2I;
    using bvhint3 = tbvh::Vec3I;
    using bvhuint2 = tbvh::Vec2U;
    using bvhvec2 = tbvh::Vec2F;
    using bvhvec3 = tbvh::Vec3F;
    using bvhvec4 = tbvh::Vec4F;
    using bvhmat4 = tbvh::bvhmat4;
}

#define NO_DOUBLE_PRECISION_SUPPORT
#define TINYBVH_USE_CUSTOM_VECTOR_TYPES
#define TINYBVH_USE_COPY_MOVE_SEMANTICS
#define BVH8_MASSIVE_LEAFS
#define PARANOID
#include "tiny_bvh.h"

namespace tbvh {

ROMANORENDER_API tinybvh::BVHContext& get_context() noexcept;

} /* end namespace tbvh */

#pragma once

#if !defined(__ROMANORENDER_MAT44)
#define __ROMANORENDER_MAT44

#include "romanorender/vec3.h"

#include "Imath/ImathMatrix.h"

#include <algorithm>

ROMANORENDER_NAMESPACE_BEGIN

enum Mat44FTransformOrder_ : uint32_t
{
    Mat44FTransformOrder_SRT,
    Mat44FTransformOrder_STR,
    Mat44FTransformOrder_RST,
    Mat44FTransformOrder_RTS,
    Mat44FTransformOrder_TSR,
    Mat44FTransformOrder_TRS,
};

enum Mat44FRotationOrder_ : uint32_t
{
    Mat44FRotationOrder_XYZ,
    Mat44FRotationOrder_XZY,
    Mat44FRotationOrder_YXZ,
    Mat44FRotationOrder_YZX,
    Mat44FRotationOrder_ZXY,
    Mat44FRotationOrder_ZYX,
};

/* Stored in row major */
class ROMANORENDER_API Mat44F
{
private:
    alignas(32) float _data[16];

public:
    Mat44F()
        : _data{1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f}
    {
    }

    Mat44F(float a, float b, float c, float d, float e, float f, float g, float h, float i, float j, float k, float l, float m, float n, float o, float p)
        : _data{a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p} {};

    Mat44F(const Imath_3_1::M44d& m) : _data{static_cast<float>(m[0][0]),
                                             static_cast<float>(m[1][0]),
                                             static_cast<float>(m[2][0]),
                                             static_cast<float>(m[3][0]),
                                             static_cast<float>(m[0][1]),
                                             static_cast<float>(m[1][1]),
                                             static_cast<float>(m[2][1]),
                                             static_cast<float>(m[3][1]),
                                             static_cast<float>(m[0][2]),
                                             static_cast<float>(m[1][2]),
                                             static_cast<float>(m[2][2]),
                                             static_cast<float>(m[3][2]),
                                             static_cast<float>(m[0][3]),
                                             static_cast<float>(m[1][3]),
                                             static_cast<float>(m[2][3]),
                                             static_cast<float>(m[3][3])} {}

    Mat44F(const Imath_3_1::M44f& m) : _data{m[0][0],
                                             m[1][0],
                                             m[2][0],
                                             m[3][0],
                                             m[0][1],
                                             m[1][1],
                                             m[2][1],
                                             m[3][1],
                                             m[0][2],
                                             m[1][2],
                                             m[2][2],
                                             m[3][2],
                                             m[0][3],
                                             m[1][3],
                                             m[2][3],
                                             m[3][3]} {}

    static Mat44F from_lookat(const Vec3F& position, const Vec3F& lookat) noexcept;

    static Mat44F from_trs(const Vec3F& t,
                           const Vec3F& r,
                           const Vec3F& s,
                           const Mat44FTransformOrder_ to = Mat44FTransformOrder_TRS,
                           const Mat44FRotationOrder_ ro = Mat44FRotationOrder_XYZ) noexcept;

    static Mat44F from_axis_angle(const Vec3F& axis, const float angle) noexcept;

    static Mat44F from_xyzt(const Vec3F& x, const Vec3F& y, const Vec3F& z, const Vec3F& t);

    const float& operator[](uint32_t i) const { return this->_data[i]; }

    float& operator[](uint32_t i) { return this->_data[i]; }

    const float& operator()(uint32_t i, uint32_t j) const { return this->_data[i * 4 + j]; }

    float& operator()(uint32_t i, uint32_t j) { return this->_data[i * 4 + j]; }

    const float* data() const noexcept { return std::addressof(this->_data[0]); }

    ROMANORENDER_FORCE_INLINE void transpose() noexcept
    {
        std::swap(this->_data[1], this->_data[4]);
        std::swap(this->_data[2], this->_data[8]);
        std::swap(this->_data[3], this->_data[12]);

        std::swap(this->_data[6], this->_data[9]);
        std::swap(this->_data[7], this->_data[13]);

        std::swap(this->_data[11], this->_data[14]);
    }

    ROMANORENDER_FORCE_INLINE Mat44F transposed() const noexcept
    {
        Mat44F res = *this;
        res.transpose();
        return res;
    }

    void decompose_xyz(Vec3F* x, Vec3F* y, Vec3F* z) const noexcept;

    void decompose_trs(Vec3F* t, Vec3F* r, Vec3F* s) const noexcept;

    void zero_translation() noexcept;

    Vec3F get_translation() const noexcept;

    void set_translation(const Vec3F& t) noexcept;

    void debug() const noexcept;
};

ROMANORENDER_API Mat44F mat44f_mul(const Mat44F& A, const Mat44F& B) noexcept;

ROMANORENDER_API Vec3F mat44f_mul_point(const float* M, const Vec3F& v) noexcept;

ROMANORENDER_API Vec3F mat44f_mul_point(const Mat44F& M, const Vec3F& v) noexcept;

ROMANORENDER_API Vec3F mat44f_mul_dir(const float* M, const Vec3F& v) noexcept;

ROMANORENDER_API Vec3F mat44f_mul_dir(const Mat44F& M, const Vec3F& v) noexcept;

ROMANORENDER_NAMESPACE_END

template <>
struct fmt::formatter<romanorender::Mat44F>
{
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    auto format(romanorender::Mat44F& m, format_context& ctx) const
    {
        return format_to(ctx.out(),
                         "{}, {}, {}, {}\n{}, {}, {}, {}\n{}, {}, {}, {}\n{}, {}, {}, {}",
                         m[0],
                         m[1],
                         m[2],
                         m[3],
                         m[4],
                         m[5],
                         m[6],
                         m[7],
                         m[8],
                         m[9],
                         m[10],
                         m[11],
                         m[12],
                         m[13],
                         m[14],
                         m[15]);
    }

    auto format(const romanorender::Mat44F& m, format_context& ctx) const
    {
        return format_to(ctx.out(),
                         "{}, {}, {}, {}\n{}, {}, {}, {}\n{}, {}, {}, {}\n{}, {}, {}, {}",
                         m[0],
                         m[1],
                         m[2],
                         m[3],
                         m[4],
                         m[5],
                         m[6],
                         m[7],
                         m[8],
                         m[9],
                         m[10],
                         m[11],
                         m[12],
                         m[13],
                         m[14],
                         m[15]);
    }
};

#endif /* !defined(__ROMANORENDER_MAT44)*/
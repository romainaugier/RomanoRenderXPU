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

/* Column Major */
class ROMANORENDER_API Mat44F
{
private:
    float m[4][4];

public:
    Mat44F() {}

    Mat44F(float m00,
           float m10,
           float m20,
           float m30,
           float m01,
           float m11,
           float m21,
           float m31,
           float m02,
           float m12,
           float m22,
           float m32,
           float m03,
           float m13,
           float m23,
           float m33)
    {
        this->m[0][0] = m00;
        this->m[0][1] = m10;
        this->m[0][2] = m20;
        this->m[0][3] = m30;
        this->m[1][0] = m01;
        this->m[1][1] = m11;
        this->m[1][2] = m21;
        this->m[1][3] = m31;
        this->m[2][0] = m02;
        this->m[2][1] = m12;
        this->m[2][2] = m22;
        this->m[2][3] = m32;
        this->m[3][0] = m03;
        this->m[3][1] = m13;
        this->m[3][2] = m23;
        this->m[3][3] = m33;
    }

    Mat44F(const Imath_3_1::M44d& _m) { for(uint32_t i = 0; i < 16; i++) { this->data()[i] = (float)_m.getValue()[i]; } }

    Mat44F(const Imath_3_1::M44f& _m) { std::memcpy(this->data(), _m.getValue(), 16 * sizeof(float)); }

    /* Static constructors */

    static Mat44F zeros() noexcept;

    static Mat44F identity() noexcept;

    static Mat44F from_translation(const Vec3F& t) noexcept;

    static Mat44F from_scale(const Vec3F& s) noexcept;

    /* rx is in degrees */
    static Mat44F from_rotx(const float rx) noexcept;

    /* ry is in degrees */
    static Mat44F from_roty(const float ry) noexcept;

    /* rz is in degrees */
    static Mat44F from_rotz(const float rz) noexcept;

    /* Rotation angles is in degrees */
    static Mat44F from_axis_angle(const Vec3F& axis, const float angle) noexcept;

    /* Rotation angles are in degress */
    static Mat44F from_trs(const Vec3F& translation,
                           const Vec3F& rotation,
                           const Vec3F& scale,
                           const Mat44FTransformOrder_ to = Mat44FTransformOrder_TRS,
                           const Mat44FRotationOrder_ ro = Mat44FRotationOrder_XYZ) noexcept;

    static Mat44F from_xyzt(const Vec3F& x, const Vec3F& y, const Vec3F& z, const Vec3F& t) noexcept;

    static Mat44F from_lookat(const Vec3F& eye,
                              const Vec3F& target,
                              const Vec3F& up = Vec3F(0.0f, 1.0f, 0.0f)) noexcept;

    /* Access operators */

    ROMANORENDER_FORCE_INLINE float& operator()(const uint32_t row, const uint32_t col) noexcept
    {
        return this->m[col][row];
    }

    ROMANORENDER_FORCE_INLINE const float& operator()(const uint32_t row, const uint32_t col) const noexcept
    {
        return this->m[col][row];
    }

    ROMANORENDER_FORCE_INLINE float& at(int row, int col) noexcept { return this->m[col][row]; }

    ROMANORENDER_FORCE_INLINE const float& at(int row, int col) const noexcept
    {
        return this->m[col][row];
    }

    /* Data */

    ROMANORENDER_FORCE_INLINE const float* data() const noexcept { return &this->m[0][0]; }

    ROMANORENDER_FORCE_INLINE float* data() noexcept { return &this->m[0][0]; }

    /* Transposition */

    ROMANORENDER_FORCE_INLINE Mat44F transpose() const noexcept
    {
        return Mat44F(this->m[0][0],
                      this->m[1][0],
                      this->m[2][0],
                      this->m[3][0],
                      this->m[0][1],
                      this->m[1][1],
                      this->m[2][1],
                      this->m[3][1],
                      this->m[0][2],
                      this->m[1][2],
                      this->m[2][2],
                      this->m[3][2],
                      this->m[0][3],
                      this->m[1][3],
                      this->m[2][3],
                      this->m[3][3]);
    }

    /* Mat Mat mul */

    Mat44F operator*(const Mat44F& other) const noexcept;

    /* Mat Vec mul */

    Vec3F transform_point(const Vec3F& point) const noexcept;

    Vec3F transform_dir(const Vec3F& point) const noexcept;

    /* Decomposition */

    void decomp_translation(Vec3F* t) const noexcept;

    void decomp_scale(Vec3F* s) const noexcept;

    void decomp_xyzt(Vec3F* x, Vec3F* y, Vec3F* z, Vec3F* t) const noexcept;

    /* Angles will be in degrees */
    void decomp_euler(Vec3F* angles) const noexcept;

    void decomp_trs(Vec3F* t, Vec3F* r, Vec3F* s) const noexcept;
};

ROMANORENDER_FORCE_INLINE Vec3F mat44_rowmajor_vec_mul_dir(const float* M, const Vec3F& v)
{
    return Vec3F(v.x * M[0] + v.y * M[1] + v.z * M[2],
                 v.x * M[4] + v.y * M[5] + v.z * M[6],
                 v.x * M[8] + v.y * M[9] + v.z * M[10]);
}

ROMANORENDER_NAMESPACE_END

template <>
struct fmt::formatter<romanorender::Mat44F>
{
    constexpr auto parse(format_parse_context& ctx) { return ctx.begin(); }

    auto format(romanorender::Mat44F& m, format_context& ctx) const
    {
        return format_to(ctx.out(),
                         "{}, {}, {}, {}\n{}, {}, {}, {}\n{}, {}, {}, {}\n{}, {}, {}, {}",
                         m(0, 0),
                         m(0, 1),
                         m(0, 2),
                         m(0, 3),
                         m(1, 0),
                         m(1, 1),
                         m(1, 2),
                         m(1, 3),
                         m(2, 0),
                         m(2, 1),
                         m(2, 2),
                         m(2, 3),
                         m(3, 0),
                         m(3, 1),
                         m(3, 2),
                         m(3, 3));
    }

    auto format(const romanorender::Mat44F& m, format_context& ctx) const
    {
        return format_to(ctx.out(),
                         "{}, {}, {}, {}\n{}, {}, {}, {}\n{}, {}, {}, {}\n{}, {}, {}, {}",
                         m(0, 0),
                         m(0, 1),
                         m(0, 2),
                         m(0, 3),
                         m(1, 0),
                         m(1, 1),
                         m(1, 2),
                         m(1, 3),
                         m(2, 0),
                         m(2, 1),
                         m(2, 2),
                         m(2, 3),
                         m(3, 0),
                         m(3, 1),
                         m(3, 2),
                         m(3, 3));
    }
};

#endif /* !defined(__ROMANORENDER_MAT44)*/
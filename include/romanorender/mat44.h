#pragma once

#if !defined(__ROMANORENDER_MAT44)
#define __ROMANORENDER_MAT44

#include "romanorender/vec3.h"

#include <algorithm>

ROMANORENDER_NAMESPACE_BEGIN

/* Stored in row major */
class ROMANORENDER_API Mat44F
{
private:
    alignas(32) float _data[16];

public:
    Mat44F() : _data{ 1.0f, 0.0f, 0.0f, 0.0f,
                      0.0f, 1.0f, 0.0f, 0.0f,
                      0.0f, 0.0f, 1.0f, 0.0f,
                      0.0f, 0.0f, 0.0f, 1.0f } {}

    Mat44F(float a, float b, float c, float d,
           float e, float f, float g, float h,
           float i, float j, float k, float l,
           float m, float n, float o, float p)
           : _data{ a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p } {};

    static Mat44F lookat(const Vec3F& position, const Vec3F& lookat) noexcept;

    const float& operator[](uint32_t i) const { return this->_data[i]; }
    float& operator [](uint32_t i) { return this->_data[i]; }

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

    void debug() const noexcept;
};

ROMANORENDER_API Mat44F mat44f_mul(const Mat44F& A, const Mat44F& B) noexcept;

ROMANORENDER_API Vec3F mat44f_mul(const Mat44F& M, const Vec3F& v) noexcept;

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_MAT44)*/
#pragma once

#include "float3.cuh"
#include "float4.cuh"

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

class Mat44F
{
private:
    union
    {
        float m[4][4];
        float4 columns[4];
    };

public:
    __device__ Mat44F() { *this = Mat44F::identity(); }

    __device__ Mat44F(const float* data) { memcpy(this->data(), data, 16 * sizeof(float)); }

    __device__ Mat44F(float m00,
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

    /* Static constructors */

    __device__ static Mat44F zeros()
    {
        Mat44F res;

        for(uint i = 0; i < 4; ++i)
        {
            res.columns[i] = make_float4(0.0f);
        }

        return res;
    }

    __device__ static Mat44F identity()
    {
        Mat44F res = zeros();
        res.m[0][0] = res.m[1][1] = res.m[2][2] = res.m[3][3] = 1.0f;
        return res;
    }

    __device__ static Mat44F from_translation(const float3& t)
    {
        return Mat44F(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, t.x, t.y, t.z, 1.0f);
    }

    __device__ static Mat44F from_scale(const float3& s)
    {
        return Mat44F(s.x, 0.0f, 0.0f, 0.0f, 0.0f, s.y, 0.0f, 0.0f, 0.0f, 0.0f, s.z, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
    }

    __device__ static Mat44F from_rotx(float rx)
    {
        const float rad = deg2radf(rx);
        float c, s;
        __sincosf(rad, &c, &s);

        return Mat44F(1.0f, 0.0f, 0.0f, 0.0f, 0.0f, c, s, 0.0f, 0.0f, -s, c, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
    }

    __device__ static Mat44F from_roty(float ry)
    {
        const float rad = deg2radf(ry);
        float c, s;
        __sincosf(rad, &c, &s);

        return Mat44F(c, 0.0f, -s, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, s, 0.0f, c, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
    }

    __device__ static Mat44F from_rotz(float rz)
    {
        const float rad = deg2radf(rz);
        float c, s;
        __sincosf(rad, &c, &s);

        return Mat44F(c, s, 0.0f, 0.0f, -s, c, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f);
    }

    __device__ static Mat44F from_axis_angle(const float3& axis, float angle)
    {
        const float3 n = normalize_float3(axis);

        const float rad = deg2radf(angle);
        float c, s;
        __sincosf(rad, &c, &s);

        const float t = 1.0f - c;
        const float x = n.x;
        const float y = n.y;
        const float z = n.z;

        return Mat44F(t * x * x + c,
                      t * x * y + s * z,
                      t * x * z - s * y,
                      0.0f,
                      t * x * y - s * z,
                      t * y * y + c,
                      t * y * z + s * x,
                      0.0f,
                      t * x * z + s * y,
                      t * y * z - s * x,
                      t * z * z + c,
                      0.0f,
                      0.0f,
                      0.0f,
                      0.0f,
                      1.0f);
    }

    __device__ static Mat44F from_trs(const float3& translation,
                                      const float3& rotation,
                                      const float3& scale,
                                      const Mat44FTransformOrder_ to = Mat44FTransformOrder_TRS,
                                      const Mat44FRotationOrder_ ro = Mat44FRotationOrder_XYZ) noexcept
    {
        const Mat44F T = Mat44F::from_translation(translation);
        const Mat44F S = Mat44F::from_scale(scale);

        const Mat44F RX = Mat44F::from_rotx(rotation.x);
        const Mat44F RY = Mat44F::from_roty(rotation.y);
        const Mat44F RZ = Mat44F::from_rotz(rotation.z);

        Mat44F R;

        switch(ro)
        {
        case Mat44FRotationOrder_XYZ:
            R = RZ * RY * RX;
            break;
        case Mat44FRotationOrder_XZY:
            R = RY * RZ * RX;
            break;
        case Mat44FRotationOrder_YXZ:
            R = RZ * RX * RY;
            break;
        case Mat44FRotationOrder_YZX:
            R = RX * RZ * RY;
            break;
        case Mat44FRotationOrder_ZXY:
            R = RY * RX * RZ;
            break;
        case Mat44FRotationOrder_ZYX:
            R = RX * RY * RZ;
            break;
        default:
            R = RZ * RY * RX;
        }

        switch(to)
        {
        case Mat44FTransformOrder_SRT:
            return R * S * T;
        case Mat44FTransformOrder_STR:
            return R * T * S;
        case Mat44FTransformOrder_RST:
            return S * T * R;
        case Mat44FTransformOrder_RTS:
            return S * R * T;
        case Mat44FTransformOrder_TSR:
            return R * S * T;
        case Mat44FTransformOrder_TRS:
            return T * R * S;
        default:
            return T * R * S;
        }
    }

    __device__ static Mat44F from_xyzt(const float3& x, const float3& y, const float3& z, const float3& t) noexcept
    {
        return Mat44F(x.x, y.x, z.x, 0.0f, x.y, y.y, z.y, 0.0f, x.z, y.z, z.z, 0.0f, t.x, t.y, t.z, 1.0f);
    }

    __device__ static Mat44F from_lookat(const float3& eye, const float3& target, const float3& up) noexcept
    {
        const float3 z = normalize_float3(make_float3(eye.x - target.x,
                                                      eye.y - target.y,
                                                      eye.z - target.z));

        const float3 x = normalize_float3(cross_float3(up, z));
        const float3 y = cross_float3(z, x);

        return Mat44F(x.x,
                      y.x,
                      z.x,
                      0.0f,
                      x.y,
                      y.y,
                      z.y,
                      0.0f,
                      x.z,
                      y.z,
                      z.z,
                      0.0f,
                      -dot_float3(x, eye),
                      -dot_float3(y, eye),
                      -dot_float3(z, eye),
                      1.0f);
    }

    /* Access operators */

    __forceinline__ __device__ float& operator()(const uint32_t row, const uint32_t col) noexcept
    {
        return this->m[col][row];
    }

    __forceinline__ __device__ const float& operator()(const uint32_t row, const uint32_t col) const noexcept
    {
        return this->m[col][row];
    }

    __forceinline__ __device__ float& at(int row, int col) noexcept { return this->m[col][row]; }

    __forceinline__ __device__ const float& at(int row, int col) const noexcept
    {
        return this->m[col][row];
    }

    __forceinline__ __device__ float* data() noexcept { return &this->m[0][0]; }

    __forceinline__ __device__ const float* data() const noexcept { return &this->m[0][0]; }

    /* Transposition */

    __inline__ __device__ Mat44F transpose() const noexcept
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

    __device__ Mat44F operator*(const Mat44F& other) const noexcept
    {
        Mat44F res = zeros();

        for(uint i = 0; i < 4; ++i)
        {
            for(uint j = 0; j < 4; ++j)
            {
                const float4 other_col = other.columns[j];
                float sum = 0.0f;

                sum = __fmaf_rn(this->m[0][i], other_col.x, sum);
                sum = __fmaf_rn(this->m[1][i], other_col.y, sum);
                sum = __fmaf_rn(this->m[2][i], other_col.z, sum);
                sum = __fmaf_rn(this->m[3][i], other_col.w, sum);

                res.m[j][i] = sum;
            }
        }

        return res;
    }

    /* Mat Vec mul */

    __device__ float3 transform_point(const float3& point) const noexcept
    {
        float3 res;

        res.x = this->m[0][0] * point.x + this->m[1][0] * point.y + this->m[2][0] * point.z + this->m[3][0];
        res.y = this->m[0][1] * point.x + this->m[1][1] * point.y + this->m[2][1] * point.z + this->m[3][1];
        res.z = this->m[0][2] * point.x + this->m[1][2] * point.y + this->m[2][2] * point.z + this->m[3][2];

        return res;
    }

    __device__ float3 transform_dir(const float3& point) const noexcept
    {
        float3 res;

        res.x = this->m[0][0] * point.x + this->m[1][0] * point.y + this->m[2][0] * point.z;
        res.y = this->m[0][1] * point.x + this->m[1][1] * point.y + this->m[2][1] * point.z;
        res.z = this->m[0][2] * point.x + this->m[1][2] * point.y + this->m[2][2] * point.z;

        return res;
    }

    /* Decomposition */

    __device__ void decomp_translation(float3* t) const noexcept
    {
        *t = make_float3(this->m[3][0], this->m[3][1], this->m[3][2]);
    }

    __device__ void decomp_scale(float3* s) const noexcept
    {
        s->x = __fsqrt_rn(this->m[0][0] * this->m[0][0] + this->m[0][1] * this->m[0][1]
                          + this->m[0][2] * this->m[0][2]);
        s->y = __fsqrt_rn(this->m[1][0] * this->m[1][0] + this->m[1][1] * this->m[1][1]
                          + this->m[1][2] * this->m[1][2]);
        s->z = __fsqrt_rn(this->m[2][0] * this->m[2][0] + this->m[2][1] * this->m[2][1]
                          + this->m[2][2] * this->m[2][2]);
    }

    __device__ void decomp_xyzt(float3* x, float3* y, float3* z, float3* t) const noexcept
    {
        float3 scale;
        this->decomp_scale(&scale);

        *x = make_float3(this->m[0][0], this->m[0][1], this->m[0][2]);
        *y = make_float3(this->m[1][0], this->m[1][1], this->m[1][2]);
        *z = make_float3(this->m[2][0], this->m[2][1], this->m[2][2]);

        if(scale.x > 1e-6f)
        {
            x->x /= scale.x;
            x->y /= scale.x;
            x->z /= scale.x;
        }

        if(scale.y > 1e-6f)
        {
            y->x /= scale.y;
            y->y /= scale.y;
            y->z /= scale.y;
        }

        if(scale.z > 1e-6f)
        {
            z->x /= scale.z;
            z->y /= scale.z;
            z->z /= scale.z;
        }

        if(t != nullptr)
        {
            this->decomp_translation(t);
        }
    }

    __device__ void decomp_euler(float3* angles) const noexcept
    {
        angles->x = deg2radf(asin(-this->m[2][1]));

        if(__cosf(angles->x) > 1e-6f)
        {
            angles->y = deg2radf(atan2(this->m[2][0], this->m[2][2]));
            angles->z = deg2radf(atan2(this->m[0][1], this->m[1][1]));
        }
        else
        {
            angles->y = deg2radf(atan2(-this->m[0][2], this->m[0][0]));
            angles->z = 0.0f;
        }
    }

    __device__ void decomp_trs(float3* t, float3* r, float3* s) const noexcept
    {
        if(t != nullptr)
        {
            this->decomp_translation(t);
        }

        if(s != nullptr)
        {
            this->decomp_scale(s);
        }

        Mat44F rotation_matrix = *this;

        for(int i = 0; i < 3; i++)
        {
            float inv_scale = s->x > 1e-6f ? __frcp_rn(s->x) : 0.0f;
            rotation_matrix.m[0][i] *= inv_scale;

            inv_scale = s->y > 1e-6f ? __frcp_rn(s->y) : 0.0f;
            rotation_matrix.m[1][i] *= inv_scale;

            inv_scale = s->z > 1e-6f ? __frcp_rn(s->z) : 0.0f;
            rotation_matrix.m[2][i] *= inv_scale;
        }

        rotation_matrix.decomp_euler(r);
    }
};
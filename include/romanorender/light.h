#pragma once

#if !defined(__ROMANORENDER_LIGHT)
#define __ROMANORENDER_LIGHT

#include "romanorender/vec2.h"
#include "romanorender/mat44.h"
#include "romanorender/light_type.h"

ROMANORENDER_NAMESPACE_BEGIN

class ROMANORENDER_API LightBase
{
protected:
    LightType_ _type;

    uint32_t _id;

    Mat44F _transform;
    Mat44FTransformOrder_ _transform_order;
    
    float _intensity;
    Vec3F _color;

public:
    LightBase(const uint32_t id) : _id(id), _transform(Mat44F::identity()), _intensity(1.0f), _color(1.0f) {}

    virtual ~LightBase() = default;

    virtual Vec3F sample_direction(const Vec3F& hit_position, const Vec2F& sample, const Vec3F& hit_normal = Vec3F(0.0f)) const noexcept = 0;

    virtual Vec3F sample_intensity(const float distance) const noexcept = 0;

    ROMANORENDER_FORCE_INLINE LightType_ get_type() const noexcept { return this->_type; }

    ROMANORENDER_FORCE_INLINE uint32_t get_id() const noexcept { return this->_id; }

    ROMANORENDER_FORCE_INLINE void set_type(const LightType_ type) { this->_type = type; }

    ROMANORENDER_FORCE_INLINE Mat44F get_transform() const noexcept { return this->_transform; }

    ROMANORENDER_FORCE_INLINE void set_transform(const Mat44F transform) noexcept { this->_transform = transform; }

    ROMANORENDER_FORCE_INLINE Mat44FTransformOrder_ get_transform_order() const noexcept { return this->_transform_order; }

    ROMANORENDER_FORCE_INLINE void set_transform_order(const Mat44FTransformOrder_ transform_order) noexcept { this->_transform_order = transform_order; }

    ROMANORENDER_FORCE_INLINE float get_intensity() const noexcept { return this->_intensity; }

    ROMANORENDER_FORCE_INLINE void set_intensity(const float intensity) noexcept { this->_intensity = intensity; }

    ROMANORENDER_FORCE_INLINE Vec3F get_color() const noexcept { return this->_color; }

    ROMANORENDER_FORCE_INLINE void set_color(const Vec3F& color) noexcept { this->_color = color; }
};

class ROMANORENDER_API LightSquare : public LightBase
{
    float _size_x = 1.0f;
    float _size_y = 1.0f;

public:
    LightSquare(const uint32_t id) : LightBase(id)
    {
        this->set_type(LightType_Square);
    }

    virtual ~LightSquare() override {}

    virtual Vec3F sample_direction(const Vec3F& hit_position, const Vec2F& sample, const Vec3F& hit_normal = Vec3F(0.0f)) const noexcept override;

    virtual Vec3F sample_intensity(const float distance) const noexcept override;

    ROMANORENDER_FORCE_INLINE void set_size(const float size_x, const float size_y) noexcept { this->_size_x = size_x; this->_size_y = size_y; }
};

class ROMANORENDER_API LightDome : public LightBase
{
public:
    LightDome(const uint32_t id) : LightBase(id)
    {
        this->set_type(LightType_Dome);
    }

    virtual ~LightDome() override {}

    virtual Vec3F sample_direction(const Vec3F& hit_position, const Vec2F& sample, const Vec3F& hit_normal = Vec3F(0.0f)) const noexcept override;

    virtual Vec3F sample_intensity(const float distance) const noexcept override;
};

class ROMANORENDER_API LightDistant : public LightBase
{
    Vec3F _orientation = Vec3F(0.0f, -1.0f, 1.0f);
    float _angle = 1.0f;

public:
    LightDistant(const uint32_t id) : LightBase(id)
    {
        this->set_type(LightType_Distant);
    }

    virtual ~LightDistant() override {}

    virtual Vec3F sample_direction(const Vec3F& hit_position, const Vec2F& sample, const Vec3F& hit_normal = Vec3F(0.0f)) const noexcept override;

    virtual Vec3F sample_intensity(const float distance) const noexcept override;

    ROMANORENDER_FORCE_INLINE void set_orientation(const Vec3F& orientation) noexcept { this->_orientation = orientation; }

    ROMANORENDER_FORCE_INLINE void set_angle(const float angle) noexcept { this->_angle = angle; }
};

class ROMANORENDER_API LightCircle : public LightBase
{
    float _size_x = 1.0f;
    float _size_y = 1.0f;

public:
    LightCircle(const uint32_t id) : LightBase(id)
    {
        this->set_type(LightType_Circle);
    }

    virtual ~LightCircle() override {}

    virtual Vec3F sample_direction(const Vec3F& hit_position, const Vec2F& sample, const Vec3F& hit_normal = Vec3F(0.0f)) const noexcept override;

    virtual Vec3F sample_intensity(const float distance) const noexcept override;

    ROMANORENDER_FORCE_INLINE void set_size(const float size_x, const float size_y) noexcept { this->_size_x = size_x; this->_size_y = size_y; }
};

class ROMANORENDER_API LightSpherical : public LightBase
{
    float _radius = 1.0f;

public:
    LightSpherical(const uint32_t id) : LightBase(id)
    {
        this->set_type(LightType_Spherical);
    }

    virtual ~LightSpherical() override {}

    virtual Vec3F sample_direction(const Vec3F& hit_position, const Vec2F& sample, const Vec3F& hit_normal = Vec3F(0.0f)) const noexcept override;

    virtual Vec3F sample_intensity(const float distance) const noexcept override;

    ROMANORENDER_FORCE_INLINE void set_radius(const float radius) noexcept { this->_radius = radius; }
};

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_LIGHT) */
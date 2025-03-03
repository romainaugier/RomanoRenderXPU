#pragma once

#if !defined(__ROMANORENDER_OBJECT)
#define __ROMANORENDER_OBJECT

#include "romanorender/vec4.h"
#include "romanorender/mat44.h"
#include "romanorender/tbvh.h"

#include "stdromano/vector.h"
#include "stdromano/string.h"
#include "stdromano/hashmap.h"

ROMANORENDER_NAMESPACE_BEGIN

enum AttributeBufferType_ : uint32_t
{
    AttributeBufferType_Custom = 0,
    AttributeBufferType_Normal = 1,
    AttributeBufferType_Radius = 2,
};

enum AttributeBufferFormat_ : uint32_t
{
    AttributeBufferFormat_Float1 = 0,
    AttributeBufferFormat_Float2 = 1,
    AttributeBufferFormat_Float3 = 2,
    AttributeBufferFormat_Int1 = 3,
    AttributeBufferFormat_Int2 = 4,
    AttributeBufferFormat_Int3 = 5,
    AttributeBufferFormat_UInt1 = 6,
    AttributeBufferFormat_UInt2 = 7,
    AttributeBufferFormat_UInt3 = 8,
};

class ROMANORENDER_API AttributeBuffer
{
    static constexpr uint32_t GEOM_BUFFER_ALIGNMENT = 16;

    void* data = nullptr;
    uint32_t* refcount = nullptr;

    uint32_t count = 0;
    uint32_t stride = 0;
    uint32_t type = 0;
    uint32_t format = 0;

public:
    AttributeBuffer() : data(nullptr),
                       count(0),
                       stride(0),
                       type(0),
                       format(0),
                       refcount(nullptr) {}

    AttributeBuffer(AttributeBufferType_ type,
                    AttributeBufferFormat_ format,
                    const uint32_t stride,
                    const uint32_t count);

    AttributeBuffer(const AttributeBuffer& other) noexcept;
    AttributeBuffer(AttributeBuffer&& other) noexcept;

    AttributeBuffer& operator=(const AttributeBuffer& other) noexcept;
    AttributeBuffer& operator=(AttributeBuffer&& other) noexcept;

    ~AttributeBuffer();

    void* get_geometry_ptr() const noexcept { return this->data; } 
    uint32_t get_count() const noexcept { return this->count; }
    uint32_t get_stride() const noexcept { return this->stride; }
    AttributeBufferType_ get_type() const noexcept { return (AttributeBufferType_)this->type; }
    AttributeBufferFormat_ get_format() const noexcept { return (AttributeBufferFormat_)this->format; }
};

class ROMANORENDER_API Object 
{
    stdromano::Vector<Vec4F> _vertices;
    stdromano::Vector<uint32_t> _indices;

    stdromano::HashMap<stdromano::String<>, AttributeBuffer> _attributes;

    tinybvh::BVH8_CPU _blas;

    Mat44F _transform;

    uint32_t _id;
    stdromano::String<> _name;

public:
    Object() {}

    Object(const Object& other) noexcept;
    Object(Object&& other) noexcept;

    Object& operator=(const Object& other) noexcept;
    Object& operator=(Object&& other) noexcept;

    static Object random_triangles(const uint32_t num_triangles, const float scale) noexcept;
    static Object cube(const Vec3F& center, const Vec3F& scale) noexcept;
    static Object geodesic(const Vec3F& center, const Vec3F& scale, const uint32_t subdiv_level) noexcept;
    static Object plane(const Vec3F& center, const Vec3F& scale) noexcept;

    void build_blas() noexcept;

    ROMANORENDER_FORCE_INLINE const stdromano::Vector<Vec4F>& get_vertices() const noexcept { return this->_vertices; };
    ROMANORENDER_FORCE_INLINE stdromano::Vector<Vec4F>& get_vertices() noexcept { return this->_vertices; };
    ROMANORENDER_FORCE_INLINE const stdromano::Vector<uint32_t>& get_indices() const noexcept { return this->_indices; };
    ROMANORENDER_FORCE_INLINE stdromano::Vector<uint32_t>& get_indices() noexcept { return this->_indices; };
    ROMANORENDER_FORCE_INLINE const tinybvh::BVH8_CPU& get_blas() const noexcept { return this->_blas; }
    ROMANORENDER_FORCE_INLINE const Mat44F& get_transform() const noexcept { return this->_transform; }
    ROMANORENDER_FORCE_INLINE uint32_t get_id() const noexcept { return this->_id; }
    ROMANORENDER_FORCE_INLINE const stdromano::String<>& get_name() const noexcept { return this->_name; }

    ROMANORENDER_FORCE_INLINE void set_id(uint32_t id) { this->_id = id; }
    ROMANORENDER_FORCE_INLINE void set_name(const stdromano::String<>& name) { this->_name = std::move(name); }
    ROMANORENDER_FORCE_INLINE void set_transform(const Mat44F& transform) noexcept { this->_transform = transform; }

    void add_attribute_buffer(const stdromano::String<>& name, AttributeBuffer& buffer) noexcept;
    const AttributeBuffer* get_attribute_buffer(const stdromano::String<>& name) const noexcept;
};

bool objects_from_obj_file(const char* file_path, stdromano::Vector<Object>& object) noexcept;

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_OBJECT) */
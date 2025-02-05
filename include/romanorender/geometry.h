#pragma once

#if !defined(__ROMANORENDER_GEOMETRY)
#define __ROMANORENDER_GEOMETRY

#include "romanorender/romanorender.h"

#include "stdromano/vector.h"

ROMANORENDER_NAMESPACE_BEGIN

enum GeometryType_ : uint32_t
{
    GeometryType_Triangle = 0,
    GeometryType_Point = 1,
    GeometryType_Curve = 2,
    GeometryType_Instance = 3,
};

enum GeometryBufferType_ : uint32_t
{
    GeometryBufferType_Index = 0,
    GeometryBufferType_Vertex = 1,
    GeometryBufferType_VertexAttribute = 2,
    GeometryBufferType_VertexAttributeNormal = 3,
    GeometryBufferType_VertexAttributeRadius = 4, /* Used for points/spheres */
};

enum GeometryBufferFormat_ : uint32_t
{
    GeometryBufferFormat_Float1 = 0,
    GeometryBufferFormat_Float2 = 1,
    GeometryBufferFormat_Float3 = 2,
    GeometryBufferFormat_Int1 = 3,
    GeometryBufferFormat_Int2 = 4,
    GeometryBufferFormat_Int3 = 5,
    GeometryBufferFormat_UInt1 = 6,
    GeometryBufferFormat_UInt2 = 7,
    GeometryBufferFormat_UInt3 = 8,
};

class GeometryBuffer
{
    static constexpr uint32_t GEOM_BUFFER_ALIGNMENT = 16;

    void* data = nullptr;
    uint32_t* refcount = nullptr;

    uint32_t count = 0;
    uint32_t stride = 0;
    uint32_t type = 0;
    uint32_t format = 0;

public:
    GeometryBuffer() : data(nullptr),
                       count(0),
                       stride(0),
                       type(0),
                       format(0),
                       refcount(nullptr) {}

    GeometryBuffer(GeometryBufferType_ type,
                   GeometryBufferFormat_ format,
                   const uint32_t stride,
                   const uint32_t count);

    GeometryBuffer(const GeometryBuffer& other) noexcept;
    GeometryBuffer(GeometryBuffer&& other) noexcept;

    GeometryBuffer& operator=(const GeometryBuffer& other) noexcept;
    GeometryBuffer& operator=(GeometryBuffer&& other) noexcept;

    ~GeometryBuffer();

    void* get_geometry_ptr() const noexcept { return this->data; } 
    uint32_t get_count() const noexcept { return this->count; }
    uint32_t get_stride() const noexcept { return this->stride; }
    GeometryBufferType_ get_type() const noexcept { return (GeometryBufferType_)this->type; }
    GeometryBufferFormat_ get_format() const noexcept { return (GeometryBufferFormat_)this->format; }
};

using GeometryBuffers = stdromano::Vector<GeometryBuffer>;

class Geometry
{
    GeometryBuffers geometry_buffers;
    uint32_t flags;
    uint32_t id;

public:
    Geometry(const uint32_t geometry_type) : flags(geometry_type), id(0) {}

    ROMANORENDER_FORCE_INLINE uint32_t get_geometry_type() const noexcept { return this->flags & 0x3; }
    ROMANORENDER_FORCE_INLINE uint32_t get_id() const noexcept { return this->id; }
    ROMANORENDER_FORCE_INLINE size_t get_geometry_buffers_count() const noexcept { return this->geometry_buffers.size(); }
    ROMANORENDER_FORCE_INLINE const GeometryBuffers& get_geometry_buffers() const noexcept { return this->geometry_buffers; }
    ROMANORENDER_FORCE_INLINE GeometryBuffers& get_geometry_buffers() noexcept { return this->geometry_buffers; }

    ROMANORENDER_FORCE_INLINE void set_id(const uint32_t id) noexcept { this->id = id; }

    /* Returns the pointer to the geometry buffer where you can store data */
    void* add_geometry_buffer(GeometryBufferType_ type,
                              GeometryBufferFormat_ format,
                              const uint32_t stride,
                              const uint32_t count) noexcept;
};

using Geometries = stdromano::Vector<Geometry>;

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_GEOMETRY) */
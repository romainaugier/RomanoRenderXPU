#include "romanorender/geometry.h"

ROMANORENDER_NAMESPACE_BEGIN

GeometryBuffer::GeometryBuffer(GeometryBufferType_ type,
                               GeometryBufferFormat_ format,
                               const uint32_t stride,
                               const uint32_t count)
{
    ROMANORENDER_ASSERT(stride < UINT8_MAX, "Stride cannot be greater than " ROMANORENDER_STRINGIZE(UINT8_MAX));

    this->type = (uint32_t)type;
    this->format = (uint32_t)format;
    this->stride = stride;
    this->count = count;
    this->refcount = new uint32_t(1);

    this->data = stdromano::mem_aligned_alloc(count * stride, GEOM_BUFFER_ALIGNMENT);
}

GeometryBuffer::GeometryBuffer(const GeometryBuffer& other) noexcept : 
                                                              data(other.data),
                                                              count(other.count),
                                                              stride(other.stride),
                                                              type(other.type),
                                                              format(other.format),
                                                              refcount(other.refcount)
{
    if(this->refcount != nullptr)
    {
        ++(*this->refcount);
    }
}

GeometryBuffer& GeometryBuffer::operator=(const GeometryBuffer& other) noexcept
{
    if(this != &other)
    {
        if(this->refcount != nullptr)
        {
            --(*this->refcount);

            if(*this->refcount == 0)
            {
                stdromano::mem_aligned_free(this->data);
                delete this->refcount;
            }
        }

        std::memmove(this, &other, sizeof(GeometryBuffer));
    
        if(this->refcount != nullptr)
        {
            ++(*this->refcount);
        }
    }

    return *this;
}

GeometryBuffer::GeometryBuffer(GeometryBuffer&& other) noexcept
{
    std::memmove(this, &other, sizeof(GeometryBuffer));
    std::memset(&other, 0, sizeof(GeometryBuffer));
}

GeometryBuffer& GeometryBuffer::operator=(GeometryBuffer&& other) noexcept
{
    if(this != &other)
    {
        if(this->refcount != nullptr)
        {
            --(*this->refcount);

            if(*this->refcount == 0)
            {
                stdromano::mem_aligned_free(this->data);
                delete this->refcount;
            }
        }

        std::memmove(this, &other, sizeof(GeometryBuffer));
        std::memset(&other, 0, sizeof(GeometryBuffer));
    }

    return *this;
}

GeometryBuffer::~GeometryBuffer()
{
    --(*this->refcount);

    if(this->data != nullptr && *this->refcount == 0)
    {
        stdromano::mem_aligned_free(this->data);
        delete this->refcount;
    }
}

void* Geometry::add_geometry_buffer(GeometryBufferType_ type,
                                    GeometryBufferFormat_ format,
                                    const uint32_t stride,
                                    const uint32_t count) noexcept
{
    this->geometry_buffers.emplace_back(type, format, stride, count);

    return this->geometry_buffers.at(this->geometry_buffers.size() - 1)->get_geometry_ptr();
}

ROMANORENDER_NAMESPACE_END
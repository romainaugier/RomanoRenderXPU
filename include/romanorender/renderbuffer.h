#pragma once

#if !defined(__ROMANORENDER_RENDERBUFFER)
#define __ROMANORENDER_RENDERBUFFER

#include "romanorender/vec4.h"

#include "stdromano/vector.h"

#include "GL/glew.h"

ROMANORENDER_NAMESPACE_BEGIN

class ROMANORENDER_API Bucket
{
    Vec4F* pixels = nullptr;

    uint16_t xstart = 0;
    uint16_t ystart = 0;

    uint16_t global_xsize = 0;
    uint16_t global_ysize = 0;

    uint16_t xsize = 0;
    uint16_t ysize = 0;

    uint16_t id = 0;

    /* From bucket space to global space */
    ROMANORENDER_FORCE_INLINE Vec4F* get_address_from_coords(const uint16_t x, const uint16_t y) const noexcept
    {
        return std::addressof(this->pixels[this->global_xsize * (this->ystart + y) + this->xstart + x]);
    }

public:
    Bucket();

    Bucket(Vec4F* pixels,
           const uint16_t xstart, 
           const uint16_t ystart, 
           const uint16_t xsize,
           const uint16_t ysize,
           const uint16_t global_xsize,
           const uint16_t global_ysize,
           const uint16_t id) : 
           pixels(pixels),
           xstart(xstart),
           ystart(ystart),
           xsize(xsize),
           ysize(ysize),
           global_xsize(global_xsize),
           global_ysize(global_ysize),
           id(id) {}

    Bucket(const Bucket& other) noexcept;

    Bucket(Bucket&& other) noexcept;

    ~Bucket();

    /* Coordinates must be in bucket space (0...xsize & 0...ysize) */
    void set_pixel(const Vec4F* color, const uint16_t x, const uint16_t y) noexcept;

    void set_pixels(const Vec4F* color) noexcept;

    ROMANORENDER_FORCE_INLINE uint16_t get_x_start() const noexcept { return this->xstart; }
    ROMANORENDER_FORCE_INLINE uint16_t get_y_start() const noexcept { return this->ystart; }
    ROMANORENDER_FORCE_INLINE uint16_t get_x_end() const noexcept { return this->xstart + this->xsize; }
    ROMANORENDER_FORCE_INLINE uint16_t get_y_end() const noexcept { return this->ystart + this->ysize; }
    ROMANORENDER_FORCE_INLINE uint16_t get_x_size() const noexcept { return this->xsize; }
    ROMANORENDER_FORCE_INLINE uint16_t get_y_size() const noexcept { return this->ysize; }
    ROMANORENDER_FORCE_INLINE uint16_t get_global_x_size() const noexcept { return this->global_xsize; }
    ROMANORENDER_FORCE_INLINE uint16_t get_global_y_size() const noexcept { return this->global_ysize; }
    ROMANORENDER_FORCE_INLINE uint16_t get_id() const noexcept { return this->id; }
};

using Buckets = stdromano::Vector<Bucket>;

class ROMANORENDER_API RenderBuffer
{
    Buckets buckets;
    Vec4F* pixels = nullptr;

    uint16_t xsize = 0;
    uint16_t ysize = 0;
    uint16_t bucket_size = 0;

    uint32_t flags = 0;

    GLuint gl_texture_id;
    GLuint gl_framebuffer_id;

    size_t pixels_buffer_size() const noexcept
    {
        return this->xsize * this->ysize * sizeof(Vec4F);
    }

    void generate_buckets() noexcept;

public:
    RenderBuffer();

    RenderBuffer(const uint16_t xsize, const uint16_t ysize, const uint8_t bucket_size);

    ~RenderBuffer();

    void reinitialize(const uint16_t xsize, const uint16_t ysize, const uint16_t bucket_size) noexcept;

    void update_gl_texture() const noexcept;

    void blit_default_gl_buffer() const noexcept;

    ROMANORENDER_FORCE_INLINE void clear() noexcept
    {
        std::memset(this->pixels, 0, this->pixels_buffer_size());
    }

    Buckets& get_buckets() noexcept { return this->buckets; }
};

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_RENDERBUFFER) */
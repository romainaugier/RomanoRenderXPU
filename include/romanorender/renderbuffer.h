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

    uint8_t xsize = 0;
    uint8_t ysize = 0;

    uint16_t id = 0;

    size_t pixels_buffer_size() const noexcept
    {
        return this->xsize * this->ysize * sizeof(Vec4F);
    }

public:
    Bucket();

    Bucket(const uint16_t xstart, 
         const uint16_t ystart, 
         const uint8_t xsize,
         const uint8_t ysize,
         const uint16_t id);

    Bucket(Bucket&& other) noexcept;

    ~Bucket();

    void set_pixel(const Vec4F* color, const uint16_t x, const uint16_t y) noexcept;

    void set_pixels(const Vec4F* color) noexcept;

    ROMANORENDER_FORCE_INLINE uint16_t get_x_start() const noexcept { return this->xstart; }
    ROMANORENDER_FORCE_INLINE uint16_t get_y_start() const noexcept { return this->ystart; }
    ROMANORENDER_FORCE_INLINE uint16_t get_x_end() const noexcept { return this->xstart + this->xsize; }
    ROMANORENDER_FORCE_INLINE uint16_t get_y_end() const noexcept { return this->ystart + this->ysize; }
    ROMANORENDER_FORCE_INLINE uint16_t get_x_size() const noexcept { return this->xsize; }
    ROMANORENDER_FORCE_INLINE uint16_t get_y_size() const noexcept { return this->ysize; }
    ROMANORENDER_FORCE_INLINE uint16_t get_id() const noexcept { return this->id; }

    ROMANORENDER_FORCE_INLINE Vec4F* get_scanline_at_y(const uint32_t y) const noexcept
    {
        return std::addressof(this->pixels[(y - this->get_y_start()) * this->xsize]);
    }
};

using Buckets = stdromano::Vector<Bucket>;

ROMANORENDER_API void generate_buckets(Buckets* buckets, 
                                       const uint32_t xres,
                                       const uint32_t yres,
                                       const uint16_t bucket_size) noexcept;

class ROMANORENDER_API RenderBuffer
{
    Vec4F* pixels = nullptr;

    uint16_t xsize = 0;
    uint16_t ysize = 0;

    uint32_t flags = 0;

    GLuint gl_texture_id;
    GLuint gl_framebuffer_id;

    size_t pixels_buffer_size() const noexcept
    {
        return this->xsize * this->ysize * sizeof(Vec4F);
    }

public:
    RenderBuffer();

    RenderBuffer(const uint16_t xsize, const uint16_t ysize);

    ~RenderBuffer();

    void reinitialize(const uint16_t xsize, const uint16_t ysize) noexcept;

    void update_bucket(const Bucket* bucket) noexcept;

    void update_gl_texture() const noexcept;

    void blit_default_gl_buffer() const noexcept;

    ROMANORENDER_FORCE_INLINE void clear() noexcept
    {
        std::memset(this->pixels, 0, this->pixels_buffer_size());
    }
};

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_RENDERBUFFER) */
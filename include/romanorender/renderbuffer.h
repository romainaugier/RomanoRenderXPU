#pragma once

#if !defined(__ROMANORENDER_RENDERBUFFER)
#define __ROMANORENDER_RENDERBUFFER

#include "romanorender/vec4.h"

#include "stdromano/vector.h"

ROMANORENDER_NAMESPACE_BEGIN

class alignas(16) ROMANORENDER_API Tile
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
    Tile();

    Tile(const uint16_t xstart, 
         const uint16_t ystart, 
         const uint8_t xsize,
         const uint8_t ysize,
         const uint16_t id);

    ~Tile();

    void set_pixel(const Vec4F* color, const uint16_t x, const uint16_t y) noexcept;

    ROMANORENDER_FORCE_INLINE uint16_t x_start() const noexcept { return this->xstart; }
    ROMANORENDER_FORCE_INLINE uint16_t y_start() const noexcept { return this->ystart; }
    ROMANORENDER_FORCE_INLINE uint16_t x_end() const noexcept { return this->xstart + this->xsize; }
    ROMANORENDER_FORCE_INLINE uint16_t y_end() const noexcept { return this->ystart + this->ysize; }
    ROMANORENDER_FORCE_INLINE uint16_t x_size() const noexcept { return this->xsize; }
    ROMANORENDER_FORCE_INLINE uint16_t y_size() const noexcept { return this->ysize; }

    ROMANORENDER_FORCE_INLINE Vec4F* scanline_at_y(const uint32_t y) const noexcept
    {
        return std::addressof(this->pixels[y * this->xsize]);
    }
};

using Tiles = stdromano::Vector<Tile>;

ROMANORENDER_API void generate_tiles(Tiles* tiles, 
                                     const uint32_t xres,
                                     const uint32_t yres,
                                     const uint16_t tile_size) noexcept;

class RenderBuffer
{
    Vec4F* pixels = nullptr;

    uint16_t xsize = 0;
    uint16_t ysize = 0;

    uint32_t flags = 0;

    size_t pixels_buffer_size() const noexcept
    {
        return this->xsize * this->ysize * sizeof(Vec4F);
    }

public:
    RenderBuffer();

    RenderBuffer(const uint16_t xsize, const uint16_t ysize);

    ~RenderBuffer();

    void reinitialize(const uint16_t xsize, const uint16_t ysize) noexcept;

    void update_tile(const Tile* tile) noexcept;

    ROMANORENDER_FORCE_INLINE void clear() noexcept
    {
        std::memset(this->pixels, 0, this->pixels_buffer_size());
    }
};

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_RENDERBUFFER) */
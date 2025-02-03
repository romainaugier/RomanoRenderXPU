#include "romanorender/renderbuffer.h"

#include "stdromano/memory.h"

ROMANORENDER_NAMESPACE_BEGIN

Tile::Tile()
{
    std::memset(this, 0, sizeof(Tile));
}

Tile::Tile(const uint16_t xstart, 
           const uint16_t ystart, 
           const uint8_t xsize,
           const uint8_t ysize,
           const uint16_t id)
{
    this->xstart = xstart; 
    this->ystart = ystart; 
    this->xsize = xsize; 
    this->ysize = ysize; 
    this->id = id; 

    this->pixels = static_cast<Vec4F*>(stdromano::mem_aligned_alloc(this->pixels_buffer_size(), 32));
}

Tile::~Tile()
{
    if(this->pixels != nullptr)
    {
        stdromano::mem_aligned_free(this->pixels);
        this->pixels = nullptr;
    }
}

void Tile::set_pixel(const Vec4F* color, const uint16_t x, const uint16_t y) noexcept
{
    std::memcpy(std::addressof(this->pixels[y * this->xsize + x]), color, sizeof(Vec4F));
}

void generate_tiles(Tiles* tiles, 
                    const uint32_t xres, 
                    const uint32_t yres,
                    const uint16_t tile_size) noexcept
{
    tiles->clear();

    for(uint32_t x = 0; x < xres; x += tile_size)
    {
        for(uint32_t y = 0; y < yres; y += tile_size)
        {
            const uint8_t xsize = tile_size > (xres - x) ? (xres - x) : tile_size;
            const uint8_t ysize = tile_size > (yres - y) ? (yres - y) : tile_size;

            tiles->emplace_back((uint16_t)x, (uint16_t)y, xsize, ysize, (uint16_t)(tiles->size() - 1));
        }
    }
}

RenderBuffer::RenderBuffer()
{
    std::memset(this, 0, sizeof(RenderBuffer));
}

RenderBuffer::RenderBuffer(const uint16_t xres, const uint16_t yres)
{
    this->reinitialize(xres, yres);
}

RenderBuffer::~RenderBuffer()
{
    if(this->pixels != nullptr)
    {
        stdromano::mem_aligned_free(this->pixels);
        this->pixels = nullptr;
    }
}

void RenderBuffer::reinitialize(const uint16_t xres, const uint16_t yres) noexcept
{
    if(this->pixels != nullptr)
    {
        stdromano::mem_aligned_free(this->pixels);
        this->pixels = nullptr;
    }

    this->xsize = xsize;
    this->ysize = ysize;

    this->pixels = static_cast<Vec4F*>(stdromano::mem_aligned_alloc(this->pixels_buffer_size(), 32));
}

void RenderBuffer::update_tile(const Tile* tile) noexcept
{
    for(uint32_t y = tile->y_start(); y < tile->y_end(); y++)
    {
        Vec4F* scanline = std::addressof(this->pixels[y * this->xsize + tile->x_start()]);
        std::memcpy(scanline, tile->scanline_at_y(y), tile->x_size() * sizeof(Vec4F));
    }
}

ROMANORENDER_NAMESPACE_END
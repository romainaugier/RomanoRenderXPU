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

Tile::Tile(Tile&& other) noexcept
{
    std::memmove(this, &other, sizeof(Tile));
    std::memset(&other, 0, sizeof(Tile));
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
    std::memcpy(std::addressof(this->pixels[y * this->get_x_size() + x]), color, sizeof(Vec4F));
}

void Tile::set_pixels(const Vec4F* color) noexcept
{
    for(uint16_t x = 0; x < this->get_x_size(); x++)
    {
        for(uint16_t y = 0; y < this->get_y_size(); y++)
        {
            this->pixels[y * this->get_x_size() + x] = *color;
        }
    }
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

void RenderBuffer::reinitialize(const uint16_t xsize, const uint16_t ysize) noexcept
{
    if(this->pixels != nullptr)
    {
        stdromano::mem_aligned_free(this->pixels);
        this->pixels = nullptr;

        glDeleteTextures(1, &this->gl_texture_id);
    }

    this->xsize = xsize;
    this->ysize = ysize;

    this->pixels = static_cast<Vec4F*>(stdromano::mem_aligned_alloc(this->pixels_buffer_size(), 32));

    glGenTextures(1, &this->gl_texture_id);
    glBindTexture(GL_TEXTURE_2D, this->gl_texture_id);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR); 
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, xsize, ysize, 0, GL_RGBA, GL_FLOAT, this->pixels);
    glBindTexture(GL_TEXTURE_2D, 0);

    glGenFramebuffers(1, &this->gl_framebuffer_id);
    glBindFramebuffer(GL_FRAMEBUFFER, this->gl_framebuffer_id);
    glFramebufferTexture(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, this->gl_texture_id, 0);
    glBindFramebuffer(GL_FRAMEBUFFER, 0);
}

void RenderBuffer::update_tile(const Tile* tile) noexcept
{
    for(uint32_t y = tile->get_y_start(); y < tile->get_y_end(); y++)
    {
        Vec4F* scanline = std::addressof(this->pixels[y * this->xsize + tile->get_x_start()]);
        std::memcpy(scanline, tile->get_scanline_at_y(y), tile->get_x_size() * sizeof(Vec4F));
    }
}

void RenderBuffer::update_gl_texture() const noexcept
{
    glBindTexture(GL_TEXTURE_2D, this->gl_texture_id);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->xsize, this->ysize, GL_RGBA, GL_FLOAT, this->pixels);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void RenderBuffer::blit_default_gl_buffer() const noexcept
{
    glBindFramebuffer(GL_READ_FRAMEBUFFER, this->gl_framebuffer_id);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, this->xsize, this->ysize, 0, 0, this->xsize, this->ysize, GL_COLOR_BUFFER_BIT, GL_LINEAR);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}

ROMANORENDER_NAMESPACE_END
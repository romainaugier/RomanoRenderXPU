#include "romanorender/renderbuffer.h"
#include "romanorender/cuda_utils.h"

#include "stdromano/memory.h"

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "romanorender/stb_image_write.h"

#include <cuda_runtime.h>

ROMANORENDER_NAMESPACE_BEGIN

Bucket::Bucket() { std::memset(this, 0, sizeof(Bucket)); }

Bucket::Bucket(Bucket&& other) noexcept { std::memcpy(this, &other, sizeof(Bucket)); }

Bucket::Bucket(const Bucket& other) noexcept { std::memcpy(this, &other, sizeof(Bucket)); }

Bucket::~Bucket() {}

void Bucket::set_pixel(const Vec4F* color, const uint16_t x, const uint16_t y) noexcept
{
    ROMANORENDER_ASSERT(x < this->xsize, "Bucket coordinates must be in bucket space");
    ROMANORENDER_ASSERT(y < this->ysize, "Bucket coordinates must be in bucket space");

    std::memcpy(this->get_address_from_coords(x, y), color, sizeof(Vec4F));
}

void Bucket::set_pixels(const Vec4F* color) noexcept
{
    for(uint16_t x = 0; x < this->get_x_size(); x++)
    {
        for(uint16_t y = 0; y < this->get_y_size(); y++)
        {
            std::memcpy(this->get_address_from_coords(x, y), color, sizeof(Vec4F));
        }
    }
}

RenderBuffer::RenderBuffer() { std::memset(this, 0, sizeof(RenderBuffer)); }

RenderBuffer::RenderBuffer(const uint16_t xres, const uint16_t yres, const uint16_t bucket_size, const bool no_gl)
{
    this->no_gl = no_gl;

    this->reinitialize(xres, yres, bucket_size, no_gl);
}

RenderBuffer::~RenderBuffer()
{
    if(this->pixels != nullptr)
    {
        CUDA_CHECK(cudaFreeHost(this->pixels));
        this->pixels = nullptr;
    }
}

void RenderBuffer::generate_buckets() noexcept
{
    this->buckets.clear();

    for(uint32_t x = 0; x < this->xsize; x += this->bucket_size)
    {
        for(uint32_t y = 0; y < this->ysize; y += this->bucket_size)
        {
            const uint16_t xsize = bucket_size > (this->xsize - x) ? (this->xsize - x) : this->bucket_size;
            const uint16_t ysize = bucket_size > (this->ysize - y) ? (this->ysize - y) : this->bucket_size;

            this->buckets.emplace_back(this->pixels,
                                       (uint16_t)x,
                                       (uint16_t)y,
                                       xsize,
                                       ysize,
                                       this->xsize,
                                       this->ysize,
                                       (uint16_t)(this->buckets.size() - 1));
        }
    }
}

void RenderBuffer::reinitialize(const uint16_t xsize,
                                const uint16_t ysize,
                                const uint16_t bucket_size,
                                const bool no_gl) noexcept
{
    this->no_gl = no_gl;

    if(this->pixels != nullptr)
    {
        CUDA_CHECK(cudaFreeHost(this->pixels));

        this->pixels = nullptr;

        if(!this->no_gl)
        {
            glDeleteTextures(1, &this->gl_texture_id);
        }
    }

    this->xsize = xsize;
    this->ysize = ysize;
    this->bucket_size = bucket_size;

    CUDA_CHECK(cudaMallocHost(&this->pixels, this->pixels_buffer_size()));

    this->generate_buckets();

    if(this->no_gl)
    {
        return;
    }

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

void RenderBuffer::update_gl_texture() const noexcept
{
    if(this->no_gl)
    {
        return;
    }

    glBindTexture(GL_TEXTURE_2D, this->gl_texture_id);
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, this->xsize, this->ysize, GL_RGBA, GL_FLOAT, this->pixels);
    glBindTexture(GL_TEXTURE_2D, 0);
}

void RenderBuffer::blit_default_gl_buffer() const noexcept
{
    if(this->no_gl)
    {
        return;
    }

    glBindFramebuffer(GL_READ_FRAMEBUFFER, this->gl_framebuffer_id);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
    glBlitFramebuffer(0, 0, this->xsize, this->ysize, 0, 0, this->xsize, this->ysize, GL_COLOR_BUFFER_BIT, GL_LINEAR);
    glBindFramebuffer(GL_READ_FRAMEBUFFER, 0);
    glBindFramebuffer(GL_DRAW_FRAMEBUFFER, 0);
}

bool RenderBuffer::to_jpg(const char* filepath) const noexcept
{
    uint8_t* rgb8_buffer = (uint8_t*)stdromano::mem_aligned_alloc(this->xsize * this->ysize * 3 * sizeof(uint8_t), 32);

    for(uint32_t i = 0; i < this->xsize * this->ysize; i++)
    {
        for(uint32_t j = 0; j < 3; j++)
        {
            rgb8_buffer[i * 3 + j] = (uint8_t)(maths::clampf(this->pixels[i][j]) * 255.0f);
        }
    }

    int res = stbi_write_jpg(filepath, this->xsize, this->ysize, 3, (const void*)rgb8_buffer, 100);

    stdromano::mem_aligned_free(rgb8_buffer);

    return res != 0;
}

ROMANORENDER_NAMESPACE_END
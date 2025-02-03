#include "romanorender/renderengine.h"

#include "stdromano/random.h"

ROMANORENDER_NAMESPACE_BEGIN

#define INITIAL_SAMPLE_VALUE 1

RenderEngine::RenderEngine()
{
    constexpr uint32_t default_xres = 1280; 
    constexpr uint32_t default_yres = 720;
    constexpr uint32_t default_bucket_size = 32;

    this->settings[RenderEngineSetting_XSize] = default_xres;
    this->settings[RenderEngineSetting_YSize] = default_yres;
    this->settings[RenderEngineSetting_BucketSize] = default_bucket_size;

    this->reinitialize();
}

RenderEngine::RenderEngine(const uint32_t xres, const uint32_t yres)
{
    constexpr uint32_t default_bucket_size = 32;

    this->settings[RenderEngineSetting_XSize] = xres;
    this->settings[RenderEngineSetting_YSize] = yres;
    this->settings[RenderEngineSetting_BucketSize] = default_bucket_size;

    this->reinitialize();
}

RenderEngine::~RenderEngine()
{

}

void RenderEngine::reinitialize() noexcept
{
    const uint16_t xres = this->get_setting(RenderEngineSetting_XSize);
    const uint16_t yres = this->get_setting(RenderEngineSetting_YSize);
    const uint16_t bucket_size = this->get_setting(RenderEngineSetting_BucketSize);

    generate_buckets(&this->buckets, xres, yres, bucket_size); 

    this->buffer.reinitialize(xres, yres);

    this->current_sample = INITIAL_SAMPLE_VALUE;
}

void RenderEngine::set_setting(const uint32_t setting, const uint32_t value, const bool noreinit) noexcept
{
    this->settings[setting] = value;

    if(!noreinit)
    {
        switch(setting)
        {
            case RenderEngineSetting_XSize:
            case RenderEngineSetting_YSize:
            case RenderEngineSetting_BucketSize:
                this->reinitialize();
                break;

            case RenderEngineSetting_MaxBounces:
                this->clear();
                break;
            
            default:
                break;
        }
    }
}

uint32_t RenderEngine::get_setting(const uint32_t setting) const noexcept
{
    const auto it = this->settings.find(setting);

    return it == this->settings.end() ? UINT32_MAX : it.value();
}

void RenderEngine::render_sample() noexcept
{
    for(auto& bucket: this->buckets)
    {
        Vec4F bucket_color(stdromano::pcg_float(bucket.get_id() + 0),
                         stdromano::pcg_float(bucket.get_id() + 1),
                         stdromano::pcg_float(bucket.get_id() + 2),
                         1.0f);

        bucket.set_pixels(&bucket_color);

        this->buffer.update_bucket(&bucket);
    }

    this->current_sample++;
    this->buffer.update_gl_texture();
}

void RenderEngine::render_full() noexcept
{

}

void RenderEngine::clear() noexcept
{
    this->current_sample = INITIAL_SAMPLE_VALUE;
}

ROMANORENDER_NAMESPACE_END
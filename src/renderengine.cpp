#include "romanorender/renderengine.h"

ROMANORENDER_NAMESPACE_BEGIN

#define INITIAL_SAMPLE_VALUE 1

RenderEngine::RenderEngine()
{
    constexpr uint32_t default_xres = 1280; 
    constexpr uint32_t default_yres = 720;

    this->set_setting(RenderEngineSetting_XRes, default_xres);
    this->set_setting(RenderEngineSetting_YRes, default_yres);

    this->reinitialize();
}

RenderEngine::RenderEngine(const uint32_t xres, const uint32_t yres)
{
    this->set_setting(RenderEngineSetting_XRes, xres);
    this->set_setting(RenderEngineSetting_YRes, yres);

    this->reinitialize();
}

RenderEngine::~RenderEngine()
{

}

void RenderEngine::reinitialize() noexcept
{
    const uint16_t xres = this->get_setting(RenderEngineSetting_XRes);
    const uint16_t yres = this->get_setting(RenderEngineSetting_YRes);
    const uint16_t tile_size = this->get_setting(RenderEngineSetting_TileSize);

    generate_tiles(&this->tiles, xres, yres, tile_size); 

    this->buffer.reinitialize(xres, yres);

    this->current_sample = INITIAL_SAMPLE_VALUE;
}

void RenderEngine::set_setting(const uint32_t setting, const uint32_t value) noexcept
{
    this->settings[setting] = value;

    switch(setting)
    {
        case RenderEngineSetting_XRes:
        case RenderEngineSetting_YRes:
        case RenderEngineSetting_TileSize:
            this->reinitialize();
            break;

        case RenderEngineSetting_MaxBounces:
            this->clear();
            break;
        
        default:
            break;
    }
}

uint32_t RenderEngine::get_setting(const uint32_t setting) const noexcept
{
    const auto it = this->settings.find(setting);

    return it == this->settings.end() ? UINT32_MAX : it.value();
}

void RenderEngine::render_sample() noexcept
{

}

void RenderEngine::render_full() noexcept
{

}

void RenderEngine::clear() noexcept
{
    this->current_sample = INITIAL_SAMPLE_VALUE;
}

ROMANORENDER_NAMESPACE_END
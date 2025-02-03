#pragma once

#if !defined(__ROMANORENDER_RENDERENGINE)
#define __ROMANORENDER_RENDERENGINE

#include "romanorender/renderbuffer.h"

#include "stdromano/hashmap.h"

ROMANORENDER_NAMESPACE_BEGIN

enum RenderEngineSetting_ : uint32_t
{
    RenderEngineSetting_XSize = 1,
    RenderEngineSetting_YSize = 2,
    RenderEngineSetting_MaxSamples = 3,
    RenderEngineSetting_MaxBounces = 4,
    RenderEngineSetting_BucketSize = 5,
    RenderEngineSetting_Device = 6,
};

enum RenderEngineFlag_ : uint32_t
{
    RenderEngineFlag_Initialized = 0x1,
};

class ROMANORENDER_API RenderEngine
{
    RenderBuffer buffer;
    Buckets buckets;

    stdromano::HashMap<uint32_t, uint32_t> settings;

    uint32_t current_sample = 0;

    uint32_t flags = 0;

    void reinitialize() noexcept;

public:
    RenderEngine();

    RenderEngine(const uint32_t xres, const uint32_t yres);

    ~RenderEngine();

    void set_setting(const uint32_t setting, const uint32_t value, const bool noreinit = false) noexcept;

    uint32_t get_setting(const uint32_t setting) const noexcept;

    ROMANORENDER_FORCE_INLINE const RenderBuffer* get_renderbuffer() const noexcept { return &this->buffer; }
    ROMANORENDER_FORCE_INLINE uint32_t get_current_sample() const noexcept { return this->current_sample; }

    /* Accumulates just one sample */
    void render_sample() noexcept;

    /* Accumulates all samples */
    void render_full() noexcept;

    void clear() noexcept;
};

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_RENDERENGINE) */
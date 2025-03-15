#pragma once

#if !defined(__ROMANORENDER_RENDERENGINE)
#define __ROMANORENDER_RENDERENGINE

#include "romanorender/integrator.h"
#include "romanorender/renderbuffer.h"

#include "stdromano/hashmap.h"

ROMANORENDER_NAMESPACE_BEGIN

enum RenderEngineDevice_ : uint32_t
{
    RenderEngineDevice_CPU = 1,
    RenderEngineDevice_GPU = 2,
};

enum RenderEngineSetting_ : uint32_t
{
    RenderEngineSetting_XSize = 1,
    RenderEngineSetting_YSize = 2,
    RenderEngineSetting_MaxSamples = 3,
    RenderEngineSetting_MaxBounces = 4,
    RenderEngineSetting_BucketSize = 5,
    RenderEngineSetting_Device = 6,
    RenderEngineSetting_NoOpenGL = 7,
};

enum RenderEngineFlag_ : uint32_t
{
    RenderEngineFlag_Initialized = 0x1,
};

class ROMANORENDER_API RenderEngine
{
    Scene scene;

    RenderBuffer buffer;

    stdromano::HashMap<uint32_t, uint32_t> settings;

    uint32_t current_sample = 0;

    uint32_t flags = 0;

    void reinitialize() noexcept;

public:
    RenderEngine(const bool no_gl = false, const uint32_t device = RenderEngineDevice_CPU);

    RenderEngine(const uint32_t xres,
                 const uint32_t yres,
                 const bool no_gl = false,
                 const uint32_t device = RenderEngineDevice_CPU);

    ~RenderEngine();

    void set_setting(const uint32_t setting, const uint32_t value, const bool noreinit = false) noexcept;

    uint32_t get_setting(const uint32_t setting) const noexcept;

    ROMANORENDER_FORCE_INLINE const RenderBuffer* get_renderbuffer() const noexcept { return &this->buffer; }

    ROMANORENDER_FORCE_INLINE uint32_t get_current_sample() const noexcept { return this->current_sample; }

    ROMANORENDER_FORCE_INLINE Scene* get_scene() noexcept { return &this->scene; }

    void prepare_for_rendering() noexcept;

    /* Accumulates just one sample */
    void render_sample(integrator_func integrator) noexcept;

    /* Accumulates all samples */
    void render_full(integrator_func integrator) noexcept;

    void clear() noexcept;
};

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_RENDERENGINE) */
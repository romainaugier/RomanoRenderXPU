#include "romanorender/renderengine.h"
#include "romanorender/optix_utils.h"

#include "stdromano/logger.h"
#include "stdromano/random.h"
#include "stdromano/threading.h"

#define STDROMANO_ENABLE_PROFILING
#include "stdromano/profiling.h"

ROMANORENDER_NAMESPACE_BEGIN

#define INITIAL_SAMPLE_VALUE 1

RenderEngine::RenderEngine(const bool no_gl, const uint32_t device)
{
    stdromano::log_info("Initializing Render Engine");

    log_cuda_version();

    stdromano::global_threadpool.get_instance();

    constexpr uint32_t default_xres = 1280;
    constexpr uint32_t default_yres = 720;
    constexpr uint32_t default_bucket_size = 64;

    this->settings[RenderEngineSetting_XSize] = default_xres;
    this->settings[RenderEngineSetting_YSize] = default_yres;
    this->settings[RenderEngineSetting_BucketSize] = default_bucket_size;
    this->settings[RenderEngineSetting_NoOpenGL] = (uint32_t)no_gl;
    this->settings[RenderEngineSetting_Device] = device;

    this->scene.set_backend(device == RenderEngineDevice_CPU ? SceneBackend_CPU : SceneBackend_GPU);

    this->reinitialize();
}

RenderEngine::RenderEngine(const uint32_t xres, const uint32_t yres, const bool no_gl, const uint32_t device)
    : RenderEngine(no_gl, device)
{
    constexpr uint32_t default_bucket_size = 64;

    this->settings[RenderEngineSetting_XSize] = xres;
    this->settings[RenderEngineSetting_YSize] = yres;
    this->settings[RenderEngineSetting_BucketSize] = default_bucket_size;

    this->reinitialize();
}

RenderEngine::~RenderEngine() { stdromano::global_threadpool.stop(); }

void RenderEngine::reinitialize() noexcept
{
    const uint16_t xres = this->get_setting(RenderEngineSetting_XSize);
    const uint16_t yres = this->get_setting(RenderEngineSetting_YSize);
    const uint16_t bucket_size = this->get_setting(RenderEngineSetting_BucketSize);
    const bool no_gl = this->get_setting(RenderEngineSetting_NoOpenGL);

    this->buffer.reinitialize(xres, yres, bucket_size, no_gl);

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

        case RenderEngineSetting_Device:
            this->scene.set_backend(value == RenderEngineDevice_CPU ? SceneBackend_CPU : SceneBackend_GPU);
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

void RenderEngine::prepare_for_rendering() noexcept
{
    this->get_scene()->get_camera()->set_xres(this->get_setting(RenderEngineSetting_XSize));
    this->get_scene()->get_camera()->set_yres(this->get_setting(RenderEngineSetting_YSize));
}

void RenderEngine::render_sample(integrator_func integrator) noexcept
{
    SCOPED_PROFILE_START(stdromano::ProfileUnit::MilliSeconds, render_sample);

    const uint32_t device = this->get_setting(RenderEngineSetting_Device);

    switch(device)
    {
    case RenderEngineDevice_CPU:
    {
        for(auto& bucket : this->buffer.get_buckets())
        {
            stdromano::global_threadpool.add_work(
                [&]()
                {
                    for(uint16_t x = bucket.get_x_start(); x < bucket.get_x_end(); x++)
                    {
                        for(uint16_t y = bucket.get_y_start(); y < bucket.get_y_end(); y++)
                        {
                            const Vec4F output = integrator == nullptr
                                                     ? Vec4F(0.0f)
                                                     : integrator(&this->scene, x, y, this->current_sample);

                            bucket.set_pixel(&output, x - bucket.get_x_start(), y - bucket.get_y_start());
                        }
                    }
                });
        }

        stdromano::global_threadpool.wait();
        break;
    }
    case RenderEngineDevice_GPU:
    {
        const uint32_t xres = this->get_setting(RenderEngineSetting_XSize);
        const uint32_t yres = this->get_setting(RenderEngineSetting_YSize);

        const Vec3F camera_pos = this->get_scene()->get_camera()->get_ray_origin();
        const Vec3F camera_dir = this->get_scene()->get_camera()->get_ray_direction(xres / 2, yres / 2);

        OptixParams params;
        std::memcpy(
            &params.camera_transform, this->get_scene()->get_camera()->get_transform().data(), 16 * sizeof(float));
        params.camera_fov = this->get_scene()->get_camera()->get_fov();
        params.camera_aspect = this->get_scene()->get_camera()->get_aspect();
        params.pixels = (float4*)this->get_renderbuffer()->get_pixels();
        params.handle = *reinterpret_cast<OptixTraversableHandle*>(this->get_scene()->get_as_handle());
        params.current_sample = this->current_sample;

        OptixManager::get_instance().update_params(&params);

        OptixManager::get_instance().launch(xres, yres);

        CUDA_SYNC_CHECK();
    }
    }

    this->current_sample++;
    this->buffer.update_gl_texture();
}

void RenderEngine::render_full(integrator_func integrator) noexcept {}

void RenderEngine::clear() noexcept { this->current_sample = INITIAL_SAMPLE_VALUE; }

ROMANORENDER_NAMESPACE_END
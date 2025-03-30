#include "romanorender/renderengine.h"
#include "romanorender/optix_utils.h"
#include "romanorender/sampling.h"

#include "stdromano/logger.h"
#include "stdromano/random.h"
#include "stdromano/threading.h"

#define STDROMANO_ENABLE_PROFILING
#include "stdromano/profiling.h"

ROMANORENDER_NAMESPACE_BEGIN

#define INITIAL_SAMPLE_VALUE 1

void atexit_handler_stdromano_global_threadpool() { stdromano::atexit_handler_global_threadpool(); }

RenderEngine::RenderEngine(const bool no_gl, const uint32_t device)
{
    stdromano::log_info("Initializing Render Engine");

    stdromano::global_threadpool.get_instance();

    STDROMANO_ATEXIT_REGISTER(atexit_handler_stdromano_global_threadpool, true);

    log_cuda_version();

    sampler().initialize();

    this->_render_thread = new stdromano::Thread([&]() { this->render_loop(); });
    this->_render_thread->start();

    constexpr uint32_t default_xres = 1280;
    constexpr uint32_t default_yres = 720;
    constexpr uint32_t default_bucket_size = 64;

    this->settings[RenderEngineSetting_XSize] = default_xres;
    this->settings[RenderEngineSetting_YSize] = default_yres;
    this->settings[RenderEngineSetting_BucketSize] = default_bucket_size;
    this->settings[RenderEngineSetting_MaxBounces] = 6;
    this->settings[RenderEngineSetting_MaxSamples] = 2048;
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

RenderEngine::~RenderEngine()
{
    stdromano::log_debug("Shutting down render engine");

    this->_stop.store(true);
    this->_is_rendering.store(false);

    stdromano::thread_sleep(50);

    this->_render_thread->join();
    delete this->_render_thread;
}

void RenderEngine::reinitialize() noexcept
{
    const uint16_t xres = this->get_setting(RenderEngineSetting_XSize);
    const uint16_t yres = this->get_setting(RenderEngineSetting_YSize);
    const uint16_t bucket_size = this->get_setting(RenderEngineSetting_BucketSize);
    const bool no_gl = this->get_setting(RenderEngineSetting_NoOpenGL);

    this->buffer.reinitialize(xres, yres, bucket_size, no_gl);

    this->current_sample = INITIAL_SAMPLE_VALUE;
}

void RenderEngine::render_loop()
{
    stdromano::log_debug("Starting Renderloop Thread");

    while(!this->_stop.load())
    {
        while(this->_is_rendering.load())
        {
            bool had_changes = this->_any_change.exchange(false);

            if(had_changes || this->_scene_graph.is_dirty())
            {
                this->prepare_for_rendering();
                this->clear();
            }

            this->render_sample(this->_integrator);

            stdromano::thread_sleep(1);
        }

        stdromano::thread_sleep(10);
    }
}

void RenderEngine::set_setting(const uint32_t setting, const uint32_t value, const bool noreinit) noexcept
{
    this->settings[setting] = value;

    bool restart_render = false;

    if(this->_is_rendering.load())
    {
        this->_is_rendering.store(false);
        restart_render = true;
    }

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

    this->_any_change.store(true);

    if(restart_render)
    {
        this->start_rendering(this->_integrator);
    }
}

uint32_t RenderEngine::get_setting(const uint32_t setting) const noexcept
{
    const auto it = this->settings.find(setting);

    return it == this->settings.end() ? UINT32_MAX : it.value();
}

void RenderEngine::set_camera_transform(const Mat44F& transform) noexcept
{
    bool restart_render = false;

    if(this->get_scene()->get_camera() == nullptr)
    {
        return;
    }

    if(this->_is_rendering.load())
    {
        this->_is_rendering.store(false);
        restart_render = true;
    }

    this->clear();

    this->get_scene()->get_camera()->set_transform(transform);

    if(restart_render)
    {
        this->start_rendering(this->_integrator);
    }
}

void RenderEngine::prepare_for_rendering() noexcept
{
    if(this->get_scene_graph().is_dirty())
    {
        SCOPED_PROFILE_START(stdromano::ProfileUnit::Seconds, scene_graph_execution);

        if(!this->get_scene_graph().execute())
        {
            stdromano::log_error("Error caught while executing scene graph");
            
            const SceneGraphNode* error_node = this->get_scene_graph().get_error_node();

            if(error_node != nullptr)
            {
                stdromano::log_error(error_node->get_error().empty() ? "Unknown error" : error_node->get_error().c_str());
            }
            else
            {
                stdromano::log_error("Unknown error");
            }

            this->stop_rendering();

            return;
        }
    }

    this->get_scene()->build_from_scenegraph(this->get_scene_graph());

    if(this->get_scene()->get_num_lights() == 0)
    {
        this->stop_rendering();

        stdromano::log_error("No light detected in the scene, aborting render");

        return;
    }

    if(this->get_scene()->get_camera() == nullptr)
    {
        stdromano::log_debug("Creating default camera");

        ObjectCamera* cam = new ObjectCamera;

        cam->set_focal(50.0);
        cam->set_name("default");

        this->get_scene()->set_camera(cam->get_camera());

        ObjectsManager::get_instance().add_object(cam);
    }

    this->get_scene()->get_camera()->set_xres(this->get_setting(RenderEngineSetting_XSize));
    this->get_scene()->get_camera()->set_yres(this->get_setting(RenderEngineSetting_YSize));
}

void RenderEngine::start_rendering(integrator_func integrator) noexcept
{
    stdromano::log_debug("Start rendering loop");
    this->_integrator = integrator;
    this->_is_rendering.store(true);
}

void RenderEngine::stop_rendering() noexcept
{
    stdromano::log_debug("Stop rendering loop");
    this->_is_rendering.store(false);
}

void RenderEngine::render_sample(integrator_func integrator) noexcept
{
    if(this->get_scene_graph().is_dirty())
    {
        stdromano::log_error("Cannot render, scene graph is dirty");
        return;
    }

    SCOPED_PROFILE_START(stdromano::ProfileUnit::MilliSeconds, render_sample);

    const uint32_t device = this->get_setting(RenderEngineSetting_Device);
    const uint16_t max_bounces = this->get_setting(RenderEngineSetting_MaxBounces);

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
                            const Vec4F previous = bucket.get_pixel(x - bucket.get_x_start(), y - bucket.get_y_start());

                            const Vec4F output = integrator == nullptr
                                                     ? Vec4F(0.0f)
                                                     : integrator(&this->scene, x, y, this->current_sample, max_bounces);

                            const Vec4F color = lerp_vec4ff(previous, output, 1.0f / static_cast<float>(this->current_sample));
                                                    
                            bucket.set_pixel(&color, x - bucket.get_x_start(), y - bucket.get_y_start());
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

        OptixParams params;
        std::memcpy(&params.camera_transform,
                    this->get_scene()->get_camera()->get_transform().data(),
                    16 * sizeof(float));
        params.camera_fov = this->get_scene()->get_camera()->get_fov();
        params.camera_aspect = this->get_scene()->get_camera()->get_aspect();
        params.pixels = (float4*)this->get_renderbuffer()->get_pixels();
        
        for(uint32_t i = 0; i < NUM_PMJ02_SEQUENCES; i++)
        {
            params.pmj_samples[i] = reinterpret_cast<const float2*>(sampler().get_pmj02_sequence_ptr(i));
        }

        params.handle = *reinterpret_cast<OptixTraversableHandle*>(this->get_scene()->get_as_handle());
        params.current_sample = this->current_sample;
        params.seed = 0x5643718FE9;

        optix_manager().update_params(&params);

        optix_manager().launch(xres, yres);

        break;
    }
    }

    this->current_sample++;

    if(this->current_sample >= this->get_setting(RenderEngineSetting_MaxSamples))
    {
        this->stop_rendering();
    }
}

void RenderEngine::render_full(integrator_func integrator) noexcept {}

void RenderEngine::update_gl_texture() noexcept { this->buffer.update_gl_texture(); }

void RenderEngine::clear() noexcept { this->current_sample = INITIAL_SAMPLE_VALUE; }

ROMANORENDER_NAMESPACE_END
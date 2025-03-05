#include "romanorender/renderengine.h"

#include "stdromano/random.h"

#define STDROMANO_ENABLE_PROFILING
#include "stdromano/profiling.h"

using namespace romanorender;

int main()
{
    stdromano::set_log_level(stdromano::LogLevel::Debug);

    constexpr uint32_t xres = 1280;
    constexpr uint32_t yres = 720;

    RenderEngine engine(xres, yres, true);
    Camera camera(Vec3F(-5.0f, -8.0f, 10.0f), Vec3F(0.0f, 0.0f, 0.0f), 50.0f, xres, yres);

    engine.get_scene()->set_camera(camera);

    Object cube = Object::cube(Vec3F(0.0f, 0.0f, 0.0f), Vec3F(1.0f));
    cube.build_blas();

    engine.get_scene()->add_object(cube);
    engine.get_scene()->build_tlas();

    engine.render_sample(integrator_debug);

    if(!engine.get_renderbuffer()->to_jpg("test_render_cube.jpg"))
    {
        return 1;
    }

    return 0;
}
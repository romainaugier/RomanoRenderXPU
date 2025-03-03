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
    Camera camera(Vec3F(3.0f, 5.0f, -5.0f), Vec3F(0.0f, 0.0f, 0.0f), 50.0f, xres, yres);

    engine.get_scene()->set_camera(camera);

    Object geodesic = Object::geodesic(Vec3F(0.0f, 1.0f, 0.0f), Vec3F(1.0f), 4);
    geodesic.set_name(stdromano::String<>::make_ref("geodesic", 9));
    geodesic.build_blas();

    Object plane = Object::plane(Vec3F(0.0f), Vec3F(5.0f));
    plane.set_name(stdromano::String<>::make_ref("plane", 5));
    plane.build_blas();

    engine.get_scene()->add_object(geodesic);
    engine.get_scene()->add_object(plane);
    engine.get_scene()->build_tlas();

    engine.render_sample(integrator_mask);

    if(!engine.get_renderbuffer()->to_jpg("test_render_scene.jpg"))
    {
        return 1;
    }

    return 0;
}
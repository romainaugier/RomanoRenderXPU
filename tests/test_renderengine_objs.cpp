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
    Camera camera(Vec3F(0.0f, 0.0f, 0.08f), Vec3F(0.0f, 0.1f, 0.0f), 24.0f, xres, yres);

    engine.get_scene()->set_camera(camera);

    stdromano::Vector<Object> objects;

    if(!objects_from_obj_file(stdromano::String<>("{}/stanford-bunny.obj", TESTS_DATA_DIR).c_str(), objects))
    {
        return 1;
    }

    if(objects.size() == 0)
    {
        return 0;
    }

    for(Object& object : objects)
    {
        object.build_blas();
        engine.get_scene()->add_object(object);
    }

    engine.get_scene()->build_tlas();

    engine.render_sample(integrator_debug);

    if(!engine.get_renderbuffer()->to_jpg("test_render_objs.jpg"))
    {
        return 1;
    }

    return 0;
}
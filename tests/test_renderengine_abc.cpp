#include "romanorender/renderengine.h"

#include "stdromano/random.h"
#include "stdromano/threading.h"

#define STDROMANO_ENABLE_PROFILING
#include "stdromano/profiling.h"

#include <mutex>

using namespace romanorender;

int main()
{
    stdromano::set_log_level(stdromano::LogLevel::Debug);

    constexpr uint32_t xres = 1280;
    constexpr uint32_t yres = 720;

    RenderEngine engine(xres, yres, true);
    Camera camera(Vec3F(0.0f, 1.5f, 4.0f), Vec3F(0.0f, 1.0f, 0.0f), 24.0f, xres, yres);

    engine.get_scene()->set_camera(&camera);

    if(!objects_from_abc_file(stdromano::String<>("{}/cornell_box_shaderball.abc", TESTS_DATA_DIR).c_str()))
    {
        return 1;
    }

    SCOPED_PROFILE_START(stdromano::ProfileUnit::Seconds, scene_loading);

    stdromano::Mutex load_mutex;

    for(Object* object : ObjectsManager::get_instance().get_objects())
    {
        if(ObjectMesh* mesh = dynamic_cast<ObjectMesh*>(object))
        {
            mesh->build_blas();

            load_mutex.lock();
            engine.get_scene()->add_object_mesh(mesh);
            load_mutex.unlock();
        }
        else if(ObjectCamera* cam = dynamic_cast<ObjectCamera*>(object))
        {
            cam->set_xres(xres);
            cam->set_yres(yres);

            engine.get_scene()->set_camera(cam->get_camera());
        }
    }

    stdromano::global_threadpool.wait();

    SCOPED_PROFILE_STOP(scene_loading);

    engine.get_scene()->build_tlas();

    engine.render_sample(integrator_debug);

    if(!engine.get_renderbuffer()->to_jpg("test_render_abc.jpg"))
    {
        return 1;
    }

    return 0;
}
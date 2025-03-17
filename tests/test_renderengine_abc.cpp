#include "romanorender/renderengine.h"

#include "stdromano/random.h"
#include "stdromano/threading.h"

#define STDROMANO_ENABLE_PROFILING
#include "stdromano/profiling.h"

#include <mutex>

using namespace romanorender;

void atexit_handler_stdromano_global_threadpool() { stdromano::atexit_handler_global_threadpool(); }

int main()
{
    STDROMANO_ATEXIT_REGISTER(atexit_handler_stdromano_global_threadpool, true);
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

    SceneGraphNode* mesh = SceneGraphNodesManager::get_instance().create_node("mesh");

    engine.get_scene_graph().add_node(mesh);

    engine.get_scene_graph().connect_nodes(mesh->get_id(), 0, 0);

    engine.prepare_for_rendering();

    SCOPED_PROFILE_STOP(scene_loading);

    engine.render_sample(integrator_debug);

    if(!engine.get_renderbuffer()->to_jpg("test_render_abc.jpg"))
    {
        return 1;
    }

    return 0;
}
#include "romanorender/renderengine.h"
#include "romanorender/scenegraph.h"

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

    RenderEngine engine(xres, yres, true, RenderEngineDevice_GPU);
    Camera camera(Vec3F(0.0f, 1.5f, 4.0f), Vec3F(0.0f, 1.0f, 0.0f), 24.0f, xres, yres);

    engine.get_scene()->set_camera(&camera);

    if(!objects_from_abc_file(stdromano::String<>("{}/cornell_box_shaderball.abc", TESTS_DATA_DIR).c_str()))
    {
        return 1;
    }

    SCOPED_PROFILE_START(stdromano::ProfileUnit::Seconds, scene_loading);

    SceneGraph scenegraph;

    SceneGraphNode* mesh = SceneGraphNodesManager::get_instance().create_node("mesh");

    scenegraph.add_node(mesh);

    scenegraph.connect_nodes(mesh->get_id(), 0, 0);

    if(!scenegraph.execute())
    {
        stdromano::log_error("Error caught during scenegraph execution");
        return 1;
    }

    stdromano::log_debug("Objects computed from the scenegraph:");

    for(const Object* obj : *scenegraph.get_result())
    {
        stdromano::log_debug(" - {}", obj->get_name());
    }

    engine.get_scene()->build_from_scenegraph(scenegraph);

    engine.prepare_for_rendering();

    SCOPED_PROFILE_STOP(scene_loading);

    engine.render_sample(nullptr);

    if(!engine.get_renderbuffer()->to_jpg("test_render_scenegraph_gpu.jpg"))
    {
        return 1;
    }

    return 0;
}
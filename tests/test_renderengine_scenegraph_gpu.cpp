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

    if(!objects_from_abc_file(stdromano::String<>("{}/cornell_box_shaderball.abc", TESTS_DATA_DIR).c_str()))
    {
        return 1;
    }

    SCOPED_PROFILE_START(stdromano::ProfileUnit::Seconds, scene_loading);

    SceneGraphNode* mesh = SceneGraphNodesManager::get_instance().create_node("mesh");
    engine.get_scene_graph().add_node(mesh);

    SceneGraphNode* camera = SceneGraphNodesManager::get_instance().create_node("camera");
    engine.get_scene_graph().add_node(camera);
    camera->get_parameter("posz")->set_float(5.0f);
    camera->get_parameter("posy")->set_float(0.5f);

    SceneGraphNode* merge = SceneGraphNodesManager::get_instance().create_node("merge");
    engine.get_scene_graph().add_node(merge);

    engine.get_scene_graph().connect_nodes(mesh->get_id(), merge->get_id(), 0);
    engine.get_scene_graph().connect_nodes(camera->get_id(), merge->get_id(), 1);
    engine.get_scene_graph().connect_nodes(merge->get_id(), 0, 0);

    engine.prepare_for_rendering();

    SCOPED_PROFILE_STOP(scene_loading);

    engine.render_sample(nullptr);

    if(!engine.get_renderbuffer()->to_jpg("test_render_scenegraph_gpu.jpg"))
    {
        return 1;
    }

    return 0;
}
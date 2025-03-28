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

    SceneGraphNode* mesh = engine.get_scene_graph().create_node("mesh");
    mesh->get_parameter("path_pattern")->set_string("shaderball");

    SceneGraphNode* point_cloud = engine.get_scene_graph().create_node("mesh");
    point_cloud->get_parameter("path_pattern")->set_string("walls");

    SceneGraphNode* instancer = engine.get_scene_graph().create_node("instancer");

    SceneGraphNode* camera = engine.get_scene_graph().create_node("camera");
    camera->get_parameter("posz")->set_float(5.0f);
    camera->get_parameter("posy")->set_float(0.5f);

    SceneGraphNode* merge = engine.get_scene_graph().create_node("merge");

    engine.get_scene_graph().connect_nodes(mesh->get_id(), instancer->get_id(), 0);
    engine.get_scene_graph().connect_nodes(point_cloud->get_id(), instancer->get_id(), 1);
    engine.get_scene_graph().connect_nodes(instancer->get_id(), merge->get_id(), 0);
    engine.get_scene_graph().connect_nodes(camera->get_id(), merge->get_id(), 1);
    engine.get_scene_graph().connect_nodes(merge->get_id(), 0, 0);

    engine.prepare_for_rendering();

    SCOPED_PROFILE_STOP(scene_loading);

    engine.render_sample(nullptr);

    if(!engine.get_renderbuffer()->to_jpg("test_render_instancing_gpu.jpg"))
    {
        return 1;
    }

    return 0;
}
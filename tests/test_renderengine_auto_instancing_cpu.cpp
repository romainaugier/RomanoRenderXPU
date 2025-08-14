#include "romanorender/renderengine.h"
#include "romanorender/scenegraph.h"

#include "stdromano/random.hpp"
#include "stdromano/threading.hpp"

#define STDROMANO_ENABLE_PROFILING
#include "stdromano/profiling.hpp"

#include <mutex>

using namespace romanorender;

int main()
{
    stdromano::set_log_level(stdromano::LogLevel::Debug);

    constexpr uint32_t xres = 1280;
    constexpr uint32_t yres = 720;

    RenderEngine engine(xres, yres, true);

    if(!objects_from_abc_file(stdromano::StringD("{}/cornell_box_shaderball.abc", TESTS_DATA_DIR).c_str()))
    {
        return 1;
    }

    SCOPED_PROFILE_START(stdromano::ProfileUnit::Seconds, scene_loading);

    SceneGraphNode* mesh = engine.get_scene_graph().create_node("mesh");
    mesh->get_parameter("path_pattern")->set_string("shaderball");

    SceneGraphNode* attributes = engine.get_scene_graph().create_node("attributes");
    attributes->get_parameter("smooth_normals")->set_bool(true);
    attributes->get_parameter("subdivision_level")->set_int(1);
    attributes->set_input(mesh, 0);

    SceneGraphNode* set_transform = engine.get_scene_graph().create_node("set_transform");
    set_transform->get_parameter("posx")->set_float(2.0f);
    set_transform->set_input(attributes, 0);

    SceneGraphNode* merge_meshes = engine.get_scene_graph().create_node("merge");
    merge_meshes->set_input(mesh, 0);
    merge_meshes->set_input(set_transform, 1);

    SceneGraphNode* camera = engine.get_scene_graph().create_node("camera");
    camera->get_parameter("posz")->set_float(5.0f);
    camera->get_parameter("posy")->set_float(0.5f);

    SceneGraphNode* merge_cam_meshes = engine.get_scene_graph().create_node("merge");
    merge_cam_meshes->set_input(merge_meshes, 0);
    merge_cam_meshes->set_input(camera, 1);

    const uint32_t dome_light_uuid = objects_manager().add_light(LightType_Dome);
    SceneGraphNode* dome_light = engine.get_scene_graph().create_node("dome_light");
    dome_light->get_parameter("__light_uuid")->set_int(dome_light_uuid);

    SceneGraphNode* merge_all = engine.get_scene_graph().create_node("merge");
    merge_all->set_input(merge_cam_meshes, 0);
    merge_all->set_input(dome_light, 1);

    engine.get_scene_graph().get_output_node()->set_input(merge_all, 0);

    engine.prepare_for_rendering();

    SCOPED_PROFILE_STOP(scene_loading);

    engine.render_sample(integrator_pathtrace);

    if(!engine.get_renderbuffer()->to_jpg("test_render_auto_instancing_cpu.jpg"))
    {
        return 1;
    }

    return 0;
}
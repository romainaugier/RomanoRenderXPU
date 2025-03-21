#include "romanorender/scenegraph.h"

#include "stdromano/logger.h"

#include <regex>

ROMANORENDER_NAMESPACE_BEGIN

class ROMANORENDER_API SceneGraphNode_Output : public SceneGraphNode
{
public:
    SceneGraphNode_Output() : SceneGraphNode(1, 0) {}

    virtual const char* get_input_name(const uint32_t input) const noexcept override
    {
        return "objects";
    }

    virtual const char* get_type_name() const noexcept override { return "__output"; }

    virtual bool execute() override
    {
        for(const SceneGraphNode* input : this->get_inputs())
        {
            for(const Object* object : input->get_objects())
            {
                this->get_objects().emplace_back(object->reference());
            }
        }

        return true;
    }
};

class ROMANORENDER_API SceneGraphNode_Mesh : public SceneGraphNode
{
public:
    SceneGraphNode_Mesh() : SceneGraphNode(0)
    {
        this->add_parameter("path_pattern", ParameterType_String, ".*");
    }

    virtual const char* get_input_name(const uint32_t input) const noexcept override { return ""; }

    virtual const char* get_type_name() const noexcept override { return "mesh"; }

    virtual bool execute() override
    {
        std::regex path_regex(this->get_parameter("path_pattern")->get_string().c_str());

        for(const Object* object : ObjectsManager::get_instance().get_objects())
        {
            if(const ObjectMesh* mesh = dynamic_cast<const ObjectMesh*>(object))
            {
                std::cmatch cm;

                if(std::regex_search(object->get_path().c_str(), cm, path_regex))
                {
                    this->get_objects().emplace_back(object->reference());
                }
            }
        }

        return true;
    }
};

class ROMANORENDER_API SceneGraphNode_Camera : public SceneGraphNode
{
public:
    SceneGraphNode_Camera() : SceneGraphNode(0)
    {
        this->add_parameter("name_pattern", ParameterType_String, ".*");

        this->add_parameter("focal", ParameterType_Float, 50.0f);

        this->add_parameter("posx", ParameterType_Float, 0.0f);
        this->add_parameter("posy", ParameterType_Float, 0.0f);
        this->add_parameter("posz", ParameterType_Float, 0.0f);

        this->add_parameter("rotx", ParameterType_Float, 0.0f);
        this->add_parameter("roty", ParameterType_Float, 0.0f);
        this->add_parameter("rotz", ParameterType_Float, 0.0f);
    }

    virtual const char* get_input_name(const uint32_t input) const noexcept override { return ""; }

    virtual const char* get_type_name() const noexcept override { return "camera"; }

    virtual bool execute() override
    {
        std::regex path_regex(this->get_parameter("name_pattern")->get_string().c_str());

        Object* object = nullptr;
        ObjectsMatchingPatternIterator it = 0;
        ObjectCamera* camera = nullptr;

        while(objects_manager().get_objects_matching_pattern(it, path_regex, &object))
        {
            if(const ObjectCamera* cam = dynamic_cast<const ObjectCamera*>(object))
            {
                camera = const_cast<ObjectCamera*>(cam);
                break;
            }
        }

        if(camera == nullptr)
        {
            this->set_error(stdromano::String<>("Cannot find any camera matching pattern: {}",
                            this->get_parameter("name_pattern")->get_string()));
            return false;
        }

        ObjectCamera* node_camera = camera->reference();

        node_camera->set_focal(this->get_parameter("focal")->get_float());

        const Vec3F t(this->get_parameter("posx")->get_float(),
                      this->get_parameter("posy")->get_float(),
                      this->get_parameter("posz")->get_float());

        const Vec3F r(this->get_parameter("rotx")->get_float(),
                      this->get_parameter("roty")->get_float(),
                      this->get_parameter("rotz")->get_float());

        const Mat44F transform = Mat44F::from_trs(t, r, Vec3F(1.0f));

        node_camera->set_transform(transform);

        this->get_objects().emplace_back(node_camera);

        return true;
    }
};

class ROMANORENDER_API SceneGraphNode_Merge : public SceneGraphNode
{
public:
    SceneGraphNode_Merge() : SceneGraphNode(2) {}

    virtual const char* get_input_name(const uint32_t input) const noexcept override
    {
        switch(input)
        {
        case 0:
            return "objects to merge";
        case 1:
            return "objects to merge";
        default:
            return "";
        }

        return "";
    }

    virtual const char* get_type_name() const noexcept override { return "merge"; }

    virtual bool execute() override
    {
        for(const SceneGraphNode* input : this->get_inputs())
        {
            for(const Object* object : input->get_objects())
            {
                this->get_objects().emplace_back(object->reference());
            }
        }

        return true;
    }
};

void register_builtin_nodes(SceneGraphNodesManager& manager) noexcept
{
    manager.register_node_type(stdromano::String<>::make_ref("__output", 8),
                               []() -> SceneGraphNode* { return new SceneGraphNode_Output; });

    manager.register_node_type(stdromano::String<>::make_ref("mesh", 4),
                               []() -> SceneGraphNode* { return new SceneGraphNode_Mesh; });

    manager.register_node_type(stdromano::String<>::make_ref("camera", 6),
                               []() -> SceneGraphNode* { return new SceneGraphNode_Camera; });

    manager.register_node_type(stdromano::String<>::make_ref("merge", 5),
                               []() -> SceneGraphNode* { return new SceneGraphNode_Merge; });
}

ROMANORENDER_NAMESPACE_END
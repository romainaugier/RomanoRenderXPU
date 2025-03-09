#include "romanorender/scenegraph.h"

#include <regex>

ROMANORENDER_NAMESPACE_BEGIN

class ROMANORENDER_API SceneGraphNode_Output : public SceneGraphNode
{
public:
    SceneGraphNode_Output() : SceneGraphNode(1) {}

    virtual const char* get_input_name(const uint32_t input) const noexcept override { return "Objects"; }

    virtual const char* get_type_name() const noexcept override { return "output"; }

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
    SceneGraphNode_Mesh() : SceneGraphNode(0) {}

    virtual const char* get_input_name(const uint32_t input) const noexcept override { return ""; }

    virtual const char* get_type_name() const noexcept override { return "mesh"; }

    virtual bool execute() override
    {
        std::regex path_regex(".*");

        for(const Object* object : ObjectsManager::get_instance().get_objects())
        {
            std::cmatch cm;

            if(std::regex_search(object->get_path().c_str(), cm, path_regex))
            {
                this->get_objects().emplace_back(object->reference());
            }
        }

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
            return "Objects to merge";
        case 1:
            return "Objects to merge";
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

    manager.register_node_type(stdromano::String<>::make_ref("merge", 5),
                               []() -> SceneGraphNode* { return new SceneGraphNode_Merge; });
}

ROMANORENDER_NAMESPACE_END
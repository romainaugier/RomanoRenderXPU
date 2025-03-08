#include "romanorender/scenegraph.h"

#include <regex>

ROMANORENDER_NAMESPACE_BEGIN

class ROMANORENDER_API SceneGraphNode_Mesh : public SceneGraphNode
{
public:
    SceneGraphNode_Mesh() {}

    virtual uint32_t get_num_inputs() const noexcept override { return 0; }

    virtual const char* get_type_name() const noexcept override { return "mesh"; }

    virtual bool execute() override
    {
        std::regex path_regex("test");

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

void register_builtin_nodes() noexcept
{
    SceneGraphNodesManager::get_instance().register_node_type(
        stdromano::String<>::make_ref("mesh", 4), []() -> SceneGraphNode* { return new SceneGraphNode_Mesh; });
}

ROMANORENDER_NAMESPACE_END
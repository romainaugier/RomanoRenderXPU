#include "romanorender/scenegraph.h"

#include "stdromano/logger.hpp"

#include <queue>

ROMANORENDER_NAMESPACE_BEGIN

/* Parameters */

Parameter::Parameter(const char* name, ParameterType type) : _type(type), _name(name)
{
    switch(this->_type)
    {
    case ParameterType_Int:
        this->_data._int = 0;
        this->_is_inline = true;
        break;
    case ParameterType_Float:
        this->_data._float = 0.0f;
        this->_is_inline = true;
        break;
    case ParameterType_String:
        this->_data._ptr = new stdromano::String<>();
        this->_is_inline = false;
        break;
    case ParameterType_Bool:
        this->_data._bool = false;
        this->_is_inline = true;
        break;
    }
}

Parameter::Parameter(const char* name, ParameterType type, int default_value)
    : _type(type), _name(name), _is_inline(true)
{
    if(this->_type == ParameterType_Int)
    {
        this->_data._int = default_value;
    }
    else
    {
        switch(this->_type)
        {
        case ParameterType_Int:
            this->_data._int = 0;
            break;
        case ParameterType_Float:
            this->_data._float = 0.0f;
            break;
        case ParameterType_String:
            this->_data._ptr = new stdromano::String<>();
            this->_is_inline = false;
            break;
        case ParameterType_Bool:
            this->_data._bool = false;
            break;
        }
    }
}

Parameter::Parameter(const char* name, ParameterType type, float default_value)
    : _type(type), _name(name), _is_inline(true)
{
    if(this->_type == ParameterType_Float)
    {
        this->_data._float = default_value;
    }
    else
    {
        switch(this->_type)
        {
        case ParameterType_Int:
            this->_data._int = 0;
            break;
        case ParameterType_Float:
            this->_data._float = 0.0f;
            break;
        case ParameterType_String:
            this->_data._ptr = new stdromano::String<>();
            this->_is_inline = false;
            break;
        case ParameterType_Bool:
            this->_data._bool = false;
            break;
        }
    }
}

Parameter::Parameter(const char* name, ParameterType type, bool default_value)
    : _type(type), _name(name), _is_inline(true)
{
    if(this->_type == ParameterType_Bool)
    {
        this->_data._bool = default_value;
    }
    else
    {
        switch(this->_type)
        {
        case ParameterType_Int:
            this->_data._int = 0;
            break;
        case ParameterType_Float:
            this->_data._float = 0.0f;
            break;
        case ParameterType_String:
            this->_data._ptr = new stdromano::String<>();
            this->_is_inline = false;
            break;
        case ParameterType_Bool:
            this->_data._bool = false;
            break;
        }
    }
}

Parameter::Parameter(const char* name, ParameterType type, const stdromano::String<>& default_value)
    : _type(type), _name(name)
{
    if(this->_type == ParameterType_String)
    {
        this->_data._ptr = new stdromano::String<>(default_value);
        this->_is_inline = false;
    }
    else
    {
        switch(this->_type)
        {
        case ParameterType_Int:
            this->_data._int = 0;
            this->_is_inline = true;
            break;
        case ParameterType_Float:
            this->_data._float = 0.0f;
            this->_is_inline = true;
            break;
        case ParameterType_String:
            this->_data._ptr = new stdromano::String<>();
            this->_is_inline = false;
            break;
        case ParameterType_Bool:
            this->_data._bool = false;
            this->_is_inline = true;
            break;
        }
    }
}

Parameter::Parameter(const char* name, ParameterType type, const char* default_value)
    : _type(type), _name(name)
{
    if(this->_type == ParameterType_String)
    {
        this->_data._ptr = new stdromano::String<>(default_value);
        this->_is_inline = false;
    }
    else
    {
        switch(this->_type)
        {
        case ParameterType_Int:
            this->_data._int = 0;
            this->_is_inline = true;
            break;
        case ParameterType_Float:
            this->_data._float = 0.0f;
            this->_is_inline = true;
            break;
        case ParameterType_String:
            this->_data._ptr = new stdromano::String<>();
            this->_is_inline = false;
            break;
        case ParameterType_Bool:
            this->_data._bool = false;
            this->_is_inline = true;
            break;
        }
    }
}

Parameter::~Parameter() noexcept
{
    if(!this->_is_inline && this->_data._ptr)
    {
        switch(this->_type)
        {
        case ParameterType_String:
            delete static_cast<stdromano::String<>*>(this->_data._ptr);
            break;
        default:
            break;
        }
        this->_data._ptr = nullptr;
    }
}

Parameter::Parameter(Parameter&& other) noexcept : _type(other._type),
                                                   _name(std::move(other._name)),
                                                   _is_inline(other._is_inline)
{

    this->_data = other._data;

    if(!this->_is_inline)
    {
        other._data._ptr = nullptr;
    }
}

Parameter& Parameter::operator=(Parameter&& other) noexcept
{
    if(this != &other)
    {
        if(!this->_is_inline && this->_data._ptr)
        {
            switch(this->_type)
            {
            case ParameterType_String:
                delete static_cast<stdromano::String<>*>(this->_data._ptr);
                break;
            default:
                break;
            }
        }

        this->_type = other._type;
        this->_name = std::move(other._name);
        this->_is_inline = other._is_inline;
        this->_data = other._data;

        if(!this->_is_inline)
        {
            other._data._ptr = nullptr;
        }
    }
    return *this;
}

bool Parameter::set_int(int value) noexcept
{
    if(this->_type == ParameterType_Int && this->_is_inline)
    {
        this->_data._int = value;

        if(this->_parent != nullptr)
        {
            this->_parent->set_dirty();
        }

        return true;
    }

    return false;
}

bool Parameter::set_float(float value) noexcept
{
    if(this->_type == ParameterType_Float && this->_is_inline)
    {
        this->_data._float = value;

        if(this->_parent != nullptr)
        {
            this->_parent->set_dirty();
        }

        return true;
    }

    return false;
}

bool Parameter::set_bool(bool value) noexcept
{
    if(this->_type == ParameterType_Bool && this->_is_inline)
    {
        this->_data._bool = value;

        if(this->_parent != nullptr)
        {
            this->_parent->set_dirty();
        }

        return true;
    }

    return false;
}

bool Parameter::set_string(const stdromano::String<>& value) noexcept
{
    if(this->_type == ParameterType_String && !this->_is_inline && this->_data._ptr)
    {
        *static_cast<stdromano::String<>*>(this->_data._ptr) = value;
        if(this->_parent != nullptr)
        {
        }
        this->_parent->set_dirty();
        return true;
    }

    return false;
}

bool Parameter::set_string(const char* value) noexcept
{
    if(this->_type == ParameterType_String && !this->_is_inline && this->_data._ptr)
    {
        *static_cast<stdromano::String<>*>(this->_data._ptr) = value;

        if(this->_parent != nullptr)
        {
            this->_parent->set_dirty();
        }

        return true;
    }

    return false;
}

/* SceneGraphNode */

SceneGraphNode::~SceneGraphNode() { this->clear(); }

void SceneGraphNode::clear() noexcept
{
    for(Object* object : this->_objects)
    {
        delete object;
    }

    this->_objects.clear();
}

void SceneGraphNode::prepare_objects() noexcept { this->clear(); }

size_t SceneGraphNode::get_memory_usage() const noexcept
{
    size_t memory_usage = 0;

    for(const Object* obj : this->_objects)
    {
        memory_usage += obj->get_memory_usage();
    }

    return memory_usage;
}

void SceneGraphNode::set_dirty() noexcept
{
    this->_dirty = true;

    for(SceneGraphNode* out : this->_outputs)
    {
        out->set_dirty();
    }

    if(this->_parent != nullptr)
    {
        this->_parent->set_dirty();
    }
}

/* SceneGraph */

SceneGraph::SceneGraph()
{
    this->_output_node = SceneGraphNodesManager::get_instance().create_node("__output");
    this->_output_node->set_name("output");
    this->add_node(this->_output_node);
}

SceneGraph::~SceneGraph()
{
    for(SceneGraphNode* node : this->_nodes)
    {
        delete node;
    }
}

void SceneGraph::add_node(SceneGraphNode* node) noexcept
{
    node->set_id(this->_id_counter++);

    if(node->get_name().empty())
    {
        node->set_name(node->get_type_name());
    }

    node->_parent = this;

    this->_nodes.push_back(node);

    this->set_dirty();
}

SceneGraphNode* SceneGraph::create_node(const stdromano::String<>& type_name) noexcept
{
    SceneGraphNode* node = SceneGraphNodesManager::get_instance().create_node(type_name);

    if(node != nullptr)
    {
        this->add_node(node);
    }

    return node;
}

void SceneGraph::remove_node(const uint32_t node_id) noexcept
{
    SceneGraphNode* node_to_delete = nullptr;

    for(SceneGraphNode* node : this->_nodes)
    {
        if(node->get_id() == node_id)
        {
            node_to_delete = node;
            break;
        }
    }

    if(node_to_delete != nullptr && node_to_delete->get_type_name()[0] != '_')
    {
        this->remove_node(node_to_delete);
    }
}

void SceneGraph::remove_node(SceneGraphNode* node) noexcept
{
    for(SceneGraphNode* in : node->get_inputs())
    {
        if(in != nullptr)
        {
            in->remove_output(node);
        }
    }

    for(SceneGraphNode* out : node->get_outputs())
    {
        for(uint32_t i = 0; i < out->get_inputs().size(); i++)
        {
            if(out->get_inputs()[i] == node)
            {
                out->get_inputs()[i] = nullptr;
            }
        }
    }

    const auto& it = this->_nodes.cfind(node);

    this->_nodes.erase(it);

    delete node;

    this->set_dirty();
}

SceneGraphNode* SceneGraph::get_node_by_id(const uint32_t id) noexcept
{
    for(SceneGraphNode* node : this->_nodes)
    {
        if(node->get_id() == id)
        {
            return node;
        }
    }

    return nullptr;
}

void SceneGraph::connect_nodes(const uint32_t lhs, const uint32_t rhs, const uint32_t input) noexcept
{
    SceneGraphNode* lhs_ptr = nullptr;
    SceneGraphNode* rhs_ptr = nullptr;

    for(SceneGraphNode* node : this->_nodes)
    {
        if(node->get_id() == lhs)
        {
            lhs_ptr = node;
        }

        if(node->get_id() == rhs)
        {
            rhs_ptr = node;
        }

        if(lhs_ptr != nullptr && rhs_ptr != nullptr)
        {
            break;
        }
    }

    if(lhs_ptr != nullptr && rhs_ptr != nullptr)
    {
        this->connect_nodes(lhs_ptr, rhs_ptr, input);
    }
}

void SceneGraph::connect_nodes(SceneGraphNode* lhs, SceneGraphNode* rhs, const uint32_t input) noexcept
{
    ROMANORENDER_ASSERT(input < rhs->get_num_inputs(), "Not enough inputs on node");

    rhs->get_inputs()[input] = lhs;
    rhs->set_dirty();
    lhs->add_output(rhs);

    this->set_dirty();
}

using SceneGraphNodeBatch = stdromano::Vector<SceneGraphNode*>;

bool SceneGraph::execute() noexcept
{
    if(!this->is_dirty())
    {
        return true;
    }

    this->_memory_usage = 0;
    this->_error_node = nullptr;

    stdromano::HashMap<const SceneGraphNode*, uint32_t> in_degrees;

    for(SceneGraphNode* node : this->_nodes)
    {
        in_degrees[node] = node->get_num_inputs();
    }

    std::queue<SceneGraphNode*> current_level;

    for(SceneGraphNode* node : this->_nodes)
    {
        if(in_degrees[node] == 0)
        {
            current_level.push(node);
        }
    }

    stdromano::Vector<SceneGraphNodeBatch> sorted_batches;

    while(!current_level.empty())
    {
        SceneGraphNodeBatch batch;

        std::queue<SceneGraphNode*> next_level;

        while(!current_level.empty())
        {
            SceneGraphNode* node = current_level.front();
            current_level.pop();
            batch.push_back(node);
        }

        sorted_batches.emplace_back(std::move(batch));

        for(SceneGraphNode* node : sorted_batches.back())
        {
            for(SceneGraphNode* out : node->get_outputs())
            {
                if(--in_degrees[out] == 0)
                {
                    next_level.push(out);
                }
            }
        }

        current_level = next_level;
    }

    size_t sorted_nodes = 0;

    for(const SceneGraphNodeBatch& batch : sorted_batches)
    {
        sorted_nodes += batch.size();
    }

    if(sorted_nodes != this->_nodes.size())
    {
        return false;
    }

    stdromano::log_debug("Starting scenegraph execution");

    for(const SceneGraphNodeBatch& batch : sorted_batches)
    {
        for(SceneGraphNode* node : batch)
        {
            if(node->is_dirty())
            {
                node->prepare_objects();

                stdromano::log_debug("Executing node: {} ({})", node->get_name(), node->get_id());

                if(!node->execute())
                {
                    this->_error_node = node;

                    return false;
                }

                node->set_not_dirty();

                if(std::strcmp(node->get_type_name(), "camera") == 0)
                {
                    const Parameter* link_to_flying_camera_parm = node->get_parameter("link_to_flying_camera");

                    if(link_to_flying_camera_parm != nullptr && link_to_flying_camera_parm->get_bool())
                    {
                        this->_flying_camera_node = node;
                    }
                }
            }

            this->_memory_usage += node->get_memory_usage();
        }
    }

    char mem_usage_fmt[16];
    stdromano::format_byte_size((float)this->_memory_usage, mem_usage_fmt);

    stdromano::log_debug("Finished scenegraph execution ({} used by nodes)", mem_usage_fmt);

    this->set_not_dirty();

    return true;
}

const stdromano::Vector<Object*>* SceneGraph::get_result() const noexcept
{
    return this->_output_node == nullptr ? nullptr : &this->_output_node->get_objects();
}

/* SceneGraphNodesManager */

SceneGraphNodesManager::SceneGraphNodesManager() { register_builtin_nodes(*this); }

SceneGraphNodesManager::~SceneGraphNodesManager() {}

void SceneGraphNodesManager::register_node_type(const stdromano::String<>& type_name,
                                                std::function<SceneGraphNode*()>&& factory) noexcept
{
    this->_types.push_back(std::move(type_name));

    this->_factories[stdromano::String<>::make_ref(this->_types.back())] = std::move(factory);

    stdromano::log_debug("Registered new node type: {}", this->_types.back());
}

SceneGraphNode* SceneGraphNodesManager::create_node(const stdromano::String<>& type_name) noexcept
{
    const auto it = this->_factories.find(type_name);

    if(it == this->_factories.end())
    {
        return nullptr;
    }

    return it->second();
}

json serialize_graph(const SceneGraph& graph) noexcept
{
    json serialized_graph;

    serialized_graph["nodes"] = json::array();

    for(auto* node : graph.get_nodes())
    {
        json node_data;

        node_data["id"] = node->get_id();
        node_data["name"] = node->get_name().c_str();
        node_data["type"] = node->get_type_name();

        node_data["parameters"] = json::array();

        for(auto& param : node->get_parameters())
        {
            json param_data;

            param_data["name"] = param.get_name().c_str();
            param_data["type"] = static_cast<int>(param.get_type());

            switch(param.get_type())
            {
            case ParameterType_Int:
                param_data["value"] = param.get_int();
                break;
            case ParameterType_Float:
                param_data["value"] = param.get_float();
                break;
            case ParameterType_Bool:
                param_data["value"] = param.get_bool();
                break;
            case ParameterType_String:
                param_data["value"] = param.get_string().c_str();
                break;
            }

            node_data["parameters"].push_back(param_data);
        }

        node_data["inputs"] = json::array();

        for(auto* input : node->get_inputs())
        {
            node_data["inputs"].push_back(input == nullptr ? -1 : (int)input->get_id());
        }

        serialized_graph["nodes"].push_back(node_data);
    }

    return serialized_graph;
}

void deserialize_graph(const json& serialized_graph, SceneGraph& graph) noexcept
{
    stdromano::HashMap<uint32_t, SceneGraphNode*> nodes_map;

    for(const auto& node_data : serialized_graph["nodes"])
    {
        const uint32_t node_id = node_data["id"];
        const std::string type_name = node_data["type"];

        SceneGraphNode* node = SceneGraphNodesManager::get_instance().create_node(type_name.c_str());

        if(node)
        {
            node->set_name(node_data["name"].get<std::string>().c_str());

            for(const auto& paramData : node_data["parameters"])
            {
                const ParameterType type = static_cast<ParameterType>(paramData["type"].get<int>());
                const std::string param_name = paramData["name"];

                switch(type)
                {
                case ParameterType_Int:
                    node->add_parameter(param_name.c_str(), type, paramData["value"].get<int>());
                    break;
                case ParameterType_Float:
                    node->add_parameter(param_name.c_str(), type, paramData["value"].get<float>());
                    break;
                case ParameterType_Bool:
                    node->add_parameter(param_name.c_str(), type, paramData["value"].get<bool>());
                    break;
                case ParameterType_String:
                    node->add_parameter(param_name.c_str(),
                                        type,
                                        paramData["value"].get<std::string>().c_str());
                    break;
                }
            }

            graph.add_node(node);
            nodes_map[node_id] = node;
        }
    }

    for(const auto& node_data : serialized_graph["nodes"])
    {
        const uint32_t nodeId = node_data["id"];
        SceneGraphNode* node = nodes_map[nodeId];

        const auto& inputs = node_data["inputs"];

        for(size_t i = 0; i < inputs.size(); ++i)
        {
            const int inputNodeId = inputs[i];

            if(inputNodeId != -1)
            {
                graph.connect_nodes(nodes_map[inputNodeId], node, i);
            }
        }
    }
}

ROMANORENDER_NAMESPACE_END
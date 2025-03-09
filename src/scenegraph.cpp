#include "romanorender/scenegraph.h"

#include "stdromano/logger.h"

#include <queue>

ROMANORENDER_NAMESPACE_BEGIN

SceneGraphNode::~SceneGraphNode()
{
    for(Object* object : this->_objects)
    {
        delete object;
    }
}

void SceneGraphNode::prepare_objects() noexcept
{
    this->_objects.clear();

    for(const SceneGraphNode* input : this->_inputs)
    {
        for(const Object* object : input->_objects)
        {
            this->_objects.emplace_back(object->reference());
        }
    }
}

SceneGraph::SceneGraph()
{
    this->_output_node = SceneGraphNodesManager::get_instance().create_node("__output");
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

    this->_nodes.push_back(node);
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

    if(node_to_delete != nullptr)
    {
        delete node_to_delete;
    }
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
}

using SceneGraphNodeBatch = stdromano::Vector<SceneGraphNode*>;

bool SceneGraph::execute() noexcept
{
    stdromano::HashMap<const SceneGraphNode*, uint32_t> in_degrees;
    stdromano::HashMap<const SceneGraphNode*, stdromano::Vector<SceneGraphNode*> > outputs;

    for(SceneGraphNode* node : this->_nodes)
    {
        in_degrees[node] = node->get_num_inputs();

        for(const SceneGraphNode* input_node : node->get_inputs())
        {
            outputs[input_node].push_back(node);
        }
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
            for(SceneGraphNode* output : outputs[node])
            {
                if(--in_degrees[output] == 0)
                {
                    next_level.push(output);
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

    for(const SceneGraphNodeBatch& batch : sorted_batches)
    {
        for(SceneGraphNode* node : batch)
        {
            if(node->is_dirty())
            {
                node->prepare_objects();

                if(!node->execute())
                {
                    return false;
                }

                node->set_not_dirty();
            }
        }
    }

    return true;
}

const stdromano::Vector<Object*>* SceneGraph::get_result() const noexcept
{
    return this->_output_node == nullptr ? nullptr : &this->_output_node->get_objects();
}

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

ROMANORENDER_NAMESPACE_END
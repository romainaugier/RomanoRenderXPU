#include "romanorender/scenegraph.h"

#include "stdromano/logger.h"

#include <queue>

ROMANORENDER_NAMESPACE_BEGIN

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

    this->set_dirty();
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
                    return false;
                }

                node->set_not_dirty();
            }
        }
    }

    stdromano::log_debug("Finished scenegraph execution");

    this->set_not_dirty();

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
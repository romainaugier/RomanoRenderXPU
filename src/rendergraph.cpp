#include "romanorender/rendergraph.h"

#include "stdromano/logger.h"

#if defined(ROMANORENDER_WIN)
#include <Windows.h>
#else
#include <dlfcn.h>
#endif /* defined(ROMANORENDER_WIN) */

ROMANORENDER_NAMESPACE_BEGIN

NodeManager::~NodeManager()
{
    for(NodeHandle node : this->loaded_node_plugins)
    {
#if defined(ROMANORENDER_WIN)
        FreeLibrary(node.handle);
#else
        dlclose(node.handle);
#endif /* defined(ROMANORENDER_WIN) */
    }
}

bool NodeManager::load_node(const stdromano::String<>& path) noexcept
{
    NodeHandle node;

#if defined(ROMANORENDER_WIN)
    node.handle = LoadLibraryA(path.c_str());
#else
    node.handle = dlopen(path.c_str(), RTLD_LAZY);
#endif /* defined(ROMANORENDER_WIN) */

    if(node.handle == nullptr)
    {
        stdromano::log_error("Could not load node plugin: {}", path);
        return false;
    }

#if defined(ROMANORENDER_WIN)
    node_register_func register_node = (node_register_func)GetProcAddress(node.handle, "register_node");
#else
    node_register_func register_node = (node_register_func)dlsym(node.handle, "register_node");
#endif /* defined(ROMANORENDER_WIN) */

    if(register_node != nullptr)
    {
        register_node(*this);
        this->loaded_node_plugins.push_back(node);
        stdromano::log_debug("Loaded node plugin: {}", path);
        return true;
    }

#if defined(ROMANORENDER_WIN)
    FreeLibrary(node.handle);
#else
    dlclose(node.handle);
#endif /* defined(ROMANORENDER_WIN) */

    return false;
}

void NodeManager::register_node_type(const stdromano::String<>& type_name,
                                     render_graph_node_create_func factory_func) noexcept
{
    this->factories.insert(std::make_pair(type_name, factory_func));
}

void RenderGraph::sort() noexcept
{
    stdromano::HashMap<Node*, size_t> in_degree;
    std::vector<Node*> queue;

    for(Node* node : this->nodes) 
    {
        for(Node* dep : node->get_inputs()) 
        {
            in_degree[dep]++;
        }
    }

    for(Node* node : this->nodes) 
    {
        if(in_degree[node] == 0) 
        {
            queue.push_back(node);
        }
    }

    execution_order.clear();

    while(!queue.empty()) 
    {
        Node* current = queue.back();
        queue.pop_back();
        this->execution_order.push_back(current);

        for(Node* successor : current->get_inputs()) 
        {
            if(--in_degree[successor] == 0) 
            {
                queue.push_back(successor);
            }
        }
    }
}

ROMANORENDER_NAMESPACE_END
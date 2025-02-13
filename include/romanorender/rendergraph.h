#pragma once

#if !defined(__ROMANORENDER_RENDERGRAPH)
#define __ROMANORENDER_RENDERGRAPH

#include "romanorender/romanorender.h"
#include "romanorender/vec3.h"

#include "stdromano/string.h"
#include "stdromano/hashmap.h"
#include "stdromano/vector.h"

#include <variant>
#include <atomic>

ROMANORENDER_NAMESPACE_BEGIN

class Node;

class ROMANORENDER_API NodeManager
{
    struct NodeHandle
    {
    #if defined(ROMANORENDER_WIN)
            HMODULE handle;
    #else
            void* handle;
    #endif /* defined(ROMANORENDER_WIN) */
    };

    using render_graph_node_create_func = std::function<Node*()>; 
    using node_register_func = void(*)(NodeManager);

    stdromano::HashMap<stdromano::String<>, render_graph_node_create_func> factories;
    stdromano::Vector<NodeHandle> loaded_node_plugins;

    bool load_node(const stdromano::String<>& path) noexcept;

public:
    NodeManager() = default;

    ~NodeManager();

    void register_node_type(const stdromano::String<>& type_name,
                            render_graph_node_create_func factory_func) noexcept;

    Node* get_node(const stdromano::String<>& type_name) const noexcept
    {
        const auto it = this->factories.find(type_name);

        if(it == this->factories.end())
        {
            return nullptr;
        }

        return it->second();
    }
};

/* First 3 bits are for data types, and the rest for flags */
enum NodeDataBlockType_
{
    NodeDataBlockType_Invalid = 0,
    NodeDataBlockType_Geometry = 1,
    NodeDataBlockType_Transform = 2,
    NodeDataBlockType_Camera = 3,
    NodeDataBlockType_Light = 4,
    NodeDataBlockType_Material = 5,
    NodeDataBlockType_Entity = 6,

    NodeDataBlockType_List = 8,
};

class NodeDataBlock 
{
    struct SharedData 
    {
        void* ptr;
        size_t size;
        std::atomic_int refcount;
        uint32_t type;
    };

    SharedData* shared = nullptr;

public:
    NodeDataBlock() = default;
    
    NodeDataBlock(const NodeDataBlock& other) : shared(other.shared) 
    {
        if(this->shared != nullptr) 
        {
            ++this->shared->refcount;
        }
    }

    NodeDataBlock(NodeDataBlock&& other) : shared(other.shared)
    {
        other.shared = nullptr;
    }

    NodeDataBlock& operator=(NodeDataBlock&& other) noexcept
    {
        this->shared = other.shared;
        other.shared = nullptr;

        return *this;
    }

    NodeDataBlock(void* data, size_t size, uint32_t type) :
        shared(new SharedData{data, size, 1, type}) {}

    ~NodeDataBlock() 
    {
        if(this->shared && --this->shared->refcount == 0) 
        {
            delete[] static_cast<char*>(this->shared->ptr);
            delete this->shared;

            this->shared = nullptr;
        }
    }

    NodeDataBlock duplicate() const noexcept
    {
        void* new_data = new char[this->shared->size];
        std::memcpy(new_data, this->shared->ptr, this->shared->size);
        return std::move(NodeDataBlock(new_data, this->shared->size, this->shared->type));
    }

    uint32_t get_type() const noexcept
    {
        if(this->shared == nullptr)
        {
            return 0;
        }

        return this->shared->type;
    }

    void* get_data() const noexcept
    {
        if(this->shared == nullptr)
        {
            return 0;
        }

        return this->shared->ptr;
    }

    size_t get_data_size() const noexcept
    {
        if(this->shared == nullptr)
        {
            return 0;
        }

        return this->shared->size;
    }
};

using ParameterValue = std::variant<uint32_t, 
                                    float, 
                                    stdromano::String<>,
                                    Vec3F>;

class ROMANORENDER_API Node 
{
protected:
    stdromano::HashMap<stdromano::String<>, ParameterValue> parameters;
    stdromano::Vector<Node*> inputs;

    NodeDataBlock data_block;

    size_t num_inputs = 0;

    bool has_error;
    stdromano::String<> error_string;

public:
    virtual ~Node() = default;

    virtual void compute() = 0;

    template<typename T>
    void set_parameter(const stdromano::String<>& name, T&& value) 
    {
        this->parameters.insert(std::make_pair(name, std::forward<T>(value)));
    }

    template<typename T>
    const T& get_parameter(const stdromano::String<>& name) const 
    {
        return std::get<T>(this->parameters.find(name)->second);
    }

    void set_num_inputs(const size_t num_inputs) noexcept
    {
        this->num_inputs = num_inputs;
    }

    size_t get_num_inputs() const noexcept
    {
        return this->num_inputs;
    }

    void set_input(const size_t input_num, Node* node) 
    {
        this->inputs.push_back(node);
    }

    Node* get_input(const size_t input_num) const noexcept
    {
        return input_num >= this->inputs.size() ? nullptr : this->inputs[input_num];
    }

    const stdromano::Vector<Node*>& get_inputs() const noexcept
    {
        return this->inputs;
    }

    void duplicate_input_data(const size_t input_num) noexcept
    {
        this->data_block = std::move(this->get_input(input_num)->get_data_block().duplicate());
    }

    void clear_data_block() noexcept
    {
        this->data_block.~NodeDataBlock();
    }

    void set_data_block(void* data, size_t size, uint32_t type) noexcept
    {
        this->data_block.~NodeDataBlock();

        this->data_block = std::move(NodeDataBlock(data, size, type));
    }

    NodeDataBlock get_data_block() const noexcept
    {
        return this->data_block;
    }

    void set_error(const char* error_string) noexcept
    {
        this->has_error = true;
        this->error_string = stdromano::String<>::make_ref(error_string, std::strlen(error_string));
    }
};

class ROMANORENDER_API RenderGraph 
{
    std::vector<Node*> nodes;
    std::vector<Node*> execution_order;

    void sort() noexcept;

public:
    void add_node(Node* node) 
    {
        this->nodes.push_back(node);
    }

    void execute() 
    {
        this->sort();
        
        for(Node* node : execution_order) 
        {
            node->compute();
        }
    }
};

/* Example */
class GeometryNode : public Node 
{
public:
    void compute() override 
    {
        float scale = this->get_parameter<float>("scale");
        Vec3F offset = this->get_parameter<Vec3F>("offset");
    }
};

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_RENDERGRAPH) */
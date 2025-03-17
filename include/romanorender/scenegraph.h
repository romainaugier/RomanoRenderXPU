#pragma once

#if !defined(__ROMANORENDER_SCENEGRAPH)
#define __ROMANORENDER_SCENEGRAPH

#include "romanorender/object.h"

ROMANORENDER_NAMESPACE_BEGIN

class ROMANORENDER_API SceneGraphNode
{
    stdromano::Vector<Object*> _objects;
    stdromano::Vector<SceneGraphNode*> _inputs;
    stdromano::Vector<SceneGraphNode*> _outputs;

    uint32_t _id;
    stdromano::String<> _name;

    uint32_t _num_outputs = 0;

    bool _dirty = true;

    void clear() noexcept;

    void prepare_objects() noexcept;

    void set_id(const uint32_t id) noexcept { this->_id = id; }

public:
    friend class SceneGraph;

    SceneGraphNode(const uint32_t num_inputs, const uint32_t num_outputs = 1)
    {
        for(uint32_t i = 0; i < num_inputs; i++)
        {
            this->_inputs.push_back(nullptr);
        }

        this->_num_outputs = num_outputs;
    }

    ~SceneGraphNode();

    ROMANORENDER_FORCE_INLINE bool is_dirty() const noexcept { return this->_dirty; }

    void set_dirty() noexcept
    {
        this->_dirty = true;

        for(SceneGraphNode* out : this->_outputs)
        {
            out->set_dirty();
        }
    }

    ROMANORENDER_FORCE_INLINE void set_not_dirty() noexcept { this->_dirty = false; }

    virtual bool execute() = 0;

    uint32_t get_num_inputs() const noexcept { return this->_inputs.size(); };

    virtual const char* get_input_name(const uint32_t input) const noexcept = 0;

    virtual const char* get_type_name() const noexcept = 0;

    ROMANORENDER_FORCE_INLINE uint32_t get_id() const noexcept { return this->_id; }

    ROMANORENDER_FORCE_INLINE const char* get_name() const noexcept { return this->_name.c_str(); }

    ROMANORENDER_FORCE_INLINE stdromano::Vector<Object*>& get_objects() noexcept
    {
        return this->_objects;
    }

    ROMANORENDER_FORCE_INLINE const stdromano::Vector<Object*>& get_objects() const noexcept
    {
        return this->_objects;
    }

    ROMANORENDER_FORCE_INLINE stdromano::Vector<SceneGraphNode*>& get_inputs() noexcept
    {
        return this->_inputs;
    }

    ROMANORENDER_FORCE_INLINE stdromano::Vector<SceneGraphNode*> get_outputs() noexcept
    {
        return this->_outputs;
    }

    ROMANORENDER_FORCE_INLINE uint32_t get_num_outputs() const noexcept
    {
        return this->_num_outputs;
    }

    ROMANORENDER_FORCE_INLINE void add_output(SceneGraphNode* output) noexcept
    {
        this->_outputs.push_back(output);
    }

    ROMANORENDER_FORCE_INLINE void remove_output(SceneGraphNode* output) noexcept
    {
        const auto& it = this->_outputs.cfind(output);
        this->_outputs.erase(it);
    }

    ROMANORENDER_FORCE_INLINE void set_name(stdromano::String<>&& name) noexcept
    {
        this->_name = std::move(name);
    }
};

class ROMANORENDER_API SceneGraph
{
    stdromano::Vector<SceneGraphNode*> _nodes;

    SceneGraphNode* _output_node = nullptr;

    uint32_t _id_counter = 0;

    bool _is_dirty = false;

public:
    SceneGraph();

    ~SceneGraph();

    ROMANORENDER_FORCE_INLINE bool is_dirty() const noexcept { return this->_is_dirty; }

    ROMANORENDER_FORCE_INLINE void set_dirty() noexcept { this->_is_dirty = true; }

    ROMANORENDER_FORCE_INLINE void set_not_dirty() noexcept { this->_is_dirty = false; }

    void add_node(SceneGraphNode* node) noexcept;

    void remove_node(const uint32_t node_id) noexcept;

    void remove_node(SceneGraphNode* node) noexcept;

    ROMANORENDER_FORCE_INLINE stdromano::Vector<SceneGraphNode*>& get_nodes() noexcept
    {
        return this->_nodes;
    }

    void connect_nodes(const uint32_t lhs, const uint32_t rhs, const uint32_t input) noexcept;

    void connect_nodes(SceneGraphNode* lhs, SceneGraphNode* rhs, const uint32_t input) noexcept;

    bool execute() noexcept;

    const stdromano::Vector<Object*>* get_result() const noexcept;
};

class ROMANORENDER_API SceneGraphNodesManager
{
public:
    static SceneGraphNodesManager& get_instance()
    {
        static SceneGraphNodesManager myInstance;

        return myInstance;
    }

    SceneGraphNodesManager(SceneGraphNodesManager const&) = delete;
    SceneGraphNodesManager(SceneGraphNodesManager&&) = delete;
    SceneGraphNodesManager& operator=(SceneGraphNodesManager const&) = delete;
    SceneGraphNodesManager& operator=(SceneGraphNodesManager&&) = delete;

    void register_node_type(const stdromano::String<>& type_name, std::function<SceneGraphNode*()>&& factory) noexcept;

    SceneGraphNode* create_node(const stdromano::String<>& type_name) noexcept;

    const stdromano::Vector<stdromano::String<> >& get_types() const noexcept
    {
        return this->_types;
    }

private:
    SceneGraphNodesManager();

    ~SceneGraphNodesManager();

    stdromano::HashMap<stdromano::String<>, std::function<SceneGraphNode*()> > _factories;
    stdromano::Vector<stdromano::String<> > _types;
};

ROMANORENDER_API void register_builtin_nodes(SceneGraphNodesManager& manager) noexcept;

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_SCENEGRAPH) */
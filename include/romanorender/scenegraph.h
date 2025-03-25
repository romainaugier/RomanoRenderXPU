#pragma once

#if !defined(__ROMANORENDER_SCENEGRAPH)
#define __ROMANORENDER_SCENEGRAPH

#include "romanorender/object.h"

#include "nlohmann/json.hpp"

ROMANORENDER_NAMESPACE_BEGIN

class SceneGraphNode;
class SceneGraph;

enum ParameterType : uint32_t
{
    ParameterType_Int = 0,
    ParameterType_Float = 1,
    ParameterType_String = 2,
    ParameterType_Bool = 3
};

class ROMANORENDER_API Parameter
{
public:
    friend class SceneGraphNode;

private:
    union
    {
        void* _ptr;
        int _int;
        float _float;
        bool _bool;
    } _data;

    ParameterType _type;
    stdromano::String<> _name;
    bool _is_inline;

    SceneGraphNode* _parent = nullptr;

public:
    Parameter(const char* name, ParameterType type);

    Parameter(const char* name, ParameterType type, int default_value);

    Parameter(const char* name, ParameterType type, float default_value);

    Parameter(const char* name, ParameterType type, bool default_value);

    Parameter(const char* name, ParameterType type, const stdromano::String<>& default_value);

    Parameter(const char* name, ParameterType type, const char* default_value);

    ~Parameter() noexcept;

    Parameter(const Parameter&) = delete;
    Parameter& operator=(const Parameter&) = delete;

    Parameter(Parameter&& other) noexcept;
    Parameter& operator=(Parameter&& other) noexcept;

    ROMANORENDER_FORCE_INLINE const stdromano::String<>& get_name() const noexcept
    {
        return this->_name;
    }

    ROMANORENDER_FORCE_INLINE ParameterType get_type() const noexcept { return this->_type; }

    ROMANORENDER_FORCE_INLINE int get_int(const int default_value = 0) const noexcept
    {
        return (this->_type == ParameterType_Int && this->_is_inline) ? this->_data._int : default_value;
    }

    ROMANORENDER_FORCE_INLINE float get_float(const float default_value = 0.0f) const noexcept
    {
        return (this->_type == ParameterType_Float && this->_is_inline) ? this->_data._float : default_value;
    }

    ROMANORENDER_FORCE_INLINE bool get_bool(const bool default_value = false) const noexcept
    {
        return (this->_type == ParameterType_Bool && this->_is_inline) ? this->_data._bool : default_value;
    }

    ROMANORENDER_FORCE_INLINE const stdromano::String<>& get_string() const noexcept
    {
        static stdromano::String<> empty_string;

        return (this->_type == ParameterType_String && !this->_is_inline && this->_data._ptr != nullptr)
                   ? *static_cast<stdromano::String<>*>(this->_data._ptr)
                   : empty_string;
    }

    bool set_int(int value) noexcept;

    bool set_float(float value) noexcept;

    bool set_bool(bool value) noexcept;

    bool set_string(const stdromano::String<>& value) noexcept;

    bool set_string(const char* value) noexcept;
};

class ROMANORENDER_API SceneGraphNode
{
public:
    friend class SceneGraph;

private:
    stdromano::Vector<Object*> _objects;
    stdromano::Vector<SceneGraphNode*> _inputs;
    stdromano::Vector<SceneGraphNode*> _outputs;
    stdromano::Vector<Parameter> _params;

    uint32_t _id;
    stdromano::String<> _name;
    stdromano::String<> _error;

    uint32_t _num_outputs = 0;

    bool _dirty = true;

    SceneGraph* _parent = nullptr;

    void clear() noexcept;

    void prepare_objects() noexcept;

    void set_id(const uint32_t id) noexcept { this->_id = id; }

public:
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

    void set_dirty() noexcept;

    ROMANORENDER_FORCE_INLINE void set_not_dirty() noexcept { this->_dirty = false; }

    virtual bool execute() = 0;

    size_t get_memory_usage() const noexcept;

    uint32_t get_num_inputs() const noexcept { return this->_inputs.size(); };

    virtual const char* get_input_name(const uint32_t input) const noexcept = 0;

    virtual const char* get_type_name() const noexcept = 0;

    ROMANORENDER_FORCE_INLINE uint32_t get_id() const noexcept { return this->_id; }

    ROMANORENDER_FORCE_INLINE const stdromano::String<>& get_name() const noexcept
    {
        return this->_name;
    }

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

    ROMANORENDER_FORCE_INLINE void add_parameter(Parameter& param) noexcept
    {
        this->_params.emplace_back(std::move(param));
        this->_params.back()._parent = this;
    }

    template <typename... Args>
    void add_parameter(Args&&... args) noexcept
    {
        this->_params.emplace_back(std::forward<Args&&>(args)...);
        this->_params.back()._parent = this;
    }

    ROMANORENDER_FORCE_INLINE Parameter* get_parameter(const stdromano::String<>& name) noexcept
    {
        for(auto& param : this->_params)
        {
            if(param.get_name() == name)
            {
                return std::addressof(param);
            }
        }

        return nullptr;
    }

    ROMANORENDER_FORCE_INLINE stdromano::Vector<Parameter>& get_parameters() noexcept
    {
        return this->_params;
    }

    ROMANORENDER_FORCE_INLINE void set_error(const char* error_string) noexcept
    {
        this->_error = error_string;
    }

    ROMANORENDER_FORCE_INLINE void set_error(stdromano::String<>& error_string) noexcept
    {
        this->_error = std::move(error_string);
    }

    ROMANORENDER_FORCE_INLINE void set_error(const stdromano::String<>& error_string) noexcept
    {
        this->_error = error_string;
    }

    ROMANORENDER_FORCE_INLINE const stdromano::String<>& get_error() const noexcept
    {
        return this->_error;
    }
};

class ROMANORENDER_API SceneGraph
{
    stdromano::Vector<SceneGraphNode*> _nodes;

    SceneGraphNode* _output_node = nullptr;
    SceneGraphNode* _error_node = nullptr;

    size_t _memory_usage = 0;

    uint32_t _id_counter = 0;

    bool _is_dirty = false;

public:
    SceneGraph();

    ~SceneGraph();

    ROMANORENDER_FORCE_INLINE bool is_dirty() const noexcept { return this->_is_dirty; }

    ROMANORENDER_FORCE_INLINE void set_dirty() noexcept { this->_is_dirty = true; }

    ROMANORENDER_FORCE_INLINE void set_not_dirty() noexcept { this->_is_dirty = false; }

    ROMANORENDER_FORCE_INLINE size_t get_memory_usage() const noexcept { return this->_memory_usage; }

    void add_node(SceneGraphNode* node) noexcept;

    void remove_node(const uint32_t node_id) noexcept;

    void remove_node(SceneGraphNode* node) noexcept;

    SceneGraphNode* get_node_by_id(const uint32_t id) noexcept;

    ROMANORENDER_FORCE_INLINE stdromano::Vector<SceneGraphNode*>& get_nodes() noexcept
    {
        return this->_nodes;
    }

    ROMANORENDER_FORCE_INLINE const stdromano::Vector<SceneGraphNode*>& get_nodes() const noexcept
    {
        return this->_nodes;
    }

    void connect_nodes(const uint32_t lhs, const uint32_t rhs, const uint32_t input) noexcept;

    void connect_nodes(SceneGraphNode* lhs, SceneGraphNode* rhs, const uint32_t input) noexcept;

    bool execute() noexcept;

    const stdromano::Vector<Object*>* get_result() const noexcept;

    ROMANORENDER_FORCE_INLINE SceneGraphNode* get_error_node() const noexcept
    {
        return this->_error_node;
    }
};

using json = nlohmann::json;

ROMANORENDER_API json serialize_graph(const SceneGraph& graph) noexcept;

ROMANORENDER_API void deserialize_graph(const json& serialized_graph, SceneGraph& graph) noexcept;

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
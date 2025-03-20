#pragma once

#if !defined(__ROMANORENDER_SCENEGRAPH)
#define __ROMANORENDER_SCENEGRAPH

#include "romanorender/object.h"

ROMANORENDER_NAMESPACE_BEGIN

enum class ParameterType : uint32_t
{
    Int = 0,
    Float = 1,
    String = 2,
    Bool = 3
};

class ROMANORENDER_API Parameter
{
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

public:
    Parameter(const char* name, ParameterType type) : _type(type), _name(name)
    {
        switch(this->_type)
        {
        case ParameterType::Int:
            this->_data._int = 0;
            this->_is_inline = true;
            break;
        case ParameterType::Float:
            this->_data._float = 0.0f;
            this->_is_inline = true;
            break;
        case ParameterType::String:
            this->_data._ptr = new stdromano::String<>();
            this->_is_inline = false;
            break;
        case ParameterType::Bool:
            this->_data._bool = false;
            this->_is_inline = true;
            break;
        }
    }

    Parameter(const char* name, ParameterType type, int default_value)
        : _type(type), _name(name), _is_inline(true)
    {
        if(this->_type == ParameterType::Int)
        {
            this->_data._int = default_value;
        }
        else
        {
            switch(this->_type)
            {
            case ParameterType::Int:
                this->_data._int = 0;
                break;
            case ParameterType::Float:
                this->_data._float = 0.0f;
                break;
            case ParameterType::String:
                this->_data._ptr = new stdromano::String<>();
                this->_is_inline = false;
                break;
            case ParameterType::Bool:
                this->_data._bool = false;
                break;
            }
        }
    }

    Parameter(const char* name, ParameterType type, float default_value)
        : _type(type), _name(name), _is_inline(true)
    {
        if(this->_type == ParameterType::Float)
        {
            this->_data._float = default_value;
        }
        else
        {
            switch(this->_type)
            {
            case ParameterType::Int:
                this->_data._int = 0;
                break;
            case ParameterType::Float:
                this->_data._float = 0.0f;
                break;
            case ParameterType::String:
                this->_data._ptr = new stdromano::String<>();
                this->_is_inline = false;
                break;
            case ParameterType::Bool:
                this->_data._bool = false;
                break;
            }
        }
    }

    Parameter(const char* name, ParameterType type, bool default_value)
        : _type(type), _name(name), _is_inline(true)
    {
        if(this->_type == ParameterType::Bool)
        {
            this->_data._bool = default_value;
        }
        else
        {
            switch(this->_type)
            {
            case ParameterType::Int:
                this->_data._int = 0;
                break;
            case ParameterType::Float:
                this->_data._float = 0.0f;
                break;
            case ParameterType::String:
                this->_data._ptr = new stdromano::String<>();
                this->_is_inline = false;
                break;
            case ParameterType::Bool:
                this->_data._bool = false;
                break;
            }
        }
    }

    Parameter(const char* name, ParameterType type, const stdromano::String<>& default_value)
        : _type(type), _name(name)
    {
        if(this->_type == ParameterType::String)
        {
            this->_data._ptr = new stdromano::String<>(default_value);
            this->_is_inline = false;
        }
        else
        {
            switch(this->_type)
            {
            case ParameterType::Int:
                this->_data._int = 0;
                this->_is_inline = true;
                break;
            case ParameterType::Float:
                this->_data._float = 0.0f;
                this->_is_inline = true;
                break;
            case ParameterType::String:
                this->_data._ptr = new stdromano::String<>();
                this->_is_inline = false;
                break;
            case ParameterType::Bool:
                this->_data._bool = false;
                this->_is_inline = true;
                break;
            }
        }
    }

    Parameter(const char* name, ParameterType type, const char* default_value)
        : _type(type), _name(name)
    {
        if(this->_type == ParameterType::String)
        {
            this->_data._ptr = new stdromano::String<>(default_value);
            this->_is_inline = false;
        }
        else
        {
            switch(this->_type)
            {
            case ParameterType::Int:
                this->_data._int = 0;
                this->_is_inline = true;
                break;
            case ParameterType::Float:
                this->_data._float = 0.0f;
                this->_is_inline = true;
                break;
            case ParameterType::String:
                this->_data._ptr = new stdromano::String<>();
                this->_is_inline = false;
                break;
            case ParameterType::Bool:
                this->_data._bool = false;
                this->_is_inline = true;
                break;
            }
        }
    }

    ~Parameter()
    {
        if(!this->_is_inline && this->_data._ptr)
        {
            switch(this->_type)
            {
            case ParameterType::String:
                delete static_cast<stdromano::String<>*>(this->_data._ptr);
                break;
            default:
                break;
            }
            this->_data._ptr = nullptr;
        }
    }

    Parameter(const Parameter&) = delete;
    Parameter& operator=(const Parameter&) = delete;

    Parameter(Parameter&& other) noexcept : _type(other._type),
                                            _name(std::move(other._name)),
                                            _is_inline(other._is_inline)
    {

        this->_data = other._data;

        if(!this->_is_inline)
        {
            other._data._ptr = nullptr;
        }
    }

    Parameter& operator=(Parameter&& other) noexcept
    {
        if(this != &other)
        {
            if(!this->_is_inline && this->_data._ptr)
            {
                switch(this->_type)
                {
                case ParameterType::String:
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

    ROMANORENDER_FORCE_INLINE const stdromano::String<>& get_name() const noexcept { return this->_name; }

    ROMANORENDER_FORCE_INLINE ParameterType get_type() const noexcept { return this->_type; }

    template <typename T>
    T* get_value() const noexcept
    {
        static T temp_value;

        if constexpr(std::is_same_v<T, int> && _type == ParameterType::Int)
        {
            if(this->_is_inline)
            {
                temp_value = this->_data._int;
                return &temp_value;
            }
        }
        else if constexpr(std::is_same_v<T, float> && _type == ParameterType::Float)
        {
            if(this->_is_inline)
            {
                temp_value = this->_data._float;
                return &temp_value;
            }
        }
        else if constexpr(std::is_same_v<T, bool> && _type == ParameterType::Bool)
        {
            if(this->_is_inline)
            {
                temp_value = this->_data._bool;
                return &temp_value;
            }
        }
        else if constexpr(std::is_same_v<T, stdromano::String<> > && _type == ParameterType::String)
        {
            if(!this->_is_inline && this->_data._ptr)
            {
                return static_cast<stdromano::String<>*>(this->_data._ptr);
            }
        }
        return nullptr;
    }

    template <typename T>
    bool set_value(const T& value) noexcept
    {
        if constexpr(std::is_same_v<T, int> && _type == ParameterType::Int)
        {
            if(this->_is_inline)
            {
                this->_data._int = value;
                return true;
            }
        }
        else if constexpr(std::is_same_v<T, float> && _type == ParameterType::Float)
        {
            if(this->_is_inline)
            {
                this->_data._float = value;
                return true;
            }
        }
        else if constexpr(std::is_same_v<T, bool> && _type == ParameterType::Bool)
        {
            if(this->_is_inline)
            {
                this->_data._bool = value;
                return true;
            }
        }
        else if constexpr(std::is_same_v<T, stdromano::String<> > && _type == ParameterType::String)
        {
            if(!this->_is_inline && this->_data._ptr)
            {
                *static_cast<stdromano::String<>*>(this->_data._ptr) = value;
                return true;
            }
        }
        else if constexpr(std::is_same_v<T, const char*> && _type == ParameterType::String)
        {
            if(!this->_is_inline && this->_data._ptr)
            {
                *static_cast<stdromano::String<>*>(this->_data._ptr) = value;
                return true;
            }
        }
        return false;
    }

    int get_int() const noexcept
    {
        return (this->_type == ParameterType::Int && this->_is_inline) ? this->_data._int : 0;
    }

    float get_float() const noexcept
    {
        return (this->_type == ParameterType::Float && this->_is_inline) ? this->_data._float : 0.0f;
    }

    bool get_bool() const noexcept
    {
        return (this->_type == ParameterType::Bool && this->_is_inline) ? this->_data._bool : false;
    }

    const stdromano::String<>& get_string() const noexcept
    {
        static stdromano::String<> empty_string;
        return (this->_type == ParameterType::String && !this->_is_inline && this->_data._ptr)
                   ? *static_cast<stdromano::String<>*>(this->_data._ptr)
                   : empty_string;
    }

    bool set_int(int value) noexcept
    {
        if(this->_type == ParameterType::Int && this->_is_inline)
        {
            this->_data._int = value;
            return true;
        }
        
        return false;
    }

    bool set_float(float value) noexcept
    {
        if(this->_type == ParameterType::Float && this->_is_inline)
        {
            this->_data._float = value;
            return true;
        }

        return false;
    }

    bool set_bool(bool value) noexcept
    {
        if(this->_type == ParameterType::Bool && this->_is_inline)
        {
            this->_data._bool = value;
            return true;
        }

        return false;
    }

    bool set_string(const stdromano::String<>& value) noexcept
    {
        if(this->_type == ParameterType::String && !this->_is_inline && this->_data._ptr)
        {
            *static_cast<stdromano::String<>*>(this->_data._ptr) = value;
            return true;
        }

        return false;
    }

    bool set_string(const char* value) noexcept
    {
        if(this->_type == ParameterType::String && !this->_is_inline && this->_data._ptr)
        {
            *static_cast<stdromano::String<>*>(this->_data._ptr) = value;
            return true;
        }

        return false;
    }
};

class ROMANORENDER_API SceneGraphNode
{
    stdromano::Vector<Object*> _objects;
    stdromano::Vector<SceneGraphNode*> _inputs;
    stdromano::Vector<SceneGraphNode*> _outputs;
    stdromano::Vector<Parameter> _params;

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

    ROMANORENDER_FORCE_INLINE const stdromano::String<>& get_name() const noexcept { return this->_name; }

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

    SceneGraphNode* get_node_by_id(const uint32_t id) noexcept;

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
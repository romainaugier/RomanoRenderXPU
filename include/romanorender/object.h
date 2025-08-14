#pragma once

#if !defined(__ROMANORENDER_OBJECT)
#define __ROMANORENDER_OBJECT

#include "romanorender/camera.h"
#include "romanorender/cuda_vector.h"
#include "romanorender/mat44.h"
#include "romanorender/property.h"
#include "romanorender/tbvh.h"
#include "romanorender/vec4.h"
#include "romanorender/ray.h"
#include "romanorender/light.h"


#include "stdromano/hashmap.hpp"
#include "stdromano/string.hpp"
#include "stdromano/vector.hpp"

#include <regex>

ROMANORENDER_NAMESPACE_BEGIN

enum AttributeBufferType_ : uint32_t
{
    AttributeBufferType_Custom,
    AttributeBufferType_Normal,
    AttributeBufferType_Radius,
};

enum AttributeBufferFormat_ : uint32_t
{
    AttributeBufferFormat_Float1,
    AttributeBufferFormat_Float2,
    AttributeBufferFormat_Float3,
    AttributeBufferFormat_Float4,
    AttributeBufferFormat_Int1,
    AttributeBufferFormat_Int2,
    AttributeBufferFormat_Int3,
    AttributeBufferFormat_Int4,
    AttributeBufferFormat_UInt1,
    AttributeBufferFormat_UInt2,
    AttributeBufferFormat_UInt3,
    AttributeBufferFormat_UInt4,
};

class ROMANORENDER_API AttributeBuffer
{
    static constexpr uint32_t GEOM_BUFFER_ALIGNMENT = 16;

    void* data = nullptr;
    uint32_t* refcount = nullptr;

    uint32_t count = 0;
    uint32_t stride = 0;
    uint32_t type = 0;
    uint32_t format = 0;

public:
    AttributeBuffer() : data(nullptr), count(0), stride(0), type(0), format(0), refcount(nullptr) {}

    AttributeBuffer(AttributeBufferType_ type, AttributeBufferFormat_ format, const uint32_t stride, const uint32_t count);

    AttributeBuffer(const AttributeBuffer& other) noexcept;
    AttributeBuffer(AttributeBuffer&& other) noexcept;

    AttributeBuffer& operator=(const AttributeBuffer& other) noexcept;
    AttributeBuffer& operator=(AttributeBuffer&& other) noexcept;

    ~AttributeBuffer();

    ROMANORENDER_FORCE_INLINE void* get_data_ptr() const noexcept { return this->data; }

    template<typename T>
    ROMANORENDER_FORCE_INLINE T* get_data_ptr() const noexcept { return static_cast<T*>(this->data); }

    ROMANORENDER_FORCE_INLINE uint32_t get_count() const noexcept { return this->count; }

    ROMANORENDER_FORCE_INLINE uint32_t get_stride() const noexcept { return this->stride; }

    ROMANORENDER_FORCE_INLINE AttributeBufferType_ get_type() const noexcept { return (AttributeBufferType_)this->type; }

    ROMANORENDER_FORCE_INLINE AttributeBufferFormat_ get_format() const noexcept
    {
        return (AttributeBufferFormat_)this->format;
    }

    ROMANORENDER_FORCE_INLINE size_t get_memory_usage() const noexcept
    {
        return this->stride * this->count;
    }
};

#define INVALID_OBJECT_ID 0xFFFFFFFF
#define INVALID_OBJECT_UUID 0xFFFFFFFF

class ROMANORENDER_API Object
{
public:
    friend class ObjectsManager;
    friend class Scene;

protected:
    Property<Mat44F> _transform;

    uint32_t _uuid = INVALID_OBJECT_UUID;
    uint32_t _id = INVALID_OBJECT_ID;

    stdromano::StringD _name;
    stdromano::StringD _path;

public:
    Object() {}

    Object(const Object& other)
        : _transform(other._transform), _id(other._id), _name(other._name), _path(other._path)
    {
    }

    Object(Object&& other) noexcept : _transform(std::move(other._transform)),
                                      _id(other._id),
                                      _uuid(other._uuid),
                                      _name(std::move(other._name)),
                                      _path(std::move(other._path))
    {
        other._id = INVALID_OBJECT_ID;
        other._uuid = INVALID_OBJECT_UUID;
    }

    virtual uint32_t get_hash() const noexcept = 0;

    virtual Object* reference() const noexcept = 0;

    virtual size_t get_memory_usage() const noexcept = 0;

    virtual ~Object() = default;

    ROMANORENDER_FORCE_INLINE Mat44F& get_transform() noexcept
    {
        if(!this->_transform.initialized())
        {
            this->_transform.set(std::move(Mat44F()));
        }

        return this->_transform.get();
    }

    ROMANORENDER_FORCE_INLINE uint32_t get_id() const noexcept { return this->_id; }

    ROMANORENDER_FORCE_INLINE uint64_t get_uuid() const noexcept 
    {
        return (static_cast<uint64_t>(this->get_hash()) << 32) | static_cast<uint64_t>(this->_uuid); 
    }

    ROMANORENDER_FORCE_INLINE uint32_t get_uuid_32() const noexcept
    {
        return this->_uuid;
    }

    ROMANORENDER_FORCE_INLINE const stdromano::StringD& get_name() const noexcept
    {
        return this->_name;
    }

    ROMANORENDER_FORCE_INLINE const stdromano::StringD& get_path() const noexcept
    {
        return this->_path;
    }

    ROMANORENDER_FORCE_INLINE void set_id(uint32_t id) { this->_id = id; }

    ROMANORENDER_FORCE_INLINE void set_name(const stdromano::StringD& name)
    {
        this->_name = std::move(name);
        
        if(this->_path.empty())
        {
            this->_path = stdromano::StringD("/{}", this->_name);
        }
    }

    ROMANORENDER_FORCE_INLINE void set_path(const stdromano::StringD& path)
    {
        this->_path = std::move(path);
    }

    ROMANORENDER_FORCE_INLINE void set_transform(const Mat44F& transform) noexcept
    {
        this->_transform.set(transform);
    }
};

using Vertices = stdromano::Vector<Vec4F>;
using Indices = stdromano::Vector<uint32_t>;
using VertexAttributes = stdromano::HashMap<stdromano::StringD, Property<AttributeBuffer> >;

class ROMANORENDER_API ObjectMesh : public Object
{
    Property<Vertices> _vertices;
    Property<Indices> _indices;

    VertexAttributes _vertex_attributes;
    Property<uint32_t> _material_id;

    Property<uint8_t> _visibility_flags;

    bool _is_light_mesh = false;

    Vec3F get_primitive_normal(const uint32_t primitive_index) const noexcept;

public:
    ObjectMesh() {}

    ObjectMesh(const ObjectMesh& other)
        : Object(other), _vertices(other._vertices), _indices(other._indices),
          _vertex_attributes(other._vertex_attributes), _material_id(other._material_id),
          _visibility_flags(other._visibility_flags)
    {
    }

    ObjectMesh(ObjectMesh&& other) noexcept : Object(std::move(other)),
                                              _vertices(std::move(other._vertices)),
                                              _indices(std::move(other._indices)),
                                              _vertex_attributes(std::move(other._vertex_attributes)),
                                              _material_id(std::move(other._material_id)),
                                              _visibility_flags(std::move(other._visibility_flags))
    {
    }

    virtual ~ObjectMesh() override {}

    virtual uint32_t get_hash() const noexcept override;

    virtual ObjectMesh* reference() const noexcept override;

    virtual size_t get_memory_usage() const noexcept override;

    static ObjectMesh* cube(const Vec3F& center, const Vec3F& scale) noexcept;
    static ObjectMesh* geodesic(const Vec3F& center, const Vec3F& scale, const uint32_t subdiv_level) noexcept;
    static ObjectMesh* plane(const Vec3F& center, const Vec3F& scale) noexcept;

    void build_blas() noexcept;

    ROMANORENDER_FORCE_INLINE Vertices& get_vertices() noexcept
    {
        if(!this->_vertices.initialized())
        {
            this->_vertices.set(std::move(Vertices()));
        }

        return this->_vertices.get();
    };

    ROMANORENDER_FORCE_INLINE void set_vertices(Vertices& vertices) noexcept
    {
        this->_vertices.set(std::move(vertices));
    }

    ROMANORENDER_FORCE_INLINE Indices& get_indices() noexcept
    {
        if(!this->_indices.initialized())
        {
            this->_indices.set(std::move(Indices()));
        }

        return this->_indices.get();
    };

    ROMANORENDER_FORCE_INLINE void set_indices(Indices& indices) noexcept
    {
        this->_indices.set(std::move(indices));
    }

    ROMANORENDER_FORCE_INLINE VertexAttributes& get_vertex_attributes() noexcept
    {
        return this->_vertex_attributes;
    }

    ROMANORENDER_FORCE_INLINE uint32_t get_material_id() const noexcept
    {
        return this->_material_id.get();
    }

    ROMANORENDER_FORCE_INLINE uint8_t get_visibility_flags() noexcept
    {
        if(!this->_visibility_flags.initialized())
        {
            this->_visibility_flags.set((uint8_t)VisibilityFlag_VisibleAllRays);
        }

        return this->_visibility_flags.get();
    }

    ROMANORENDER_FORCE_INLINE void set_visibility_flags(const uint8_t flags) noexcept
    {
        this->_visibility_flags.set(flags);
    }

    ROMANORENDER_FORCE_INLINE bool get_is_light_mesh() const noexcept { return this->_is_light_mesh; }

    ROMANORENDER_FORCE_INLINE void set_is_light_mesh(const bool is_light_mesh) noexcept { this->_is_light_mesh = is_light_mesh; }

    void add_vertex_attribute_buffer(const stdromano::StringD& name, AttributeBuffer& buffer) noexcept;
    const AttributeBuffer* get_vertex_attribute_buffer(const stdromano::StringD& name) const noexcept;

    Vec3F get_normal(const uint32_t primitive, const float u, const float v) const noexcept;
};

class ROMANORENDER_API ObjectInstance : public Object
{
    Property<ObjectMesh*> _instanced;

    Property<uint8_t> _visibility_flags;

public:
    ObjectInstance() {}

    ObjectInstance(const ObjectInstance& other)
        : Object(other), _instanced(other._instanced),
          _visibility_flags(other._visibility_flags)
    {
    }

    ObjectInstance(ObjectInstance&& other) noexcept : Object(std::move(other)),
                                                      _instanced(std::move(other._instanced)),
                                                      _visibility_flags(std::move(other._visibility_flags))
    {
    }

    virtual ~ObjectInstance() override {}

    virtual uint32_t get_hash() const noexcept override { return 0; }

    virtual ObjectInstance* reference() const noexcept override;

    virtual size_t get_memory_usage() const noexcept override;

    ObjectMesh* get_instanced() const noexcept { return this->_instanced.get(); }

    ROMANORENDER_FORCE_INLINE void set_instanced(ObjectMesh* instanced) noexcept 
    {
        this->_instanced.set(instanced);
    }

    ROMANORENDER_FORCE_INLINE uint8_t get_visibility_flags() noexcept
    {
        if(!this->_visibility_flags.initialized())
        {
            this->_visibility_flags.set((uint8_t)VisibilityFlag_VisibleAllRays);
        }

        return this->_visibility_flags.get();
    }

    ROMANORENDER_FORCE_INLINE void set_visibility_flags(const uint8_t flags) noexcept
    {
        return this->_visibility_flags.set(flags);
    }
};

class ROMANORENDER_API ObjectCamera : public Object
{
    Property<Camera> _camera;

public:
    ObjectCamera();

    ObjectCamera(const ObjectCamera& other) : Object(other), _camera(other._camera) {}

    ObjectCamera(ObjectCamera&& other) noexcept : Object(std::move(other)), _camera(std::move(other._camera))
    {
    }

    virtual ObjectCamera* reference() const noexcept override;

    virtual ~ObjectCamera() override {}

    virtual uint32_t get_hash() const noexcept override { return 0; }

    virtual size_t get_memory_usage() const noexcept override;

    void set_xres(const uint32_t xres) noexcept;

    void set_yres(const uint32_t yres) noexcept;

    void set_focal(const float focal) noexcept;

    Camera* get_camera() noexcept;
};

class ROMANORENDER_API ObjectLight : public Object
{
    Property<LightBase*> _light;

public:
    ObjectLight(LightBase* light = nullptr) : _light(light) {}

    ObjectLight(const ObjectLight& other) : Object(other), _light(other._light) {}

    ObjectLight(ObjectLight&& other) noexcept : Object(std::move(other)), _light(std::move(other._light))
    {

    }

    virtual ObjectLight* reference() const noexcept override;

    virtual ~ObjectLight() override; 

    virtual uint32_t get_hash() const noexcept override;

    virtual size_t get_memory_usage() const noexcept override;

    LightBase* get_light() noexcept;

    const LightBase* get_light() const noexcept;
};

ROMANORENDER_API bool objects_from_obj_file(const char* file_path) noexcept;

ROMANORENDER_API bool objects_from_abc_file(const char* file_path) noexcept;

using ObjectsMatchingPatternIterator = uint32_t;

class ROMANORENDER_API ObjectsManager
{
public:
    static ObjectsManager& get_instance()
    {
        static ObjectsManager myInstance;

        return myInstance;
    }

    ObjectsManager(ObjectsManager const&) = delete;
    ObjectsManager(ObjectsManager&&) = delete;
    ObjectsManager& operator=(ObjectsManager const&) = delete;
    ObjectsManager& operator=(ObjectsManager&&) = delete;

    /* Returns the uuid of the created object */
    uint32_t add_object(Object* obj) noexcept;

    /* Removes an object given its uuid */
    void remove_object(const uint32_t uuid) noexcept;

    void remove_object(Object* obj) noexcept;

    /* Returns the uuid of the created light */
    uint32_t add_light(LightType_ type) noexcept;

    ROMANORENDER_FORCE_INLINE const stdromano::Vector<Object*>& get_objects() noexcept
    {
        return this->_objects;
    }

    ROMANORENDER_FORCE_INLINE void add_file_dependency(const char* file_path) noexcept
    {
        this->_file_dependencies.emplace_back(file_path);
    }

    bool get_objects_matching_pattern(ObjectsMatchingPatternIterator& it, 
                                      const std::regex& pattern,
                                      Object** object) const noexcept;

    Object* get_object_matching_uuid(const uint32_t uuid) const noexcept;

    size_t get_memory_usage() const noexcept;

private:
    ObjectsManager();

    ~ObjectsManager();

    uint32_t _uuid_counter = 0;

    stdromano::Vector<Object*> _objects;
    stdromano::Vector<stdromano::StringD > _file_dependencies;
};

#define objects_manager() ObjectsManager::get_instance()

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_OBJECT) */
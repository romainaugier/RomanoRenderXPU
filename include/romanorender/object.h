#pragma once

#if !defined(__ROMANORENDER_OBJECT)
#define __ROMANORENDER_OBJECT

#include "romanorender/camera.h"
#include "romanorender/mat44.h"
#include "romanorender/property.h"
#include "romanorender/tbvh.h"
#include "romanorender/vec4.h"


#include "stdromano/hashmap.h"
#include "stdromano/string.h"
#include "stdromano/vector.h"


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

    AttributeBuffer(AttributeBufferType_ type,
                    AttributeBufferFormat_ format,
                    const uint32_t stride,
                    const uint32_t count);

    AttributeBuffer(const AttributeBuffer& other) noexcept;
    AttributeBuffer(AttributeBuffer&& other) noexcept;

    AttributeBuffer& operator=(const AttributeBuffer& other) noexcept;
    AttributeBuffer& operator=(AttributeBuffer&& other) noexcept;

    ~AttributeBuffer();

    void* get_geometry_ptr() const noexcept { return this->data; }

    uint32_t get_count() const noexcept { return this->count; }

    uint32_t get_stride() const noexcept { return this->stride; }

    AttributeBufferType_ get_type() const noexcept { return (AttributeBufferType_)this->type; }

    AttributeBufferFormat_ get_format() const noexcept { return (AttributeBufferFormat_)this->format; }
};

#define INVALID_OBJECT_ID 0xFFFFFFFF

class ROMANORENDER_API Object
{
protected:
    Property<Mat44F> _transform;

    uint32_t _id = INVALID_OBJECT_ID;

    stdromano::String<> _name;
    stdromano::String<> _path;

public:
    Object() {}

    Object(const Object& other) : _transform(other._transform), _id(other._id), _name(other._name), _path(other._path)
    {
    }

    Object(Object&& other) noexcept : _transform(std::move(other._transform)),
                                      _id(other._id),
                                      _name(std::move(other._name)),
                                      _path(std::move(other._path))
    {
        other._id = INVALID_OBJECT_ID;
    }

    virtual Object* reference() const noexcept = 0;

    virtual ~Object() {}

    ROMANORENDER_FORCE_INLINE Mat44F& get_transform() noexcept
    {
        if(!this->_transform.initialized())
        {
            this->_transform.set(std::move(Mat44F()));
        }

        return this->_transform.get();
    }

    ROMANORENDER_FORCE_INLINE uint32_t get_id() const noexcept { return this->_id; }

    ROMANORENDER_FORCE_INLINE const stdromano::String<>& get_name() const noexcept { return this->_name; }

    ROMANORENDER_FORCE_INLINE const stdromano::String<>& get_path() const noexcept { return this->_path; }

    ROMANORENDER_FORCE_INLINE void set_id(uint32_t id) { this->_id = id; }

    ROMANORENDER_FORCE_INLINE void set_name(const stdromano::String<>& name) { this->_name = std::move(name); }

    ROMANORENDER_FORCE_INLINE void set_path(const stdromano::String<>& path) { this->_path = std::move(path); }

    ROMANORENDER_FORCE_INLINE void set_transform(const Mat44F& transform) noexcept { this->_transform.set(transform); }
};

#define USE_BVH8 1

using Vertices = stdromano::Vector<Vec4F>;
using Indices = stdromano::Vector<uint32_t>;
using Attributes = stdromano::HashMap<stdromano::String<>, Property<AttributeBuffer> >;

class ROMANORENDER_API ObjectMesh : public Object
{
    Property<Vertices> _vertices;
    Property<Indices> _indices;

    Attributes _attributes;

#if USE_BVH8
    tinybvh::BVH8_CPU _blas;
#else
    tinybvh::BVH _blas;
#endif /* USE_BVH8 */

    Property<uint32_t> _material_id;

public:
    ObjectMesh() {}

    ObjectMesh(const ObjectMesh& other)
        : Object(other), _vertices(other._vertices), _indices(other._indices), _attributes(other._attributes),
          _material_id(other._material_id)
    {
    }

    ObjectMesh(ObjectMesh&& other) noexcept : Object(std::move(other)),
                                              _vertices(std::move(other._vertices)),
                                              _indices(std::move(other._indices)),
                                              _attributes(std::move(other._attributes)),
                                              _material_id(std::move(other._material_id))
    {
    }

    virtual Object* reference() const noexcept override
    {
        ObjectMesh* new_object = new ObjectMesh();

        new_object->_transform.reference(this->_transform.get_ptr());
        new_object->_vertices.reference(this->_vertices.get_ptr());
        new_object->_indices.reference(this->_indices.get_ptr());
        new_object->_material_id.reference(this->_material_id.get_ptr());

        for(const auto& [key, prop] : this->_attributes)
        {
            new_object->_attributes[key].reference(prop.get_ptr());
        }

        new_object->_id = this->_id;
        new_object->_name = this->_name;
        new_object->_path = this->_path;

        return new_object;
    }

    virtual ~ObjectMesh() override {}

    static ObjectMesh cube(const Vec3F& center, const Vec3F& scale) noexcept;
    static ObjectMesh geodesic(const Vec3F& center, const Vec3F& scale, const uint32_t subdiv_level) noexcept;
    static ObjectMesh plane(const Vec3F& center, const Vec3F& scale) noexcept;

    void build_blas() noexcept;

    void subdivide(const uint32_t subdiv_level) noexcept;

    ROMANORENDER_FORCE_INLINE stdromano::Vector<Vec4F>& get_vertices() noexcept
    {
        if(!this->_vertices.initialized())
        {
            this->_vertices.set(std::move(Vertices()));
        }

        return this->_vertices.get();
    };

    ROMANORENDER_FORCE_INLINE stdromano::Vector<uint32_t>& get_indices() noexcept
    {
        if(!this->_indices.initialized())
        {
            this->_indices.set(std::move(Indices()));
        }

        return this->_indices.get();
    };

#if USE_BVH8
    ROMANORENDER_FORCE_INLINE const tinybvh::BVH8_CPU& get_blas() const noexcept { return this->_blas; }
#else
    ROMANORENDER_FORCE_INLINE const tinybvh::BVH& get_blas() const noexcept { return this->_blas; }
#endif /* USE_BVH8 */

    ROMANORENDER_FORCE_INLINE uint32_t get_material_id() const noexcept { return this->_material_id.get(); }

    void add_attribute_buffer(const stdromano::String<>& name, AttributeBuffer& buffer) noexcept;
    const AttributeBuffer* get_attribute_buffer(const stdromano::String<>& name) const noexcept;

    Vec3F get_primitive_normal(const uint32_t primitive_index) const noexcept;
};

class ROMANORENDER_API ObjectCamera : public Object
{
    Property<Camera> _camera;

public:
    ObjectCamera() {}

    ObjectCamera(const ObjectCamera& other) : Object(other), _camera(other._camera) {}

    ObjectCamera(ObjectCamera&& other) noexcept : Object(std::move(other)), _camera(std::move(other._camera)) {}

    virtual Object* reference() const noexcept override
    {
        ObjectCamera* new_object = new ObjectCamera();

        new_object->_transform.reference(this->_transform.get_ptr());
        new_object->_camera.reference(this->_camera.get_ptr());
        new_object->_id = this->_id;
        new_object->_name = this->_name;
        new_object->_path = this->_path;

        return new_object;
    }

    virtual ~ObjectCamera() override {}

    void set_xres(const uint32_t xres) noexcept;

    void set_yres(const uint32_t yres) noexcept;

    void set_focal(const float focal) noexcept;

    Camera* get_camera() noexcept;
};

ROMANORENDER_API bool objects_from_obj_file(const char* file_path) noexcept;

ROMANORENDER_API bool objects_from_abc_file(const char* file_path) noexcept;

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

    ROMANORENDER_FORCE_INLINE void add_object(Object* obj) noexcept { this->_objects.emplace_back(obj); }

    ROMANORENDER_FORCE_INLINE const stdromano::Vector<Object*>& get_objects() noexcept { return this->_objects; }

    ROMANORENDER_FORCE_INLINE void add_file_dependency(const char* file_path) noexcept
    {
        this->_file_dependencies.emplace_back(file_path);
    }

private:
    ObjectsManager();

    ~ObjectsManager();

    stdromano::Vector<Object*> _objects;
    stdromano::Vector<stdromano::String<> > _file_dependencies;
};

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_OBJECT) */
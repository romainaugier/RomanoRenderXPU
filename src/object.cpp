#include "romanorender/object.h"

#include "stdromano/random.h"

#include <map>

ROMANORENDER_NAMESPACE_BEGIN

AttributeBuffer::AttributeBuffer(AttributeBufferType_ type,
                                 AttributeBufferFormat_ format,
                                 const uint32_t stride,
                                 const uint32_t count)
{
    ROMANORENDER_ASSERT(stride < UINT8_MAX, "Stride cannot be greater than " ROMANORENDER_STRINGIZE(UINT8_MAX));

    this->type = (uint32_t)type;
    this->format = (uint32_t)format;
    this->stride = stride;
    this->count = count;
    this->refcount = new uint32_t(1);

    this->data = stdromano::mem_aligned_alloc(count * stride, GEOM_BUFFER_ALIGNMENT);
}

AttributeBuffer::AttributeBuffer(const AttributeBuffer& other) noexcept : 
                                                              data(other.data),
                                                              count(other.count),
                                                              stride(other.stride),
                                                              type(other.type),
                                                              format(other.format),
                                                              refcount(other.refcount)
{
    if(this->refcount != nullptr)
    {
        ++(*this->refcount);
    }
}

AttributeBuffer& AttributeBuffer::operator=(const AttributeBuffer& other) noexcept
{
    if(this != &other)
    {
        if(this->refcount != nullptr)
        {
            --(*this->refcount);

            if(*this->refcount == 0)
            {
                stdromano::mem_aligned_free(this->data);
                delete this->refcount;
            }
        }

        std::memmove(this, &other, sizeof(AttributeBuffer));
    
        if(this->refcount != nullptr)
        {
            ++(*this->refcount);
        }
    }

    return *this;
}

AttributeBuffer::AttributeBuffer(AttributeBuffer&& other) noexcept
{
    std::memmove(this, &other, sizeof(AttributeBuffer));
    std::memset(&other, 0, sizeof(AttributeBuffer));
}

AttributeBuffer& AttributeBuffer::operator=(AttributeBuffer&& other) noexcept
{
    if(this != &other)
    {
        if(this->refcount != nullptr)
        {
            --(*this->refcount);

            if(*this->refcount == 0)
            {
                stdromano::mem_aligned_free(this->data);
                delete this->refcount;
            }
        }

        std::memmove(this, &other, sizeof(AttributeBuffer));
        std::memset(&other, 0, sizeof(AttributeBuffer));
    }

    return *this;
}

AttributeBuffer::~AttributeBuffer()
{
    --(*this->refcount);

    if(this->data != nullptr && *this->refcount == 0)
    {
        stdromano::mem_aligned_free(this->data);
        delete this->refcount;
    }
}

Object::Object(const Object& other) noexcept
{
    this->_vertices = other._vertices;
    this->_indices = other._indices;
    this->_blas = other._blas;
    this->_transform = other._transform;
    this->_id = other._id;
    this->_name = other._name;
}

Object::Object(Object&& other) noexcept
{
    this->_vertices = std::move(other._vertices);
    this->_indices = std::move(other._indices);
    this->_blas = std::move(other._blas);
    this->_transform = std::move(other._transform);
    this->_id = other._id;
    this->_name = std::move(other._name);

    other.~Object();
}

Object& Object::operator=(const Object& other) noexcept
{
    this->_vertices = other._vertices;
    this->_indices = other._indices;
    this->_blas = other._blas;
    this->_transform = other._transform;
    this->_id = other._id;
    this->_name = other._name;

    return *this;
}

Object& Object::operator=(Object&& other) noexcept
{
    this->_vertices = std::move(other._vertices);
    this->_indices = std::move(other._indices);
    this->_blas = std::move(other._blas);
    this->_transform = std::move(other._transform);
    this->_id = other._id;
    this->_name = std::move(other._name);

    other.~Object();

    return *this;
}

Object Object::random_triangles(const uint32_t num_triangles, const float scale) noexcept
{
    Object triangles;

    for(uint32_t i = 0; i < num_triangles; i++)
    {
        const float x = stdromano::wang_hash_float(i ^ 0x473819UL);
        const float y = stdromano::wang_hash_float(i ^ 0xFA4838UL);
        const float z = stdromano::wang_hash_float(i ^ 0xEE438FUL);

        triangles.get_vertices().push_back(Vec4F(x + scale * stdromano::next_random_float_01(),
                                                 y + scale * stdromano::next_random_float_01(),
                                                 z + scale * stdromano::next_random_float_01(),
                                                 0.0f));
        triangles.get_vertices().push_back(Vec4F(x + scale * stdromano::next_random_float_01(),
                                                 y + scale * stdromano::next_random_float_01(),
                                                 z + scale * stdromano::next_random_float_01(),
                                                 0.0f));
        triangles.get_vertices().push_back(Vec4F(x + scale * stdromano::next_random_float_01(),
                                                 y + scale * stdromano::next_random_float_01(),
                                                 z + scale * stdromano::next_random_float_01(),
                                                 0.0f));

        // triangles.get_indices().push_back(i * 3 + 0);
        // triangles.get_indices().push_back(i * 3 + 1);
        // triangles.get_indices().push_back(i * 3 + 2);
    }

    return std::move(triangles);
}

Object Object::cube(const Vec3F& center, const Vec3F& scale) noexcept
{
    Object cube;

    cube.get_vertices().push_back(Vec4F((center.x + scale.x / 2.0f), (center.y + scale.y / 2.0f), (center.z + scale.z / 2.0f), 0.0f));
    cube.get_vertices().push_back(Vec4F((center.x + scale.x / 2.0f), (center.y + scale.y / 2.0f), (center.z - scale.z / 2.0f), 0.0f));
    cube.get_vertices().push_back(Vec4F((center.x + scale.x / 2.0f), (center.y - scale.y / 2.0f), (center.z + scale.z / 2.0f), 0.0f));
    cube.get_vertices().push_back(Vec4F((center.x + scale.x / 2.0f), (center.y - scale.y / 2.0f), (center.z - scale.z / 2.0f), 0.0f));
    cube.get_vertices().push_back(Vec4F((center.x - scale.x / 2.0f), (center.y + scale.y / 2.0f), (center.z + scale.z / 2.0f), 0.0f));
    cube.get_vertices().push_back(Vec4F((center.x - scale.x / 2.0f), (center.y + scale.y / 2.0f), (center.z - scale.z / 2.0f), 0.0f));
    cube.get_vertices().push_back(Vec4F((center.x - scale.x / 2.0f), (center.y - scale.y / 2.0f), (center.z + scale.z / 2.0f), 0.0f));
    cube.get_vertices().push_back(Vec4F((center.x - scale.x / 2.0f), (center.y - scale.y / 2.0f), (center.z - scale.z / 2.0f), 0.0f));

    cube.get_indices().push_back(0); cube.get_indices().push_back(2); cube.get_indices().push_back(6);
    cube.get_indices().push_back(0); cube.get_indices().push_back(6); cube.get_indices().push_back(4);
    cube.get_indices().push_back(1); cube.get_indices().push_back(3); cube.get_indices().push_back(7);
    cube.get_indices().push_back(1); cube.get_indices().push_back(7); cube.get_indices().push_back(5);
    cube.get_indices().push_back(0); cube.get_indices().push_back(1); cube.get_indices().push_back(5);
    cube.get_indices().push_back(0); cube.get_indices().push_back(5); cube.get_indices().push_back(4);
    cube.get_indices().push_back(2); cube.get_indices().push_back(3); cube.get_indices().push_back(7);
    cube.get_indices().push_back(2); cube.get_indices().push_back(7); cube.get_indices().push_back(6);
    cube.get_indices().push_back(0); cube.get_indices().push_back(1); cube.get_indices().push_back(3);
    cube.get_indices().push_back(0); cube.get_indices().push_back(3); cube.get_indices().push_back(2);
    cube.get_indices().push_back(4); cube.get_indices().push_back(5); cube.get_indices().push_back(7);
    cube.get_indices().push_back(4); cube.get_indices().push_back(7); cube.get_indices().push_back(6);

    return std::move(cube);
}

Object Object::geodesic(const Vec3F& center, const Vec3F& scale, const uint32_t subdiv_level) noexcept
{
    Object geodesic = Object::cube(Vec3F(0.0f, 0.0f, 0.0f), Vec3F(2.0f, 2.0f, 2.0f));

    for(auto& v : geodesic.get_vertices()) 
    {
        Vec3F pos(v.x, v.y, v.z);

        const float len = length_vec3f(pos);

        if(len > 0.0f) 
        {
            pos /= len;
            v.x = pos.x;
            v.y = pos.y;
            v.z = pos.z;
        }
    }

    for(uint32_t level = 0; level < subdiv_level; ++level)
    {
        stdromano::Vector<uint32_t> old_indices = geodesic.get_indices();
        geodesic.get_indices().clear();
        stdromano::Vector<Vec4F> new_vertices = geodesic.get_vertices();

        std::map<std::pair<uint32_t, uint32_t>, uint32_t> edge_map;

        for(size_t i = 0; i < old_indices.size(); i += 3) 
        {
            uint32_t i0 = old_indices[i];
            uint32_t i1 = old_indices[i + 1];
            uint32_t i2 = old_indices[i + 2];

            auto edge0 = std::minmax(i0, i1);
            auto edge1 = std::minmax(i1, i2);
            auto edge2 = std::minmax(i2, i0);

            std::pair<uint32_t, uint32_t> edges[3] = { edge0, edge1, edge2 };
            uint32_t mid[3];

            for(int e = 0; e < 3; ++e) 
            {
                const auto& edge = edges[e];
                auto it = edge_map.find(edge);

                if(it != edge_map.end()) 
                {
                    mid[e] = it->second;
                }
                else 
                {
                    const Vec4F& va = new_vertices[edge.first];
                    const Vec4F& vb = new_vertices[edge.second];
                    const Vec3F a(va.x, va.y, va.z);
                    const Vec3F b(vb.x, vb.y, vb.z);
                    const Vec3F midpoint = normalize_safe_vec3f((a + b) * 0.5f);

                    uint32_t new_idx = new_vertices.size();
                    new_vertices.emplace_back(midpoint.x, midpoint.y, midpoint.z, 0.0f);
                    edge_map[edge] = new_idx;
                    mid[e] = new_idx;
                }
            }

            geodesic.get_indices().push_back(i0);
            geodesic.get_indices().push_back(mid[0]);
            geodesic.get_indices().push_back(mid[2]);

            geodesic.get_indices().push_back(mid[0]);
            geodesic.get_indices().push_back(i1);
            geodesic.get_indices().push_back(mid[1]);

            geodesic.get_indices().push_back(mid[2]);
            geodesic.get_indices().push_back(mid[1]);
            geodesic.get_indices().push_back(i2);

            geodesic.get_indices().push_back(mid[0]);
            geodesic.get_indices().push_back(mid[1]);
            geodesic.get_indices().push_back(mid[2]);
        }

        geodesic.get_vertices() = std::move(new_vertices);
    }

    for(auto& v : geodesic.get_vertices()) 
    {
        v.x = v.x * scale.x + center.x;
        v.y = v.y * scale.y + center.y;
        v.z = v.z * scale.z + center.z;
    }

    return std::move(geodesic);
}

Object Object::plane(const Vec3F& center, const Vec3F& scale) noexcept
{
    Object plane;

    plane.get_vertices().push_back(Vec4F(center.x - scale.x / 2.0f, 
                                         center.y, 
                                         center.z + scale.z / 2.0f, 
                                         0.0f));
    plane.get_vertices().push_back(Vec4F(center.x - scale.x / 2.0f, 
                                         center.y, 
                                         center.z - scale.z / 2.0f,
                                         0.0f));
    plane.get_vertices().push_back(Vec4F(center.x + scale.x / 2.0f,
                                         center.y,
                                         center.z - scale.z / 2.0f,
                                         0.0f));

    plane.get_indices().push_back(0);
    plane.get_indices().push_back(1);
    plane.get_indices().push_back(2);

    plane.get_vertices().push_back(Vec4F(center.x + scale.x / 2.0f, 
                                         center.y, 
                                         center.z + scale.z / 2.0f,
                                         0.0f));

    plane.get_indices().push_back(3);
    plane.get_indices().push_back(0);
    plane.get_indices().push_back(2);

    return std::move(plane);
}

void Object::build_blas() noexcept
{
    if(this->_indices.size() > 0)
    {
        this->_blas.Build((tbvh::Vec4F*)this->_vertices.data(), this->_indices.data(), this->_indices.size() / 3);
    }
    else
    {
        this->_blas.Build((tbvh::Vec4F*)this->_vertices.data(), this->_vertices.size() / 3);
    }
}

void Object::add_attribute_buffer(const stdromano::String<>& name, AttributeBuffer& buffer) noexcept
{
    this->_attributes.insert(std::make_pair(name, std::move(buffer)));
}

const AttributeBuffer* Object::get_attribute_buffer(const stdromano::String<>& name) const noexcept
{
    const auto it = this->_attributes.find(name);

    return it == this->_attributes.end() ? nullptr : &it->second;
}

bool objects_from_obj_file(const char* file_path, stdromano::Vector<Object>& object) noexcept
{
    return true;
}

ROMANORENDER_NAMESPACE_END
#include "romanorender/object.h"

#include "stdromano/char.hpp"
#include "stdromano/logger.hpp"
#include "stdromano/random.hpp"
#include "stdromano/threading.hpp"
#include <unistd.h>

#define STDROMANO_ENABLE_PROFILING
#include "stdromano/profiling.hpp"

#include "Alembic/AbcCoreOgawa/All.h"
#include "Alembic/AbcGeom/All.h"

#include <charconv>
#include <cstdio>
#include <map>
#include <unordered_set>


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

AttributeBuffer::AttributeBuffer(const AttributeBuffer& other) noexcept : data(other.data),
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

/* Mesh */

uint32_t ObjectMesh::get_hash() const noexcept
{
    uint32_t hash = 0;

    hash ^= stdromano::hash_murmur3(this->_vertices.get().data(),
                                    this->_vertices.get().size() * sizeof(Vertices::value_type),
                                    0xABCDEF);

    hash ^= stdromano::hash_murmur3(this->_indices.get().data(),
                                    this->_indices.get().size() * sizeof(Indices::value_type),
                                    0xABCDEF);

    for(const auto& it : this->_vertex_attributes)
    {
        hash ^= stdromano::hash_murmur3(it.second.get().get_data_ptr(),
                                        it.second.get().get_count() * it.second.get().get_stride(),
                                        0xABCDEF);
    }

    return hash;
}

ObjectMesh* ObjectMesh::reference() const noexcept
{
    ObjectMesh* new_object = new ObjectMesh();

    new_object->_transform.reference(this->_transform.get_ptr());
    new_object->_vertices.reference(this->_vertices.get_ptr());
    new_object->_indices.reference(this->_indices.get_ptr());
    new_object->_material_id.reference(this->_material_id.get_ptr());
    new_object->_visibility_flags.reference(this->_visibility_flags.get_ptr());

    for(const auto& it : this->_vertex_attributes)
    {
        new_object->_vertex_attributes[it.first].reference(it.second.get_ptr());
    }

    new_object->_id = this->_id;
    new_object->_uuid = this->_uuid;
    new_object->_name = this->_name;
    new_object->_path = this->_path;

    return new_object;
}

size_t ObjectMesh::get_memory_usage() const noexcept
{
    size_t mem_usage = 0;

    mem_usage += (this->_transform.owns_data() ? sizeof(Mat44F) : 0) + sizeof(Property<Mat44F>);
    mem_usage += sizeof(uint32_t);
    mem_usage += sizeof(stdromano::StringD) + this->_name.size() * sizeof(char);
    mem_usage += sizeof(stdromano::StringD) + this->_path.size() * sizeof(char);

    mem_usage += (this->_vertices.owns_data() ? this->_vertices.get().memory_usage() : 0)
                 + sizeof(Property<Vertices>);
    mem_usage += (this->_indices.owns_data() ? this->_indices.get().memory_usage() : 0)
                 + sizeof(Property<Indices>);

    for(const auto& it : this->_vertex_attributes)
    {
        mem_usage += (it.second.owns_data() ? it.second.get().get_memory_usage() : 0)
                     + sizeof(Property<AttributeBuffer>);
    }

    mem_usage += this->_vertex_attributes.memory_usage();

    mem_usage += sizeof(Property<bool>);

    return mem_usage;
}

ObjectMesh* ObjectMesh::cube(const Vec3F& center, const Vec3F& scale) noexcept
{
    ObjectMesh* cube = new ObjectMesh;

    cube->get_vertices().push_back(Vec4F((center.x + scale.x / 2.0f),
                                        (center.y + scale.y / 2.0f),
                                        (center.z + scale.z / 2.0f),
                                        0.0f));
    cube->get_vertices().push_back(Vec4F((center.x + scale.x / 2.0f),
                                        (center.y + scale.y / 2.0f),
                                        (center.z - scale.z / 2.0f),
                                        0.0f));
    cube->get_vertices().push_back(Vec4F((center.x + scale.x / 2.0f),
                                        (center.y - scale.y / 2.0f),
                                        (center.z + scale.z / 2.0f),
                                        0.0f));
    cube->get_vertices().push_back(Vec4F((center.x + scale.x / 2.0f),
                                        (center.y - scale.y / 2.0f),
                                        (center.z - scale.z / 2.0f),
                                        0.0f));
    cube->get_vertices().push_back(Vec4F((center.x - scale.x / 2.0f),
                                        (center.y + scale.y / 2.0f),
                                        (center.z + scale.z / 2.0f),
                                        0.0f));
    cube->get_vertices().push_back(Vec4F((center.x - scale.x / 2.0f),
                                        (center.y + scale.y / 2.0f),
                                        (center.z - scale.z / 2.0f),
                                        0.0f));
    cube->get_vertices().push_back(Vec4F((center.x - scale.x / 2.0f),
                                        (center.y - scale.y / 2.0f),
                                        (center.z + scale.z / 2.0f),
                                        0.0f));
    cube->get_vertices().push_back(Vec4F((center.x - scale.x / 2.0f),
                                        (center.y - scale.y / 2.0f),
                                        (center.z - scale.z / 2.0f),
                                        0.0f));

    // Face 1: Front
    cube->get_indices().push_back(0);
    cube->get_indices().push_back(6);
    cube->get_indices().push_back(2);

    cube->get_indices().push_back(6);
    cube->get_indices().push_back(0);
    cube->get_indices().push_back(4);

    // Face 2: Back
    cube->get_indices().push_back(7);
    cube->get_indices().push_back(1);
    cube->get_indices().push_back(3);

    cube->get_indices().push_back(5);
    cube->get_indices().push_back(1);
    cube->get_indices().push_back(7);

    // Face 3: Top
    cube->get_indices().push_back(1);
    cube->get_indices().push_back(5);
    cube->get_indices().push_back(0);

    cube->get_indices().push_back(4);
    cube->get_indices().push_back(0);
    cube->get_indices().push_back(5);

    // Face 4: Bottom
    cube->get_indices().push_back(3);
    cube->get_indices().push_back(2);
    cube->get_indices().push_back(7);

    cube->get_indices().push_back(7);
    cube->get_indices().push_back(2);
    cube->get_indices().push_back(6);

    // Face 5: Right
    cube->get_indices().push_back(1);
    cube->get_indices().push_back(0);
    cube->get_indices().push_back(3);

    cube->get_indices().push_back(3);
    cube->get_indices().push_back(0);
    cube->get_indices().push_back(2);

    // Face 6: Left
    cube->get_indices().push_back(7);
    cube->get_indices().push_back(4);
    cube->get_indices().push_back(5);

    cube->get_indices().push_back(6);
    cube->get_indices().push_back(4);
    cube->get_indices().push_back(7);

    cube->set_name("cube");
    cube->set_path("/cube");

    cube->set_transform(Mat44F::identity());

    return cube;
}

ObjectMesh* ObjectMesh::geodesic(const Vec3F& center, const Vec3F& scale, const uint32_t subdiv_level) noexcept
{
    ObjectMesh* geodesic = ObjectMesh::cube(Vec3F(0.0f, 0.0f, 0.0f), Vec3F(2.0f, 2.0f, 2.0f));

    for(auto& v : geodesic->get_vertices())
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
        stdromano::Vector<uint32_t> old_indices = geodesic->get_indices();
        geodesic->get_indices().clear();
        stdromano::Vector<Vec4F> new_vertices = std::move(geodesic->get_vertices());

        std::map<std::pair<uint32_t, uint32_t>, uint32_t> edge_map;

        for(size_t i = 0; i < old_indices.size(); i += 3)
        {
            uint32_t i0 = old_indices[i];
            uint32_t i1 = old_indices[i + 1];
            uint32_t i2 = old_indices[i + 2];

            auto edge0 = std::minmax(i0, i1);
            auto edge1 = std::minmax(i1, i2);
            auto edge2 = std::minmax(i2, i0);

            std::pair<uint32_t, uint32_t> edges[3] = {edge0, edge1, edge2};
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

            geodesic->get_indices().push_back(i0);
            geodesic->get_indices().push_back(mid[0]);
            geodesic->get_indices().push_back(mid[2]);

            geodesic->get_indices().push_back(mid[0]);
            geodesic->get_indices().push_back(i1);
            geodesic->get_indices().push_back(mid[1]);

            geodesic->get_indices().push_back(mid[2]);
            geodesic->get_indices().push_back(mid[1]);
            geodesic->get_indices().push_back(i2);

            geodesic->get_indices().push_back(mid[0]);
            geodesic->get_indices().push_back(mid[1]);
            geodesic->get_indices().push_back(mid[2]);
        }

        geodesic->get_vertices() = std::move(new_vertices);
    }

    for(auto& v : geodesic->get_vertices())
    {
        v.x = v.x * scale.x + center.x;
        v.y = v.y * scale.y + center.y;
        v.z = v.z * scale.z + center.z;
    }

    geodesic->set_name("geodesic");
    geodesic->set_path("/geodesic");

    geodesic->set_transform(Mat44F::identity());

    return geodesic;
}

ObjectMesh* ObjectMesh::plane(const Vec3F& center, const Vec3F& scale) noexcept
{
    ObjectMesh* plane = new ObjectMesh;

    plane->get_vertices().push_back(Vec4F(center.x + scale.x / 2.0f, center.y + scale.y / 2.0f, center.z + scale.z / 2.0f, 0.0f));
    plane->get_vertices().push_back(Vec4F(center.x + scale.x / 2.0f, center.y - scale.y / 2.0f, center.z - scale.z / 2.0f, 0.0f));
    plane->get_vertices().push_back(Vec4F(center.x - scale.x / 2.0f, center.y - scale.y / 2.0f, center.z - scale.z / 2.0f, 0.0f));
    plane->get_vertices().push_back(Vec4F(center.x - scale.x / 2.0f, center.y + scale.y / 2.0f, center.z + scale.z / 2.0f, 0.0f));

    plane->get_indices().push_back(0);
    plane->get_indices().push_back(1);
    plane->get_indices().push_back(2);

    plane->get_indices().push_back(2);
    plane->get_indices().push_back(3);
    plane->get_indices().push_back(0);

    plane->set_name("plane");
    plane->set_path("/plane");

    plane->set_transform(Mat44F::identity());

    return plane;
}

void ObjectMesh::add_vertex_attribute_buffer(const stdromano::StringD& name, AttributeBuffer& buffer) noexcept
{
    this->_vertex_attributes.insert(std::make_pair(name, Property<AttributeBuffer>(std::move(buffer))));
}

const AttributeBuffer* ObjectMesh::get_vertex_attribute_buffer(const stdromano::StringD& name) const noexcept
{
    const auto it = this->_vertex_attributes.find(name);

    return it == this->_vertex_attributes.end() ? nullptr : it->second.get_ptr();
}

Vec3F ObjectMesh::get_primitive_normal(const uint32_t primitive_index) const noexcept
{
    Vec3F N;

    const Vec4F& v0 = this->_vertices.get()[this->_indices.get()[primitive_index * 3 + 0]];
    const Vec4F& v1 = this->_vertices.get()[this->_indices.get()[primitive_index * 3 + 1]];
    const Vec4F& v2 = this->_vertices.get()[this->_indices.get()[primitive_index * 3 + 2]];

    const Vec4F A = normalize_safe_vec4f(v1 - v0);
    const Vec4F B = normalize_safe_vec4f(v2 - v0);

    N.x = A.y * B.z - A.z * B.y;
    N.y = A.z * B.x - A.x * B.z;
    N.z = A.x * B.y - A.y * B.x;

    return N;
}

Vec3F ObjectMesh::get_normal(const uint32_t primitive, const float u, const float v) const noexcept
{
    const AttributeBuffer* N_buffer = this->get_vertex_attribute_buffer("N");

    if(N_buffer == nullptr)
    {
        return this->get_primitive_normal(primitive);
    }

    const Vec3F* normals = N_buffer->get_data_ptr<Vec3F>();

    const Vec3F n0 = normals[this->_indices.get()[primitive * 3 + 0]];
    const Vec3F n1 = normals[this->_indices.get()[primitive * 3 + 1]];
    const Vec3F n2 = normals[this->_indices.get()[primitive * 3 + 2]];

    const float w = 1.0f - u - v;

    return n0 * w + n1 * u + n2 * v;
}

/* Instance */

ObjectInstance* ObjectInstance::reference() const noexcept
{
    ObjectInstance* new_object = new ObjectInstance();

    new_object->_transform.reference(this->_transform.get_ptr());
    new_object->_instanced.reference(this->_instanced.get_ptr());
    new_object->_visibility_flags.reference(this->_visibility_flags.get_ptr());
    new_object->_id = this->_id;
    new_object->_uuid = this->_uuid;
    new_object->_name = this->_name;
    new_object->_path = this->_path;

    return new_object;
}

size_t ObjectInstance::get_memory_usage() const noexcept
{
    size_t mem_usage = 0;

    mem_usage += (this->_transform.owns_data() ? sizeof(Mat44F) : 0) + sizeof(Property<Mat44F>);
    mem_usage += sizeof(uint32_t);
    mem_usage += sizeof(stdromano::StringD) + this->_name.size() * sizeof(char);
    mem_usage += sizeof(stdromano::StringD) + this->_path.size() * sizeof(char);

    mem_usage += sizeof(Property<ObjectMesh*>);

    return mem_usage;
}

/* Camera */

ObjectCamera::ObjectCamera()
{
    this->_camera.set(Camera());
}

ObjectCamera* ObjectCamera::reference() const noexcept
{
    ObjectCamera* new_object = new ObjectCamera();

    new_object->_transform.reference(this->_transform.get_ptr());
    new_object->_camera.reference(this->_camera.get_ptr());
    new_object->_id = this->_id;
    new_object->_uuid = this->_uuid;
    new_object->_name = this->_name;
    new_object->_path = this->_path;

    return new_object;
}

size_t ObjectCamera::get_memory_usage() const noexcept
{
    size_t mem_usage = 0;

    mem_usage += (this->_transform.owns_data() ? sizeof(Mat44F) : 0) + sizeof(Property<Mat44F>);
    mem_usage += sizeof(uint32_t);
    mem_usage += sizeof(stdromano::StringD) + this->_name.size() * sizeof(char);
    mem_usage += sizeof(stdromano::StringD) + this->_path.size() * sizeof(char);

    mem_usage += (this->_camera.owns_data() ? sizeof(Camera) : 0) + sizeof(Property<Camera>);

    return mem_usage;
}

Camera* ObjectCamera::get_camera() noexcept
{
    if(!this->_camera.initialized())
    {
        this->_camera.set(std::move(Camera()));
    }

    if(!this->_transform.initialized())
    {
        this->_transform.set(Mat44F());
    }

    this->_camera.get().set_transform(this->_transform.get());
    return this->_camera.get_ptr();
}

void ObjectCamera::set_xres(const uint32_t xres) noexcept
{
    if(!this->_camera.initialized())
    {
        this->_camera.set(std::move(Camera()));
    }

    this->_camera.get().set_xres(xres);
}

void ObjectCamera::set_yres(const uint32_t yres) noexcept
{
    if(!this->_camera.initialized())
    {
        this->_camera.set(std::move(Camera()));
    }

    this->_camera.get().set_yres(yres);
}

void ObjectCamera::set_focal(const float focal) noexcept
{
    if(!this->_camera.initialized())
    {
        this->_camera.set(std::move(Camera()));
    }

    this->_camera.get().set_focal(focal);
}

/* Light */

ObjectLight* ObjectLight::reference() const noexcept
{
    ObjectLight* new_object = new ObjectLight();

    new_object->_transform.reference(this->_transform.get_ptr());
    new_object->_light.reference(this->_light.get_ptr());
    new_object->_id = this->_id;
    new_object->_uuid = this->_uuid;
    new_object->_name = this->_name;
    new_object->_path = this->_path;

    return new_object;
}

ObjectLight::~ObjectLight()
{
    {
        if(this->_light.owns_data())
        {
            delete this->_light.get();
        }
    }
}

uint32_t ObjectLight::get_hash() const noexcept
{
    return 0;
}

size_t ObjectLight::get_memory_usage() const noexcept
{
    return 0;
}

LightBase* ObjectLight::get_light() noexcept
{
    if(!this->_light.initialized())
    {
        this->_light.set(new LightDome(0));
    }

    if(!this->_transform.initialized())
    {
        this->_transform.set(Mat44F());
    }

    return this->_light.get();
}

const LightBase* ObjectLight::get_light() const noexcept
{
    return this->_light.initialized() ? this->_light.get() : nullptr;
}

/* Get objects from file */

bool objects_from_obj_file(const char* file_path) noexcept
{
    SCOPED_PROFILE_START(stdromano::ProfileUnit::Seconds, obj_file_load);

    std::size_t num_objects = 0;

    std::FILE* file_handle = std::fopen(file_path, "r");

    if(file_handle == nullptr)
    {
        stdromano::log_error("Error while trying to open file: {}", file_path);
        return false;
    }

    std::fseek(file_handle, 0, SEEK_END);
    const size_t file_size = std::ftell(file_handle);
    std::rewind(file_handle);
    stdromano::StringD file_content = std::move(stdromano::StringD::make_zeroed(file_size + 1));
    std::fread(file_content.c_str(), sizeof(char), file_size, file_handle);
    std::fclose(file_handle);

    file_content[file_size] = '\0';

    stdromano::StringD split;
    stdromano::StringD::split_iterator split_it = 0;

    size_t current_line = 0;

    Vertices global_vertices;
    stdromano::HashMap<uint32_t, uint32_t> index_map;

    ObjectMesh* current_object = new ObjectMesh;

    while(file_content.split("\n", split_it, split))
    {
        current_line++;

        if(split.empty() || split[0] == '#')
        {
            continue;
        }

        switch(split[0])
        {
        case 'v':
        {
            if(!std::isspace(split[1]))
            {
                break;
            }

            global_vertices.emplace_back(0.0f);

            size_t j = 1;

            for(uint32_t i = 0; i < 3; i++)
            {
                while(j < split.size() && std::isspace(split[j]))
                {
                    j++;
                }

                char* start = &split[j];
                size_t size = static_cast<size_t>(split.back() - start);

                const auto res = std::from_chars(start,
                                                 start + (split.size() - j),
                                                 global_vertices[global_vertices.size() - 1][i]);

                j += res.ptr - start;
            }

            ROMANORENDER_ABORT_IF_VEC4F_NAN(global_vertices.back());

            break;
        }
        case 'g':
        case 'o':
        {
            if(!current_object->get_indices().empty())
            {
                stdromano::log_debug("Parsed new obj mesh \"{}\": {} vertices and {} primitives",
                                     current_object->get_name(),
                                     current_object->get_vertices().size(),
                                     current_object->get_indices().size() / 3);

                num_objects++;

                current_object->set_transform(Mat44F::identity());

                objects_manager().add_object(current_object);
                current_object = new ObjectMesh;
            }

            index_map.clear();
            current_object->set_name(std::move(stdromano::StringD("{}",
                                                                   fmt::string_view(split.data() + 2,
                                                                                    split.size() - 2))));

            current_object->set_path(std::move(stdromano::StringD("/{}", current_object->get_name())));

            break;
        }
        case 'f':
        {
            size_t i = 0;
            size_t parsed = 0;

            uint32_t indices[12];
            std::memset(indices, 0, 12 * sizeof(uint32_t));

            while(i < split.size())
            {
                int32_t index = 0;

                while(i < split.size() && (stdromano::is_digit(split[i]) || split[i] == '-'))
                {
                    bool negative = false;

                    if(split[i] == '-')
                    {
                        negative = true;
                        i++;
                    }

                    while(i < split.size() && stdromano::is_digit(split[i]))
                    {
                        index = index * 10 + (split[i] - '0');
                        i++;
                    }

                    if(negative)
                        index = -index;
                }

                if(index != 0)
                {
                    index = index < 0 ? global_vertices.size() + index : index - 1;

                    const auto it = index_map.find(index);

                    if(it == index_map.end())
                    {
                        current_object->get_vertices().push_back(global_vertices[index]);
                        index_map[index] = static_cast<uint32_t>(current_object->get_vertices().size() - 1);
                    }

                    indices[parsed] = index_map[index];
                    parsed++;

                    while(i < split.size() && split[i] != ' ')
                    {
                        i++;
                    }
                }

                i++;
            }

            switch(parsed)
            {
            case 3:
                current_object->get_indices().push_back(indices[0]);
                current_object->get_indices().push_back(indices[1]);
                current_object->get_indices().push_back(indices[2]);
                break;
            case 4:
                current_object->get_indices().push_back(indices[0]);
                current_object->get_indices().push_back(indices[1]);
                current_object->get_indices().push_back(indices[2]);

                current_object->get_indices().push_back(indices[2]);
                current_object->get_indices().push_back(indices[3]);
                current_object->get_indices().push_back(indices[0]);
                break;
            case 6:
                current_object->get_indices().push_back(indices[0]);
                current_object->get_indices().push_back(indices[2]);
                current_object->get_indices().push_back(indices[4]);
                break;
            case 8:
                current_object->get_indices().push_back(indices[0]);
                current_object->get_indices().push_back(indices[2]);
                current_object->get_indices().push_back(indices[4]);

                current_object->get_indices().push_back(indices[6]);
                current_object->get_indices().push_back(indices[4]);
                current_object->get_indices().push_back(indices[0]);
                break;
            case 9:
                current_object->get_indices().push_back(indices[0]);
                current_object->get_indices().push_back(indices[3]);
                current_object->get_indices().push_back(indices[6]);
                break;
            }

            break;
        }
        }
    }

    if(!current_object->get_indices().empty())
    {
        stdromano::log_debug("Parsed new obj mesh \"{}\": {} vertices and {} primitives",
                             current_object->get_name(),
                             current_object->get_vertices().size(),
                             current_object->get_indices().size() / 3);

        num_objects++;

        current_object->set_transform(Mat44F::identity());

        objects_manager().add_object(current_object);
    }
    else
    {
        delete current_object;
    }

    stdromano::log_debug("Parsed {} objects from file: {}", num_objects, file_path);

    ObjectsManager::get_instance().add_file_dependency(file_path);

    return true;
}

using namespace Alembic::AbcGeom;
using namespace Alembic::Abc;

void abc_traverse(IObject obj, const M44d& parentXform, const stdromano::StringD& path)
{
    if(IXform::matches(obj.getHeader()))
    {
        IXform xform(obj, kWrapExisting);
        XformSample xs;
        xform.getSchema().get(xs);
        M44d localXform = xs.getMatrix();
        M44d combinedXform = parentXform * localXform;

        const stdromano::StringD new_path("{}/{}", path, xform.getName().c_str());

        for(size_t i = 0; i < obj.getNumChildren(); ++i)
        {
            abc_traverse(obj.getChild(i), combinedXform, new_path);
        }
    }
    else if(ICamera::matches(obj.getHeader()))
    {
        ICamera camera(obj, kWrapExisting);
        ICameraSchema schema = camera.getSchema();
        CameraSample sample;
        schema.get(sample);

        const stdromano::StringD new_path("{}/{}", path, camera.getName().c_str());

        ObjectCamera* new_camera = new ObjectCamera;
        new_camera->set_name(stdromano::StringD("{}", obj.getName().c_str()));
        new_camera->set_path(new_path);

        new_camera->set_focal(sample.getFocalLength());

        new_camera->set_transform(Mat44F(parentXform));

        stdromano::log_debug("Found new camera: \"{}\" (pos: {})",
                             new_camera->get_path(),
                             new_camera->get_camera()->get_ray_origin());

        ObjectsManager::get_instance().add_object(new_camera);

        for(size_t i = 0; i < obj.getNumChildren(); ++i)
        {
            abc_traverse(obj.getChild(i), parentXform, new_path);
        }
    }
    else if(IPolyMesh::matches(obj.getHeader()))
    {
        IPolyMesh mesh(obj, kWrapExisting);
        IPolyMeshSchema& schema = mesh.getSchema();
        IPolyMeshSchema::Sample sample;
        schema.get(sample);

        const stdromano::StringD new_path("{}/{}", path, mesh.getName().c_str());

        const P3fArraySamplePtr positions = sample.getPositions();
        const Int32ArraySamplePtr faceIndices = sample.getFaceIndices();
        const Int32ArraySamplePtr faceCounts = sample.getFaceCounts();

        ObjectMesh* new_object = new ObjectMesh;
        new_object->set_name(stdromano::StringD("{}", obj.getName().c_str()));
        new_object->set_path(new_path);

        size_t indexOffset = 0;
        for(size_t i = 0; i < faceCounts->size(); ++i)
        {
            int numVerts = (*faceCounts)[i];
            for(int j = 1; j < numVerts - 1; ++j)
            {
                new_object->get_indices().push_back((*faceIndices)[indexOffset + j + 1]);
                new_object->get_indices().push_back((*faceIndices)[indexOffset + j]);
                new_object->get_indices().push_back((*faceIndices)[indexOffset]);
            }

            indexOffset += numVerts;
        }

        for(size_t i = 0; i < positions->size(); ++i)
        {
            V3f pos = (*positions)[i];
            new_object->get_vertices().emplace_back(pos.x, pos.y, pos.z, 0.0f);
        }

        new_object->set_transform(Mat44F(parentXform));

        stdromano::log_debug("Found new mesh \"{}\" ({} prims)",
                             new_object->get_path(),
                             new_object->get_indices().size() / 3);

        ObjectsManager::get_instance().add_object(new_object);

        for(size_t i = 0; i < obj.getNumChildren(); ++i)
        {
            abc_traverse(obj.getChild(i), parentXform, new_path);
        }
    }
    else
    {
        for(size_t i = 0; i < obj.getNumChildren(); ++i)
        {
            abc_traverse(obj.getChild(i), parentXform, path);
        }
    }
}

bool objects_from_abc_file(const char* file_path) noexcept
{
    IArchive archive(Alembic::AbcCoreOgawa::ReadArchive(), file_path);

    if(!archive.valid())
    {
        stdromano::log_error("Error while trying to read alembic archive from file: \"{}\"", file_path);
        return false;
    }

    IObject topObjectMesh = archive.getTop();
    stdromano::StringD path("/{}", topObjectMesh.getName().c_str());

    abc_traverse(topObjectMesh, M44d(), path);

    ObjectsManager::get_instance().add_file_dependency(file_path);

    return true;
}

/* Objects Manager singleton */

ObjectsManager::ObjectsManager() {}

ObjectsManager::~ObjectsManager()
{
    for(Object* obj : this->_objects)
    {
        delete obj;
    }
}

uint32_t ObjectsManager::add_object(Object* obj) noexcept
{
    obj->_uuid = this->_uuid_counter++;

    this->_objects.emplace_back(obj);

    return obj->_uuid;
}

void ObjectsManager::remove_object(const uint32_t uuid) noexcept
{
    uint32_t pos = 0xFFFFFFFF;

    for(uint32_t i = 0; i < this->_objects.size(); i++)
    {
        if(this->_objects[i]->get_uuid() == uuid)
        {
            pos = i;
        }
    }

    if(pos != 0xFFFFFFFF)
    {
        delete this->_objects[pos];

        this->_objects.remove(pos);
    }
}

void ObjectsManager::remove_object(Object* obj) noexcept
{
    this->remove_object(obj->get_uuid());
}

uint32_t ObjectsManager::add_light(LightType_ type) noexcept
{
    const uint32_t uuid = this->_uuid_counter++;

    switch(type)
    {
        case LightType_Square:
        {
            LightSquare* light = new LightSquare(uuid);
            this->_objects.emplace_back(std::move(new ObjectLight(light)));
            this->_objects.back()->_uuid = uuid;
            this->_objects.back()->set_name("square_light");
            break;
        }
        case LightType_Dome:
        {
            LightDome* light = new LightDome(uuid);
            this->_objects.emplace_back(std::move(new ObjectLight(light)));
            this->_objects.back()->_uuid = uuid;
            this->_objects.back()->set_name("dome_light");
            break;
        }
        case LightType_Distant:
        {
            LightDistant* light = new LightDistant(uuid);
            this->_objects.emplace_back(std::move(new ObjectLight(light)));
            this->_objects.back()->_uuid = uuid;
            this->_objects.back()->set_name("distant_light");
            break;
        }
        case LightType_Circle:
        {
            LightCircle* light = new LightCircle(uuid);
            this->_objects.emplace_back(std::move(new ObjectLight(light)));
            this->_objects.back()->_uuid = uuid;
            this->_objects.back()->set_name("circle_light");
            break;
        }
        case LightType_Spherical:
        {
            LightSpherical* light = new LightSpherical(uuid);
            this->_objects.emplace_back(std::move(new ObjectLight(light)));
            this->_objects.back()->_uuid = uuid;
            this->_objects.back()->set_name("spherical_light");
            break;
        }
        default:
            return INVALID_OBJECT_UUID;
    }

    return uuid;
}

bool ObjectsManager::get_objects_matching_pattern(ObjectsMatchingPatternIterator& it,
                                                  const std::regex& pattern,
                                                  Object** object) const noexcept
{
    if(it >= this->_objects.size())
    {
        return false;
    }

    for(uint32_t i = it; i < this->_objects.size(); i++)
    {
        std::cmatch cm;

        if(std::regex_search(this->_objects[i]->get_path().c_str(), cm, pattern))
        {
            *object = this->_objects[i];
            it = i + 1;

            return true;
        }
    }

    it = 0;

    return false;
}

Object* ObjectsManager::get_object_matching_uuid(const uint32_t uuid) const noexcept
{
    for(Object* obj : this->_objects)
    {
        if(obj->get_uuid() == uuid)
        {
            return obj;
        }
    }

    return nullptr;
}

size_t ObjectsManager::get_memory_usage() const noexcept
{
    size_t mem_usage = 0;

    for(const Object* obj : this->_objects)
    {
        mem_usage += obj->get_memory_usage();
    }

    return mem_usage;
}

ROMANORENDER_NAMESPACE_END
#include "romanorender/object.h"

#include "stdromano/char.h"
#include "stdromano/logger.h"
#include "stdromano/random.h"
#include "stdromano/threading.h"

#define STDROMANO_ENABLE_PROFILING
#include "stdromano/profiling.h"

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

ObjectMesh ObjectMesh::cube(const Vec3F& center, const Vec3F& scale) noexcept
{
    ObjectMesh cube;

    cube.get_vertices().push_back(
        Vec4F((center.x + scale.x / 2.0f), (center.y + scale.y / 2.0f), (center.z + scale.z / 2.0f), 0.0f));
    cube.get_vertices().push_back(
        Vec4F((center.x + scale.x / 2.0f), (center.y + scale.y / 2.0f), (center.z - scale.z / 2.0f), 0.0f));
    cube.get_vertices().push_back(
        Vec4F((center.x + scale.x / 2.0f), (center.y - scale.y / 2.0f), (center.z + scale.z / 2.0f), 0.0f));
    cube.get_vertices().push_back(
        Vec4F((center.x + scale.x / 2.0f), (center.y - scale.y / 2.0f), (center.z - scale.z / 2.0f), 0.0f));
    cube.get_vertices().push_back(
        Vec4F((center.x - scale.x / 2.0f), (center.y + scale.y / 2.0f), (center.z + scale.z / 2.0f), 0.0f));
    cube.get_vertices().push_back(
        Vec4F((center.x - scale.x / 2.0f), (center.y + scale.y / 2.0f), (center.z - scale.z / 2.0f), 0.0f));
    cube.get_vertices().push_back(
        Vec4F((center.x - scale.x / 2.0f), (center.y - scale.y / 2.0f), (center.z + scale.z / 2.0f), 0.0f));
    cube.get_vertices().push_back(
        Vec4F((center.x - scale.x / 2.0f), (center.y - scale.y / 2.0f), (center.z - scale.z / 2.0f), 0.0f));

    // Face 1: Front
    cube.get_indices().push_back(0);
    cube.get_indices().push_back(6);
    cube.get_indices().push_back(2);

    cube.get_indices().push_back(6);
    cube.get_indices().push_back(0);
    cube.get_indices().push_back(4);

    // Face 2: Back
    cube.get_indices().push_back(7);
    cube.get_indices().push_back(1);
    cube.get_indices().push_back(3);

    cube.get_indices().push_back(5);
    cube.get_indices().push_back(1);
    cube.get_indices().push_back(7);

    // Face 3: Top
    cube.get_indices().push_back(1);
    cube.get_indices().push_back(5);
    cube.get_indices().push_back(0);

    cube.get_indices().push_back(4);
    cube.get_indices().push_back(0);
    cube.get_indices().push_back(5);

    // Face 4: Bottom
    cube.get_indices().push_back(3);
    cube.get_indices().push_back(2);
    cube.get_indices().push_back(7);

    cube.get_indices().push_back(7);
    cube.get_indices().push_back(2);
    cube.get_indices().push_back(6);

    // Face 5: Right
    cube.get_indices().push_back(1);
    cube.get_indices().push_back(0);
    cube.get_indices().push_back(3);

    cube.get_indices().push_back(3);
    cube.get_indices().push_back(0);
    cube.get_indices().push_back(2);

    // Face 6: Left
    cube.get_indices().push_back(7);
    cube.get_indices().push_back(4);
    cube.get_indices().push_back(5);

    cube.get_indices().push_back(6);
    cube.get_indices().push_back(4);
    cube.get_indices().push_back(7);

    return std::move(cube);
}

ObjectMesh ObjectMesh::geodesic(const Vec3F& center, const Vec3F& scale, const uint32_t subdiv_level) noexcept
{
    ObjectMesh geodesic = std::move(ObjectMesh::cube(Vec3F(0.0f, 0.0f, 0.0f), Vec3F(2.0f, 2.0f, 2.0f)));

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
        stdromano::Vector<Vec4F> new_vertices = std::move(geodesic.get_vertices());

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

ObjectMesh ObjectMesh::plane(const Vec3F& center, const Vec3F& scale) noexcept
{
    ObjectMesh plane;

    plane.get_vertices().push_back(Vec4F(center.x + scale.x / 2.0f, center.y, center.z + scale.z / 2.0f, 0.0f));
    plane.get_vertices().push_back(Vec4F(center.x + scale.x / 2.0f, center.y, center.z - scale.z / 2.0f, 0.0f));
    plane.get_vertices().push_back(Vec4F(center.x - scale.x / 2.0f, center.y, center.z - scale.z / 2.0f, 0.0f));
    plane.get_vertices().push_back(Vec4F(center.x - scale.x / 2.0f, center.y, center.z + scale.z / 2.0f, 0.0f));

    plane.get_indices().push_back(0);
    plane.get_indices().push_back(1);
    plane.get_indices().push_back(2);

    plane.get_indices().push_back(2);
    plane.get_indices().push_back(3);
    plane.get_indices().push_back(0);

    return std::move(plane);
}

void ObjectMesh::build_blas() noexcept
{
    if(this->_indices.get().size() > 0)
    {
        this->_blas.Build(
            (tbvh::Vec4F*)this->_vertices.get().data(), this->_indices.get().data(), this->_indices.get().size() / 3);
    }
    else
    {
        this->_blas.Build((tbvh::Vec4F*)this->_vertices.get().data(), this->_vertices.get().size() / 3);
    }

    stdromano::log_debug("Built BLAS of object \"{}\". Bounds:\nmin({})\nmax({})",
                         this->get_name(),
                         this->_blas.aabbMin,
                         this->_blas.aabbMax);
}

void ObjectMesh::subdivide(const uint32_t subdiv_level) noexcept
{
    for(uint32_t level = 0; level < subdiv_level; ++level)
    {
        stdromano::Vector<uint32_t> old_indices = this->get_indices();
        stdromano::Vector<Vec4F> old_vertices = this->get_vertices();

        stdromano::Vector<Vec4F> new_vertices = old_vertices;
        stdromano::Vector<uint32_t> new_indices;
        std::map<std::pair<uint32_t, uint32_t>, uint32_t> edge_map;

        stdromano::Vector<Vec4F> updated_old_vertices(old_vertices.size());
        for(size_t i = 0; i < old_vertices.size(); ++i)
        {
            std::unordered_set<uint32_t> neighbors;
            for(size_t j = 0; j < old_indices.size(); j += 3)
            {
                for(int k = 0; k < 3; ++k)
                {
                    if(old_indices[j + k] == i)
                    {
                        neighbors.insert(old_indices[j + (k + 1) % 3]);
                        neighbors.insert(old_indices[j + (k + 2) % 3]);
                    }
                }
            }

            const float n = static_cast<float>(neighbors.size());
            const float beta
                = neighbors.size() == 3
                      ? 3.0f / 16.0f
                      : 1.0f / n
                            * (5.0f / 8.0f
                               - maths::powf(3.0f / 8.0f + 1.0f / 4.0f * maths::cosf(2.0f * maths::constants::pi / n),
                                             2));

            Vec4F updated_pos = old_vertices[i] * (1.0f - n * beta);
            for(auto neighbor : neighbors)
            {
                updated_pos += old_vertices[neighbor] * beta;
            }
            updated_old_vertices[i] = updated_pos;
        }

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
                    Vec4F sum_opposite(0.0f);
                    int count = 0;
                    for(size_t j = 0; j < old_indices.size(); j += 3)
                    {
                        if((old_indices[j] == edge.first && old_indices[j + 1] == edge.second)
                           || (old_indices[j] == edge.second && old_indices[j + 1] == edge.first))
                        {
                            sum_opposite += old_vertices[old_indices[j + 2]];
                            count++;
                        }
                        else if((old_indices[j + 1] == edge.first && old_indices[j + 2] == edge.second)
                                || (old_indices[j + 1] == edge.second && old_indices[j + 2] == edge.first))
                        {
                            sum_opposite += old_vertices[old_indices[j]];
                            count++;
                        }
                        else if((old_indices[j + 2] == edge.first && old_indices[j] == edge.second)
                                || (old_indices[j + 2] == edge.second && old_indices[j] == edge.first))
                        {
                            sum_opposite += old_vertices[old_indices[j + 1]];
                            count++;
                        }
                    }

                    Vec4F midpoint;

                    if(count == 2)
                    {
                        midpoint = (old_vertices[edge.first] + old_vertices[edge.second]) * 3.0f / 8.0f
                                   + sum_opposite * 1.0f / 8.0f;
                    }
                    else
                    {
                        midpoint = (old_vertices[edge.first] + old_vertices[edge.second]) * 0.5f;
                    }

                    uint32_t new_idx = new_vertices.size();
                    new_vertices.push_back(midpoint);
                    edge_map[edge] = new_idx;
                    mid[e] = new_idx;
                }
            }

            new_vertices[i0] = updated_old_vertices[i0];
            new_vertices[i1] = updated_old_vertices[i1];
            new_vertices[i2] = updated_old_vertices[i2];

            new_indices.push_back(i0);
            new_indices.push_back(mid[0]);
            new_indices.push_back(mid[2]);

            new_indices.push_back(mid[0]);
            new_indices.push_back(i1);
            new_indices.push_back(mid[1]);

            new_indices.push_back(mid[2]);
            new_indices.push_back(mid[1]);
            new_indices.push_back(i2);

            new_indices.push_back(mid[0]);
            new_indices.push_back(mid[1]);
            new_indices.push_back(mid[2]);
        }

        this->get_vertices() = std::move(new_vertices);
        this->get_indices() = std::move(new_indices);
    }
}

void ObjectMesh::add_attribute_buffer(const stdromano::String<>& name, AttributeBuffer& buffer) noexcept
{
    this->_attributes.insert(std::make_pair(name, Property<AttributeBuffer>(std::move(buffer))));
}

const AttributeBuffer* ObjectMesh::get_attribute_buffer(const stdromano::String<>& name) const noexcept
{
    const auto it = this->_attributes.find(name);

    return it == this->_attributes.end() ? nullptr : it->second.get_ptr();
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

Camera* ObjectCamera::get_camera() noexcept
{
    if(!this->_camera.initialized())
    {
        this->_camera.set(std::move(Camera()));
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

bool objects_from_obj_file(const char* file_path) noexcept
{
    SCOPED_PROFILE_START(stdromano::ProfileUnit::Seconds, obj_file_load);

    std::FILE* file_handle = std::fopen(file_path, "r");

    if(file_handle == nullptr)
    {
        stdromano::log_error("Error while trying to open file: {}", file_path);
        return false;
    }

    std::fseek(file_handle, 0, SEEK_END);
    const size_t file_size = std::ftell(file_handle);
    std::rewind(file_handle);
    stdromano::String<> file_content = std::move(stdromano::String<>::make_zeroed(file_size + 1));
    std::fread(file_content.c_str(), sizeof(char), file_size, file_handle);
    std::fclose(file_handle);

    file_content[file_size] = '\0';

    stdromano::String<> split;
    stdromano::String<>::split_iterator split_it = 0;

    size_t current_line = 0;

    stdromano::Vector<Vec4F> global_vertices;
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

                const auto res = std::from_chars(
                    start, start + (split.size() - j), global_vertices[global_vertices.size() - 1][i]);

                j += res.ptr - start;
            }

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

                ObjectsManager::get_instance().add_object(current_object);
                current_object = new ObjectMesh;
            }

            index_map.clear();
            current_object->set_name(
                std::move(stdromano::String<>("{}", fmt::string_view(split.data() + 2, split.size() - 2))));
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

        ObjectsManager::get_instance().add_object(current_object);
    }
    else
    {
        delete current_object;
    }

    ObjectsManager::get_instance().add_file_dependency(file_path);

    return true;
}

using namespace Alembic::AbcGeom;
using namespace Alembic::Abc;

void abc_traverse(IObject obj, const M44d& parentXform)
{
    if(IXform::matches(obj.getHeader()))
    {
        IXform xform(obj, kWrapExisting);
        XformSample xs;
        xform.getSchema().get(xs);
        M44d localXform = xs.getMatrix();
        M44d combinedXform = parentXform * localXform;

        for(size_t i = 0; i < obj.getNumChildren(); ++i)
        {
            abc_traverse(obj.getChild(i), combinedXform);
        }
    }
    else if(ICamera::matches(obj.getHeader()))
    {
        ICamera camera(obj, kWrapExisting);
        ICameraSchema schema = camera.getSchema();
        CameraSample sample;
        schema.get(sample);

        ObjectCamera* new_camera = new ObjectCamera;
        new_camera->set_name(stdromano::String<>("{}", obj.getName().c_str()));

        new_camera->set_focal(sample.getFocalLength());

        new_camera->set_transform(Mat44F(parentXform[0][0],
                                         parentXform[1][0],
                                         parentXform[2][0],
                                         parentXform[3][0],
                                         parentXform[0][1],
                                         parentXform[1][1],
                                         parentXform[2][1],
                                         parentXform[3][1],
                                         parentXform[0][2],
                                         parentXform[1][2],
                                         parentXform[2][2],
                                         parentXform[3][2],
                                         parentXform[0][3],
                                         parentXform[1][3],
                                         parentXform[2][3],
                                         parentXform[3][3]));

        stdromano::log_debug(
            "Found new camera: \"{}\" (pos: {})", new_camera->get_name(), new_camera->get_camera()->get_ray_origin());

        ObjectsManager::get_instance().add_object(new_camera);

        for(size_t i = 0; i < obj.getNumChildren(); ++i)
        {
            abc_traverse(obj.getChild(i), parentXform);
        }
    }
    else if(IPolyMesh::matches(obj.getHeader()))
    {
        IPolyMesh mesh(obj, kWrapExisting);
        IPolyMeshSchema& schema = mesh.getSchema();
        IPolyMeshSchema::Sample sample;
        schema.get(sample);

        const P3fArraySamplePtr positions = sample.getPositions();
        const Int32ArraySamplePtr faceIndices = sample.getFaceIndices();
        const Int32ArraySamplePtr faceCounts = sample.getFaceCounts();

        ObjectMesh* new_object = new ObjectMesh;
        new_object->set_name(stdromano::String<>("{}", obj.getName().c_str()));

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

        new_object->set_transform(Mat44F(parentXform[0][0],
                                         parentXform[1][0],
                                         parentXform[2][0],
                                         parentXform[3][0],
                                         parentXform[0][1],
                                         parentXform[1][1],
                                         parentXform[2][1],
                                         parentXform[3][1],
                                         parentXform[0][2],
                                         parentXform[1][2],
                                         parentXform[2][2],
                                         parentXform[3][2],
                                         parentXform[0][3],
                                         parentXform[1][3],
                                         parentXform[2][3],
                                         parentXform[3][3]));

        stdromano::log_debug(
            "Found new mesh \"{}\" ({} prims)", new_object->get_name(), new_object->get_indices().size() / 3);

        ObjectsManager::get_instance().add_object(new_object);

        for(size_t i = 0; i < obj.getNumChildren(); ++i)
        {
            abc_traverse(obj.getChild(i), parentXform);
        }
    }
    else
    {
        for(size_t i = 0; i < obj.getNumChildren(); ++i)
        {
            abc_traverse(obj.getChild(i), parentXform);
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

    abc_traverse(topObjectMesh, M44d());

    ObjectsManager::get_instance().add_file_dependency(file_path);

    return true;
}

ObjectsManager::ObjectsManager() {}

ObjectsManager::~ObjectsManager()
{
    for(Object* obj : this->_objects)
    {
        delete obj;
    }
}

ROMANORENDER_NAMESPACE_END
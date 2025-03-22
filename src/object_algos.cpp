#include "romanorender/object_algos.h"

ROMANORENDER_NAMESPACE_BEGIN

OBJECT_ALGOS_NAMESPACE_BEGIN

uint32_t get_or_create_midpoint(std::map<std::pair<uint32_t, uint32_t>, uint32_t>& edge_map,
                                uint32_t v1,
                                uint32_t v2,
                                const Vertices& old_vertices,
                                Vertices& new_vertices)
{
    std::pair<uint32_t, uint32_t> edge;

    if(v1 < v2)
    {
        edge = std::make_pair(v1, v2);
    }
    else
    {
        edge = std::make_pair(v2, v1);
    }

    auto it = edge_map.find(edge);

    if(it != edge_map.end())
    {
        return it->second;
    }

    const Vec4F& p1 = old_vertices[v1];
    const Vec4F& p2 = old_vertices[v2];
    Vec4F midpoint;
    midpoint.x = (p1.x + p2.x) * 0.5f;
    midpoint.y = (p1.y + p2.y) * 0.5f;
    midpoint.z = (p1.z + p2.z) * 0.5f;
    midpoint.w = 1.0f;

    uint32_t new_index = static_cast<uint32_t>(new_vertices.size());
    new_vertices.push_back(midpoint);
    edge_map[edge] = new_index;

    return new_index;
}

void apply_loop_smoothing(const Vertices& old_vertices,
                          Vertices& new_vertices,
                          const Indices& old_indices,
                          const std::map<std::pair<uint32_t, uint32_t>, uint32_t>& edge_map)
{
    stdromano::Vector<stdromano::Vector<uint32_t> > vertex_neighbors(old_vertices.size(),
                                                                     stdromano::Vector<uint32_t>());

    for(size_t i = 0; i < old_indices.size(); i += 3)
    {
        uint32_t v0 = old_indices[i];
        uint32_t v1 = old_indices[i + 1];
        uint32_t v2 = old_indices[i + 2];

        vertex_neighbors[v0].push_back(v1);
        vertex_neighbors[v0].push_back(v2);

        vertex_neighbors[v1].push_back(v0);
        vertex_neighbors[v1].push_back(v2);

        vertex_neighbors[v2].push_back(v0);
        vertex_neighbors[v2].push_back(v1);
    }

    for(auto& neighbors : vertex_neighbors)
    {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }

    for(size_t i = 0; i < old_vertices.size(); i++)
    {
        const stdromano::Vector<uint32_t>& neighbors = vertex_neighbors[i];
        size_t n = neighbors.size();

        if(n > 0)
        {
            float beta;
            if(n == 3)
                beta = 3.0f / 16.0f;
            else
                beta = 3.0f / (8.0f * n);

            Vec4F sum_neighbors = {0.0f, 0.0f, 0.0f, 0.0f};

            for(uint32_t neighbor : neighbors)
            {
                sum_neighbors.x += old_vertices[neighbor].x;
                sum_neighbors.y += old_vertices[neighbor].y;
                sum_neighbors.z += old_vertices[neighbor].z;
            }

            float original_weight = 1.0f - n * beta;
            Vec4F& new_pos = new_vertices[i];
            new_pos.x = original_weight * old_vertices[i].x + beta * sum_neighbors.x;
            new_pos.y = original_weight * old_vertices[i].y + beta * sum_neighbors.y;
            new_pos.z = original_weight * old_vertices[i].z + beta * sum_neighbors.z;
        }
    }
}

void subdivide(ObjectMesh* object, const uint32_t subdiv_level) noexcept
{
    for(uint32_t level = 0; level < subdiv_level; level++)
    {
        Vertices& old_vertices = object->get_vertices();
        Indices& old_indices = object->get_indices();
        Vertices new_vertices = old_vertices;
        Indices new_indices;

        std::map<std::pair<uint32_t, uint32_t>, uint32_t> edge_map;

        for(size_t i = 0; i < old_indices.size(); i += 3)
        {
            const uint32_t v0 = old_indices[i];
            const uint32_t v1 = old_indices[i + 1];
            const uint32_t v2 = old_indices[i + 2];

            const uint32_t m01 = get_or_create_midpoint(edge_map, v0, v1, old_vertices, new_vertices);
            const uint32_t m12 = get_or_create_midpoint(edge_map, v1, v2, old_vertices, new_vertices);
            const uint32_t m20 = get_or_create_midpoint(edge_map, v2, v0, old_vertices, new_vertices);

            new_indices.push_back(m01);
            new_indices.push_back(m12);
            new_indices.push_back(m20);

            new_indices.push_back(v0);
            new_indices.push_back(m01);
            new_indices.push_back(m20);

            new_indices.push_back(v1);
            new_indices.push_back(m12);
            new_indices.push_back(m01);

            new_indices.push_back(v2);
            new_indices.push_back(m20);
            new_indices.push_back(m12);
        }

        apply_loop_smoothing(old_vertices, new_vertices, old_indices, edge_map);

        object->set_vertices(new_vertices);
        object->set_indices(new_indices);
    }
}

void smooth_normals(ObjectMesh* object) noexcept
{
    if(object->get_vertex_attribute_buffer("N") != nullptr)
    {
        return;
    }

    AttributeBuffer N_buffer(AttributeBufferType_Normal,
                             AttributeBufferFormat_Float3,
                             sizeof(Vec3F),
                             object->get_vertices().size());

    Vec3F* Ns = N_buffer.get_data_ptr<Vec3F>();


    for(size_t i = 0; i < object->get_vertices().size(); ++i)
    {
        Ns[i] = Vec3F(0.0f, 0.0f, 0.0f);
    }

    const Vertices& vertices = object->get_vertices();
    const Indices& indices = object->get_indices();

    for(size_t i = 0; i < indices.size(); i += 3)
    {
        const uint32_t i1 = indices[i + 0];
        const uint32_t i2 = indices[i + 1];
        const uint32_t i3 = indices[i + 2];

        const Vec4F v1 = vertices[i1];
        const Vec4F v2 = vertices[i2];
        const Vec4F v3 = vertices[i3];

        const Vec3F p1(v1.x, v1.y, v1.z);
        const Vec3F p2(v2.x, v2.y, v2.z);
        const Vec3F p3(v3.x, v3.y, v3.z);

        const Vec3F edge1 = p2 - p1;
        const Vec3F edge2 = p3 - p1;
        const Vec3F face_normal = cross_vec3f(edge1, edge2);

        Ns[i1] += face_normal;
        Ns[i2] += face_normal;
        Ns[i3] += face_normal;
    }

    for(size_t i = 0; i < object->get_vertices().size(); ++i)
    {
        Ns[i] = normalize_vec3f(Ns[i]);
    }

    object->add_vertex_attribute_buffer("N", N_buffer);
}

OBJECT_ALGOS_NAMESPACE_END

ROMANORENDER_NAMESPACE_END
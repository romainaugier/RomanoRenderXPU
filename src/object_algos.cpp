#include "romanorender/object_algos.h"

#include "stdromano/hashset.hpp"

ROMANORENDER_NAMESPACE_BEGIN

OBJECT_ALGOS_NAMESPACE_BEGIN

using EdgeMap = stdromano::HashMap<Edge, uint32_t>;

uint32_t get_or_create_midpoint(EdgeMap& edge_map,
                                const uint32_t v1,
                                const uint32_t v2,
                                const Vertices& old_vertices,
                                Vertices& new_vertices)
{
    const Edge edge = v1 < v2 ? Edge(v1, v2) : Edge(v2, v1);

    auto it = edge_map.find(edge);

    if(it != edge_map.end())
    {
        return it->second;
    }

    Vec4F midpoint;
    midpoint.x = (old_vertices[v1].x + old_vertices[v2].x) * 0.5f;
    midpoint.y = (old_vertices[v1].y + old_vertices[v2].y) * 0.5f;
    midpoint.z = (old_vertices[v1].z + old_vertices[v2].z) * 0.5f;
    midpoint.w = 1.0f;

    const uint32_t new_index = static_cast<uint32_t>(new_vertices.size());
    new_vertices.push_back(midpoint);
    edge_map[edge] = new_index;

    return new_index;
}

void apply_loop_smoothing(const Vertices& old_vertices, 
                          Vertices& new_vertices,
                          const Indices& old_indices,
                          const EdgeMap& edge_map)
{
    stdromano::Vector<stdromano::Vector<uint32_t> > vertex_neighbors(old_vertices.size());
    stdromano::Vector<bool> is_boundary(old_vertices.size(), false);

    stdromano::HashSet<Edge> all_edges;
    stdromano::HashSet<Edge> boundary_edges;

    for(size_t i = 0; i < old_indices.size(); i += 3)
    {
        const uint32_t v0 = old_indices[i];
        const uint32_t v1 = old_indices[i + 1];
        const uint32_t v2 = old_indices[i + 2];

        Edge e01 = v0 < v1 ? Edge(v0, v1) : Edge(v1, v0);
        Edge e12 = v1 < v2 ? Edge(v1, v2) : Edge(v2, v1);
        Edge e20 = v2 < v0 ? Edge(v2, v0) : Edge(v0, v2);

        if(!all_edges.insert(e01).second)
            boundary_edges.erase(e01);
        else
            boundary_edges.insert(e01);

        if(!all_edges.insert(e12).second)
            boundary_edges.erase(e12);
        else
            boundary_edges.insert(e12);

        if(!all_edges.insert(e20).second)
            boundary_edges.erase(e20);
        else
            boundary_edges.insert(e20);

        vertex_neighbors[v0].push_back(v1);
        vertex_neighbors[v0].push_back(v2);

        vertex_neighbors[v1].push_back(v0);
        vertex_neighbors[v1].push_back(v2);

        vertex_neighbors[v2].push_back(v0);
        vertex_neighbors[v2].push_back(v1);
    }

    for(size_t i = 0; i < old_vertices.size(); i++)
    {
        auto& neighbors = vertex_neighbors[i];
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end() - 1);

        for(uint32_t neighbor : neighbors)
        {
            const Edge edge = i < neighbor ? Edge(i, neighbor) : Edge(neighbor, i);

            if(boundary_edges.find(edge) != boundary_edges.end())
            {
                is_boundary[i] = true;
                break;
            }
        }
    }

    for(size_t i = 0; i < old_vertices.size(); i++)
    {
        const auto& neighbors = vertex_neighbors[i];
        const size_t n = neighbors.size();

        if(n == 0)
        {
            continue;
        }

        if(is_boundary[i])
        {
            if(n >= 2)
            {
                uint32_t b1 = 0, b2 = 0;
                uint32_t boundary_count = 0;

                for(uint32_t neighbor : neighbors)
                {
                    const Edge edge = i < neighbor ? Edge(i, neighbor) : Edge(neighbor, i);

                    if(boundary_edges.find(edge) != boundary_edges.end())
                    {
                        if(boundary_count == 0)
                        {
                            b1 = neighbor;
                        }
                        else
                        {
                            b2 = neighbor;
                        }

                        boundary_count++;
                    }
                }

                if(boundary_count == 2)
                {
                    Vec4F& new_pos = new_vertices[i];
                    new_pos.x = 0.75f * old_vertices[i].x + 0.125f * old_vertices[b1].x
                                + 0.125f * old_vertices[b2].x;
                    new_pos.y = 0.75f * old_vertices[i].y + 0.125f * old_vertices[b1].y
                                + 0.125f * old_vertices[b2].y;
                    new_pos.z = 0.75f * old_vertices[i].z + 0.125f * old_vertices[b1].z
                                + 0.125f * old_vertices[b2].z;
                }
            }
        }
        else
        {
            float beta;

            if(n == 3)
            {
                beta = 3.0f / 16.0f;
            }
            else
            {
                beta = 1.0f / n
                       * (5.0f / 8.0f - maths::powf(3.0f / 8.0f + 1.0f / 4.0f * maths::cosf(2.0f * maths::constants::pi / n), 2));
            }

            Vec4F sum_neighbors = {0.0f, 0.0f, 0.0f, 0.0f};
            for(uint32_t neighbor : neighbors)
            {
                sum_neighbors.x += old_vertices[neighbor].x;
                sum_neighbors.y += old_vertices[neighbor].y;
                sum_neighbors.z += old_vertices[neighbor].z;
            }

            const float original_weight = 1.0f - n * beta;
            Vec4F& new_pos = new_vertices[i];
            new_pos.x = original_weight * old_vertices[i].x + beta * sum_neighbors.x;
            new_pos.y = original_weight * old_vertices[i].y + beta * sum_neighbors.y;
            new_pos.z = original_weight * old_vertices[i].z + beta * sum_neighbors.z;
        }
    }

    for(const auto& edge_entry : edge_map)
    {
        const auto& edge = edge_entry.first;
        const uint32_t midpoint_idx = edge_entry.second;
        const uint32_t v1 = edge.v1;
        const uint32_t v2 = edge.v2;

        const bool is_edge_boundary = boundary_edges.find(edge) != boundary_edges.end();

        if(is_edge_boundary)
        {
            new_vertices[midpoint_idx].x = (old_vertices[v1].x + old_vertices[v2].x) * 0.5f;
            new_vertices[midpoint_idx].y = (old_vertices[v1].y + old_vertices[v2].y) * 0.5f;
            new_vertices[midpoint_idx].z = (old_vertices[v1].z + old_vertices[v2].z) * 0.5f;
        }
        else
        {
            uint32_t opposite1 = UINT32_MAX;
            uint32_t opposite2 = UINT32_MAX;

            for(size_t i = 0; i < old_indices.size(); i += 3)
            {
                uint32_t a = old_indices[i];
                uint32_t b = old_indices[i + 1];
                uint32_t c = old_indices[i + 2];

                if((a == v1 && b == v2) || (a == v2 && b == v1) || (b == v1 && c == v2)
                   || (b == v2 && c == v1) || (c == v1 && a == v2) || (c == v2 && a == v1))
                {

                    uint32_t opposite = (a != v1 && a != v2) ? a : ((b != v1 && b != v2) ? b : c);

                    if(opposite1 == UINT32_MAX)
                    {
                        opposite1 = opposite;
                    }
                    else if(opposite2 == UINT32_MAX && opposite != opposite1)
                    {
                        opposite2 = opposite;
                        break;
                    }
                }
            }

            if(opposite1 != UINT32_MAX && opposite2 != UINT32_MAX)
            {
                new_vertices[midpoint_idx].x = 0.375f * (old_vertices[v1].x + old_vertices[v2].x)
                                               + 0.125f
                                                     * (old_vertices[opposite1].x
                                                        + old_vertices[opposite2].x);
                new_vertices[midpoint_idx].y = 0.375f * (old_vertices[v1].y + old_vertices[v2].y)
                                               + 0.125f
                                                     * (old_vertices[opposite1].y
                                                        + old_vertices[opposite2].y);
                new_vertices[midpoint_idx].z = 0.375f * (old_vertices[v1].z + old_vertices[v2].z)
                                               + 0.125f
                                                     * (old_vertices[opposite1].z
                                                        + old_vertices[opposite2].z);
            }
            else
            {
                new_vertices[midpoint_idx].x = (old_vertices[v1].x + old_vertices[v2].x) * 0.5f;
                new_vertices[midpoint_idx].y = (old_vertices[v1].y + old_vertices[v2].y) * 0.5f;
                new_vertices[midpoint_idx].z = (old_vertices[v1].z + old_vertices[v2].z) * 0.5f;
            }
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

        EdgeMap edge_map;
        apply_loop_smoothing(old_vertices, new_vertices, old_indices, edge_map);

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
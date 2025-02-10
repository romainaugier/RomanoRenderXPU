#include "romanorender/bvh.h"
#include "stdromano/logger.h"

ROMANORENDER_NAMESPACE_BEGIN

using get_bounds_func = void(*)(const void* geometry_data, uint32_t index, BBox* out);
using get_centroid_func = void(*)(const void* geometry_data, uint32_t index, Vec3F* out);

struct GeometryInterface
{
    get_bounds_func get_bounds;
    get_centroid_func get_centroid;

    const void* geometry_data;

    uint32_t num_elements;

    GeometryInterface(const get_bounds_func get_bounds,
                      const get_centroid_func get_centroid,
                      const void* geometry_data,
                      uint32_t num_elements) : 
                        get_bounds(get_bounds),
                        get_centroid(get_centroid),
                        geometry_data(geometry_data),
                        num_elements(num_elements) {}
};

/* TODO: do interfaces for all supported geometry types */

void point_get_bbox(const void* geometry_data, uint32_t index, BBox* out)
{
    const Geometry* geometry = (const Geometry*)geometry_data;
    const GeometryBuffer* position_buffer = geometry->get_geometry_buffers().at(0);

    const GeometryBuffer* radius_buffer = geometry->get_geometry_buffers_count() > 1 && 
                                          geometry->get_geometry_buffers().at(1)->get_type() == GeometryBufferType_VertexAttributeRadius ? 
                                          geometry->get_geometry_buffers().at(1) : nullptr;

    const Vec3F pos = ((const Vec3F*)position_buffer->get_geometry_ptr())[index];
    const float radius = radius_buffer == nullptr ? 1.0f : ((const float*)radius_buffer->get_geometry_ptr())[index];

    *out = BBox(pos + radius, pos - radius);
}

void point_get_centroid(const void* geometry_data, uint32_t index, Vec3F* out)
{
    const Geometry* geometry = (const Geometry*)geometry_data;
    const GeometryBuffer* position_buffer = geometry->get_geometry_buffers().at(0);

    const Vec3F* pos = std::addressof(((const Vec3F*)(position_buffer->get_geometry_ptr()))[index]);

    std::memcpy(out, pos, sizeof(Vec3F));
}

void triangle_get_bbox(const void* geometry_data, uint32_t index, BBox* out)
{
    ROMANORENDER_NOT_IMPLEMENTED;
}

void triangle_get_centroid(const void* geometry_data, uint32_t index, BBox* out)
{
    ROMANORENDER_NOT_IMPLEMENTED;
}

void Accelerator::PrimitiveBuffer::clear() noexcept
{
    std::memset(this->buffer, 0, this->capacity);
    this->size = 0;
}

uint32_t Accelerator::PrimitiveBuffer::add_triangle(uint32_t geom_id, 
                                                    uint32_t prim_id,
                                                    const Vec3F* v0, 
                                                    const Vec3F* v1, 
                                                    const Vec3F* v2) noexcept
{
    const uint32_t start = this->size;

    constexpr uint16_t stride = sizeof(PrimitiveHeader) + 3 * sizeof(Vec3F);
    this->check_capacity(stride);

    PrimitiveHeader* header = reinterpret_cast<PrimitiveHeader*>(buffer + size);
    header->stride = stride;
    header->type = GeometryType_Triangle;
    header->geom_id = geom_id;
    header->prim_id = prim_id;

    ::new(reinterpret_cast<PrimitiveTriangle*>(header + 1)) PrimitiveTriangle(v0, v1, v2);
    
    this->size += stride;

    return start;
}

uint32_t Accelerator::PrimitiveBuffer::add_point(uint32_t geom_id, 
                                                 uint32_t prim_id,
                                                 const Vec3F* center,
                                                 float radius) noexcept
{
    const uint32_t start = this->size;

    constexpr uint16_t stride = sizeof(PrimitiveHeader) + sizeof(float) + sizeof(Vec3F);
    this->check_capacity(stride);

    PrimitiveHeader* header = reinterpret_cast<PrimitiveHeader*>(buffer + size);
    header->stride = stride;
    header->type = GeometryType_Point;
    header->geom_id = geom_id;
    header->prim_id = prim_id;

    ::new(reinterpret_cast<PrimitivePoint*>(header + 1)) PrimitivePoint(center, &radius);
    
    this->size += stride;

    return start;
}

void Accelerator::PrimitiveBuffer::resize(size_t new_capacity) noexcept
{
    new_capacity = (new_capacity + ALIGNMENT-1) & ~(ALIGNMENT-1);
    uint8_t* new_buffer = static_cast<uint8_t*>(stdromano::mem_aligned_alloc(new_capacity, ALIGNMENT));

    if(buffer != nullptr) 
    {
        std::memcpy(new_buffer, buffer, size);
        stdromano::mem_aligned_free(buffer);
    }

    this->buffer = new_buffer;
    this->capacity = new_capacity;
}

struct BVHBuildNode
{
    static constexpr uint32_t BVH_BUILD_NODE_LEAF_MARKER = UINT32_MAX;

    BBox bounds;
    
    uint32_t left = BVH_BUILD_NODE_LEAF_MARKER;
    uint32_t right = BVH_BUILD_NODE_LEAF_MARKER;

    uint32_t primitives_offset = 0;
    uint8_t axis = 0;
    uint8_t num_primitives = 0;

    BVHBuildNode() {}

    BVHBuildNode(const BBox& bbox,
                 uint32_t left,
                 uint32_t right,
                 const uint32_t primitives_offset,
                 const uint8_t axis,
                 const uint8_t num_primitives) : 
                    bounds(bbox),
                    left(left),
                    right(right),
                    primitives_offset(primitives_offset),
                    axis(axis),
                    num_primitives(num_primitives) {}

    ROMANORENDER_FORCE_INLINE bool is_leaf() const noexcept {
        return this->left == BVH_BUILD_NODE_LEAF_MARKER && this->right == BVH_BUILD_NODE_LEAF_MARKER;
    }
};

struct PrimitiveRef
{
    const GeometryInterface* geom;
    uint32_t prim_id;

    PrimitiveRef(const GeometryInterface* geom, const uint32_t prim_id) : geom(geom), prim_id(prim_id) {}
};

struct BVHBuildContext
{
    stdromano::Vector<BVHBuildNode> nodes;
    stdromano::Vector<PrimitiveRef> ordered_prims;
    uint32_t root_node;
    uint32_t num_leaves;
};

bool build_bvh(BVHBuildContext* context, const GeometryInterface* geoms, const uint32_t num_geoms)
{
    stdromano::Vector<BBox> bounds;
    stdromano::Vector<Vec3F> centroids;
    stdromano::Vector<PrimitiveRef> references;

    for(uint32_t i = 0; i < num_geoms; i++)
    {
        const GeometryInterface* geom = std::addressof(geoms[i]);

        for(uint32_t j = 0; j < geom->num_elements; j++)
        {
            BBox bbox;
            geom->get_bounds(geom->geometry_data, j, &bbox);
            bounds.push_back(bbox);

            Vec3F centroid;
            geom->get_centroid(geom->geometry_data, j, &centroid);
            centroids.push_back(centroid);

            references.emplace_back(geom, j);
        }
    }

    std::function<uint32_t(uint32_t, uint32_t)> build_bvh_recursive = [&](uint32_t start, 
                                                                          uint32_t end) -> uint32_t 
    {
        const uint32_t node_index = context->nodes.size();
        context->nodes.emplace_back();

        context->nodes.at(node_index)->bounds = bounds[start];
        for(uint32_t i = (start + 1); i < end; i++)
        {
            context->nodes.at(node_index)->bounds.union_with(bounds[i]);
        }

        if((end - start) <= context->num_leaves)
        {
            context->nodes.at(node_index)->primitives_offset = start;
            context->nodes.at(node_index)->num_primitives = end - start;
            return node_index;
        }

        BBox centroid_bounds(centroids[start], centroids[start]);
        for(uint32_t i = start; i < end; i++)
        {
            centroid_bounds.union_with_vec3(centroids[i]);
        }

        const uint32_t split_dim = centroid_bounds.maximum_dimension();
        const float split_pos = (centroid_bounds.p0[split_dim] + centroid_bounds.p1[split_dim]) / 2.0f;

        uint32_t mid = start;
        for(uint32_t i = start; i < end; i++)
        {
            if(centroids[i][split_dim] < split_pos)
            {
                std::swap(bounds[i], bounds[mid]);
                std::swap(centroids[i], centroids[mid]);
                std::swap(references[i], references[mid]);
                mid++;
            }
        }

        context->nodes.at(node_index)->left = build_bvh_recursive(start, mid);
        context->nodes.at(node_index)->right = build_bvh_recursive(mid, end);

        return node_index;
    };

    context->root_node = build_bvh_recursive(0, bounds.size());
    context->ordered_prims = std::move(references);

    return true;
}

bool Accelerator::build(const Geometries& geometries, const uint32_t flags) noexcept
{
    stdromano::log_debug("Starting Accelerator build");

    stdromano::Vector<GeometryInterface> geom_interfaces;

    for(const Geometry& geom : geometries)
    {
        switch(geom.get_geometry_type())
        {
            case GeometryType_Point:
                geom_interfaces.emplace_back(point_get_bbox,
                                             point_get_centroid,
                                             (const void*)&geom,
                                             geom.get_geometry_buffers().at(0)->get_count());

                stdromano::log_debug("Added point geometry to accelerator");

                break;
            
            /* TODO: implement other interfaces */

            default:
                break;
        }
    }

    BVHBuildContext context;
    context.num_leaves = 8;

    bool success = build_bvh(&context, geom_interfaces.data(), geom_interfaces.size());

    if(!success)
    {
        stdromano::log_error("Error caught when building bvh, check the log for more information");
        return false;
    }

    const BVHBuildNode* root = context.nodes.at(context.root_node);

    stdromano::log_debug("BVH build successful");
    stdromano::log_debug("BVH bounds: {} -> {}", root->bounds.p0, root->bounds.p1);

    stdromano::log_debug("Creating ordered primitives buffer");

    this->primitives.clear();

    stdromano::Vector<uint32_t> primitive_offsets;

    primitive_offsets.push_back(0);

    for(const PrimitiveRef& prim : context.ordered_prims)
    {
        const Geometry* geom = (const Geometry*)prim.geom->geometry_data;
        const GeometryBuffers& buffers = geom->get_geometry_buffers();

        uint32_t offset = 0;
        switch (geom->get_geometry_type())
        {
            case GeometryType_Point:
            {
                const Vec3F* positions = (const Vec3F*)buffers[0].get_geometry_ptr();
                const float* radii = (const float*)buffers[1].get_geometry_ptr();
                offset = this->primitives.add_point(geom->get_id(),
                                                    prim.prim_id,
                                                    positions + prim.prim_id,
                                                    radii[prim.prim_id]);
                break;
            } 
            /* TODO: implement other primitives */
            case GeometryType_Triangle:
            {
                break;
            }
            default:
                continue;
        }

        primitive_offsets.push_back(offset);
    }

    char bytes_buffer[16];
    stdromano::format_byte_size((float)this->primitives.get_size(), bytes_buffer);

    stdromano::log_debug("Ordered primitives buffer created. Total size: {}", bytes_buffer);

    stdromano::log_debug("Flattening BVH");

    this->lnodes.clear();
    this->lnodes.reserve(context.nodes.size());

    std::function<uint32_t(uint32_t)> flatten = [&](uint32_t node_index)
    {
        const BVHBuildNode& node = context.nodes[node_index];

        BVHLinearNode lnode;
        lnode.bounds = node.bounds;

        if(node.is_leaf())
        {
            lnode.primitives_offset = primitive_offsets[node.primitives_offset];
            lnode.num_primitives = node.num_primitives;
        }
        else
        {
            lnode.axis = node.axis;
            lnode.second_child_offset = flatten(node.right);
            lnode.num_primitives = 0;
            flatten(node.left);
        }

        lnodes.push_back(lnode);

        return (uint32_t)(lnodes.size() - 1);
    };

    flatten(context.root_node);

    const BVHLinearNode* lroot = this->lnodes.at(context.root_node);

    stdromano::log_debug("BVH flattening successful");
    stdromano::log_debug("BVH bounds: {} -> {}", lroot->bounds.p0, lroot->bounds.p1);

    return true;
}

bool Accelerator::intersect(RayHit& rayhit) const noexcept
{
    uint32_t stack[TRAVERSAL_STACK_SIZE];
    int32_t stack_ptr = 0;
    bool hit_found = false;

    const bool dir_is_negative[3] = { rayhit.ray.inverse_direction.x < 0, 
                                      rayhit.ray.inverse_direction.y < 0,
                                      rayhit.ray.inverse_direction.z < 0 };

    stack[stack_ptr++] = 0;

    while(stack_ptr > 0)
    {
        const uint32_t current_node_index = stack[--stack_ptr];

        const BVHLinearNode* node = this->lnodes.at(current_node_index);

        if(intersect_bbox(node->bounds, rayhit.ray.origin, rayhit.ray.inverse_direction))
        {
            if(node->num_primitives == 0)
            {
                if(dir_is_negative[node->axis])
                {
                    stack[stack_ptr++] = node->second_child_offset;
                    stack[stack_ptr++] = current_node_index + 1;
                }
                else
                {
                    stack[stack_ptr++] = node->second_child_offset;
                    stack[stack_ptr++] = current_node_index + 1;
                }
            }
            else
            {
                uint8_t* prims = this->primitives.get_ptr_at(node->primitives_offset);

                for(uint32_t i = 0; i < node->num_primitives; i++)
                {
                    const PrimitiveBuffer::PrimitiveHeader* header = reinterpret_cast<const PrimitiveBuffer::PrimitiveHeader*>(prims);

                    switch(header->type)
                    {
                        case GeometryType_Point:
                        {
                            const PrimitivePoint* point = reinterpret_cast<const PrimitivePoint*>(header + 1);

                            if(Accelerator::intersect_point(point, rayhit))
                            {
                                rayhit.hit.geomID = header->geom_id;
                                rayhit.hit.primID = header->prim_id;
                                hit_found = true;
                            }

                            break;
                        }

                        case GeometryType_Triangle:
                        {
                            const PrimitiveTriangle* triangle = reinterpret_cast<const PrimitiveTriangle*>(header + 1);

                            if(Accelerator::intersect_triangle(triangle, rayhit))
                            {
                                rayhit.hit.geomID = header->geom_id;
                                rayhit.hit.primID = header->prim_id;
                                hit_found = true;
                            }

                            break;
                        }

                        /* TODO: add other geometry types */
                        
                        default:
                            break;
                    }

                    prims += header->stride;
                }

                if(hit_found)
                {
                    break;
                }
            }
        }
    }

    return hit_found;
}

bool Accelerator::occlude(RayHit& rayhit) const noexcept
{
    uint32_t stack[TRAVERSAL_STACK_SIZE];
    int32_t stack_ptr = 0;
    bool hit_found = false;

    const bool dir_is_negative[3] = { rayhit.ray.inverse_direction.x < 0, 
                                      rayhit.ray.inverse_direction.y < 0,
                                      rayhit.ray.inverse_direction.z < 0 };

    stack[stack_ptr++] = 0;

    while(stack_ptr > 0)
    {
        const uint32_t current_node_index = stack[--stack_ptr];

        const BVHLinearNode* node = this->lnodes.at(current_node_index);

        if(intersect_bbox(node->bounds, rayhit.ray.origin, rayhit.ray.inverse_direction))
        {
            if(node->num_primitives == 0)
            {
                if(dir_is_negative[node->axis])
                {
                    stack[stack_ptr++] = node->second_child_offset;
                    stack[stack_ptr++] = current_node_index + 1;
                }
                else
                {
                    stack[stack_ptr++] = node->second_child_offset;
                    stack[stack_ptr++] = current_node_index + 1;
                }
            }
            else
            {
                uint8_t* prims = this->primitives.get_ptr_at(node->primitives_offset);

                for(uint32_t i = 0; i < node->num_primitives; i++)
                {
                    const PrimitiveBuffer::PrimitiveHeader* header = reinterpret_cast<const PrimitiveBuffer::PrimitiveHeader*>(prims);

                    switch(header->type)
                    {
                        case GeometryType_Point:
                        {
                            const PrimitivePoint* point = reinterpret_cast<const PrimitivePoint*>(header + 1);

                            if(Accelerator::intersect_point(point, rayhit))
                            {
                                rayhit.hit.geomID = header->geom_id;
                                rayhit.hit.primID = header->prim_id;
                                hit_found = true;
                            }

                            break;
                        }

                        case GeometryType_Triangle:
                        {
                            const PrimitiveTriangle* triangle = reinterpret_cast<const PrimitiveTriangle*>(header + 1);

                            if(Accelerator::intersect_triangle(triangle, rayhit))
                            {
                                rayhit.hit.geomID = header->geom_id;
                                rayhit.hit.primID = header->prim_id;
                                hit_found = true;
                            }

                            break;
                        }

                        /* TODO: add other geometry types */
                        
                        default:
                            break;
                    }

                    prims += header->stride;
                }

                if(hit_found)
                {
                    break;
                }
            }
        }
    }

    return hit_found;
}
    

bool Accelerator::intersect_triangle(const PrimitiveTriangle* triangle, 
                                     RayHit& rayhit) noexcept
{
    ROMANORENDER_NOT_IMPLEMENTED;
    return false;
}

bool Accelerator::intersect_point(const PrimitivePoint* point,
                                  RayHit& rayhit) noexcept
{
    const Vec3F oc = rayhit.ray.origin - point->center;
    const float a = length2_vec3f(rayhit.ray.direction);
    const float b = dot_vec3f(oc, rayhit.ray.direction);
    const float c = length2_vec3f(oc) - point->radius * point->radius;
    const float discriminant = b * b - a * c;

    if(discriminant > 0)
    {
        const float t = (-b - sqrtf(discriminant)) / a;

        if(t > 0.0f && t < rayhit.ray.t)
        {
            rayhit.ray.t = t;
            rayhit.hit.pos = rayhit.ray.origin + (rayhit.ray.direction * t);
            rayhit.hit.normal = (rayhit.hit.pos - point->center) / point->radius;
            return true;
        }
    }

    return false;
}

ROMANORENDER_NAMESPACE_END
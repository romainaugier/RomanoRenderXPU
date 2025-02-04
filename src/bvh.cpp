#include "romanorender/bvh.h"
#include "romanorender/bbox.h"

ROMANORENDER_NAMESPACE_BEGIN

using get_bounds_func = void(*)(const void* geometry_data, uint32_t index, BBox* out);
using get_centroid_func = void(*)(const void* geometry_data, uint32_t index, Vec3F* out);

struct GeometryInterface
{
    get_bounds_func get_bounds;
    get_centroid_func get_centroid;

    const void* geometry_data;

    uint32_t num_elements;
};

ROMANORENDER_NAMESPACE_END
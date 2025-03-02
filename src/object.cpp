#include "romanorender/object.h"

ROMANORENDER_NAMESPACE_BEGIN

Object Object::cube(const Vec3F& center, const Vec3F& scale) noexcept
{
    Object cube;

    cube.get_vertices().push_back(Vec4F((center.x + scale.x), (center.y + scale.y), (center.z + scale.z), 0.0f));
    cube.get_vertices().push_back(Vec4F((center.x + scale.x), (center.y + scale.y), -(center.z + scale.z), 0.0f));
    cube.get_vertices().push_back(Vec4F((center.x + scale.x), -(center.y + scale.y), (center.z + scale.z), 0.0f));
    cube.get_vertices().push_back(Vec4F((center.x + scale.x), -(center.y + scale.y), -(center.z + scale.z), 0.0f));
    cube.get_vertices().push_back(Vec4F(-(center.x + scale.x), (center.y + scale.y), (center.z + scale.z), 0.0f));
    cube.get_vertices().push_back(Vec4F(-(center.x + scale.x), (center.y + scale.y), -(center.z + scale.z), 0.0f));
    cube.get_vertices().push_back(Vec4F(-(center.x + scale.x), -(center.y + scale.y), (center.z + scale.z), 0.0f));
    cube.get_vertices().push_back(Vec4F(-(center.x + scale.x), -(center.y + scale.y), -(center.z + scale.z), 0.0f));

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

bool objects_from_obj_file(const char* file_path, stdromano::Vector<Object>& object) noexcept
{
    return true;
}

ROMANORENDER_NAMESPACE_END
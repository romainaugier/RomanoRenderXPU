#include "romanorender/bvh.h"

#define STDROMANO_ENABLE_PROFILING
#include "stdromano/profiling.h"

using namespace romanorender;

int main()
{
    stdromano::set_log_level(stdromano::LogLevel::Debug);

    Geometry single_sphere(GeometryType_Point);
    Vec3F* positions = (Vec3F*)single_sphere.add_geometry_buffer(GeometryBufferType_Vertex, 
                                                                 GeometryBufferFormat_Float3,
                                                                 sizeof(Vec3F),
                                                                 1);
    positions[0].x = -3.0f;
    positions[0].y = 3.0f;
    positions[0].z = -6.0f;

    float* radius = (float*)single_sphere.add_geometry_buffer(GeometryBufferType_VertexAttributeRadius,
                                                              GeometryBufferFormat_Float1,
                                                              sizeof(float),
                                                              1);
    radius[0] = 10.0f;

    Geometries geoms(1);
    geoms.push_back(single_sphere);

    Accelerator accelerator;

    if(!accelerator.build(geoms, 0))
    {
        return 1;
    }

    return 0;
}
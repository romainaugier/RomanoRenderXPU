#include "romanorender/scene.h"

#include "stdromano/random.h"

#define STDROMANO_ENABLE_PROFILING
#include "stdromano/profiling.h"

using namespace romanorender;

#define NUM_POINTS 1500

#define RANDOM_FLOAT_11 ((stdromano::next_random_float_01() - 0.5f) * 2.0f)

int main()
{
    stdromano::set_log_level(stdromano::LogLevel::Debug);

    Scene scene;

    Geometry single_point(GeometryType_Point);
    Vec3F* positions = (Vec3F*)single_point.add_geometry_buffer(GeometryBufferType_Vertex, 
                                                                GeometryBufferFormat_Float3,
                                                                sizeof(Vec3F),
                                                                1);
    positions[0].x = 0.0f;
    positions[0].y = 0.0f;
    positions[0].z = -10.0f;

    float* radius = (float*)single_point.add_geometry_buffer(GeometryBufferType_VertexAttributeRadius,
                                                             GeometryBufferFormat_Float1,
                                                             sizeof(float),
                                                             1);
    radius[0] = 2.0f;

    Geometries geoms(1);
    geoms.push_back(single_point);

    const uint32_t geom_id = scene.attach_geometry(std::move(single_point));

    if(!scene.build())
    {
        return 1;
    }

    RayHit test_ray = initialize_rayhit(Vec3F(0.0f), Vec3F(0.0f, 0.0f, -1.0f), 10000.0f);

    if(!scene.occlude(test_ray))
    {
        return 1;
    }

    return 0;
}
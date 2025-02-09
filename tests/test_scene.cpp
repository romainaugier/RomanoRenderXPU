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

    Geometry points(GeometryType_Point);
    Vec3F* positions = (Vec3F*)points.add_geometry_buffer(GeometryBufferType_Vertex, 
                                                          GeometryBufferFormat_Float3,
                                                          sizeof(Vec3F),
                                                          NUM_POINTS);
    for(uint32_t i = 0; i < NUM_POINTS; i++)
    {
        positions[i].x = RANDOM_FLOAT_11 * 100.0f;
        positions[i].y = RANDOM_FLOAT_11 * 100.0f;
        positions[i].z = RANDOM_FLOAT_11 * 100.0f;
    }

    float* radius = (float*)points.add_geometry_buffer(GeometryBufferType_VertexAttributeRadius,
                                                       GeometryBufferFormat_Float1,
                                                       sizeof(float),
                                                       NUM_POINTS);
    for(uint32_t i = 0; i < NUM_POINTS; i++)
    {
        radius[0] = (stdromano::next_random_float_01() + 0.1f) * 0.5f;
    }

    const uint32_t geom_id = scene.attach_geometry(std::move(points));

    if(!scene.build())
    {
        return 1;
    }

    return 0;
}
#include "romanorender/scene.h"

#include "stdromano/random.h"

#define STDROMANO_ENABLE_PROFILING
#include "stdromano/profiling.h"

using namespace romanorender;

int main()
{
    stdromano::set_log_level(stdromano::LogLevel::Debug);

    Scene scene;

    Object cube = Object::cube(Vec3F(0.0f), Vec3F(1.0f));
    cube.build_blas();

    scene.add_object(cube);
    scene.build_tlas();

    tinybvh::Ray ray(tinybvh::bvhvec3(0.0f, 0.0f, -5.0f), tinybvh::bvhvec3(0.0f, 0.0f, 1.0f));

    if(scene.intersect(ray) == 0)
    {
        return 1;
    }

    return 0;
}
#include "romanorender/mat44.h"

#define STDROMANO_ENABLE_PROFILING
#include "stdromano/profiling.h"

using namespace romanorender;

int main(int argc, char** argv)
{
    stdromano::set_log_level(stdromano::LogLevel::Debug);

    Mat44F ident;
    ident.debug();

    Mat44F A(1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f);
    Mat44F B(1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f, 3.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f);

    A.debug();
    B.debug();

    SCOPED_PROFILE_START(stdromano::ProfileUnit::MicroSeconds, mat44f_mul);

    Mat44F C = mat44f_mul(A, B);

    SCOPED_PROFILE_STOP(mat44f_mul);

    C.debug();

    Mat44F lkt = Mat44F::from_lookat(Vec3F(0.0f, 0.0f, 0.0f), Vec3F(0.0f, 0.0f, 10.0f));

    lkt.debug();

    return 0;
}
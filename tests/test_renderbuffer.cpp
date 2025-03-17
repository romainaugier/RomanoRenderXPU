#include "romanorender/renderbuffer.h"

#include "stdromano/random.h"

#define STDROMANO_ENABLE_PROFILING
#include "stdromano/profiling.h"

using namespace romanorender;

int main()
{
    stdromano::set_log_level(stdromano::LogLevel::Debug);

    RenderBuffer buffer(1280, 720, 64, true);

    for(auto& bucket : buffer.get_buckets())
    {
        Vec4F bucket_color(stdromano::pcg_float(bucket.get_id() + 0),
                           stdromano::pcg_float(bucket.get_id() + 1),
                           stdromano::pcg_float(bucket.get_id() + 2),
                           1.0f);

        bucket.set_pixels(&bucket_color);
    }

    if(!buffer.to_jpg("test_output.jpg"))
    {
        return 1;
    }

    return 0;
}
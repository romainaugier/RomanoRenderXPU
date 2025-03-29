#include "romanorender/sampling.h"

#define STDROMANO_ENABLE_PROFILING
#include "stdromano/profiling.h"

using namespace romanorender;

#define NUM_SAMPLES 1024 * 4 * 4 * 4

int main()
{
    stdromano::set_log_level(stdromano::LogLevel::Debug);

    SCOPED_PROFILE_START(stdromano::ProfileUnit::Seconds, generate_pmj02_samples);
    const CudaVector<Vec2F>& samples = get_pmj02_samples(NUM_SAMPLES, 2, 0x123456);
    SCOPED_PROFILE_STOP(generate_pmj02_samples);

    stdromano::log_debug("Generated {} samples", samples.size());

    return 0;
}
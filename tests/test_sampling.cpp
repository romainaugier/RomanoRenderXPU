#include "romanorender/sampling.h"

#define STDROMANO_ENABLE_PROFILING
#include "stdromano/profiling.h"

using namespace romanorender;

int main()
{
    stdromano::set_log_level(stdromano::LogLevel::Debug);

    SCOPED_PROFILE_START(stdromano::ProfileUnit::Seconds, generate_32_pmj02_samples);
    stdromano::Vector<Vec2F> pmj02_samples = std::move(generate_pmj02_samples(32, 0xABCDEF));
    SCOPED_PROFILE_STOP(generate_32_pmj02_samples);

    for(uint32_t i = 0; i < pmj02_samples.size(); i++)
    {
        stdromano::log_debug("{0}: {1}", i, pmj02_samples[i]);
    }

    return 0;
}
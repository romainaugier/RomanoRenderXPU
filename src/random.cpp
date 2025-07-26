#include "romanorender/random.h"

#include "stdromano/threading.hpp"

#include <random>
#include <thread>

ROMANORENDER_NAMESPACE_BEGIN

/* Per thread seed */

static thread_local bool seeded = false;

static thread_local uint64_t pcg_state = 0x853c49e6748fea9bULL;
static thread_local uint64_t pcg_inc = 0xda3e39cb94b95bdbULL | 1;

static thread_local uint64_t xoshiro_s[4] = {0x123456789abcdef0ULL, 0x42, 0x1337, 0xdeadbeef};

void seed_thread_rngs() noexcept
{
    std::hash<size_t> hasher;
    uint64_t thread_seed = hasher(stdromano::thread_get_id());

    std::random_device rd;
    thread_seed ^= static_cast<uint64_t>(rd()) << 32;

    pcg_state = thread_seed;
    pcg_inc = (thread_seed ^ 0xda3e39cb94b95bdbULL) | 1;

    pcg_state = pcg_state * 6364136223846793005ULL + pcg_inc;

    xoshiro_s[0] = thread_seed;

    for(int i = 1; i < 4; i++)
    {
        xoshiro_s[i] = xoshiro_s[i - 1] + 0x9e3779b97f4a7c15ULL;
        xoshiro_s[i] = (xoshiro_s[i] ^ (xoshiro_s[i] >> 30)) * 0xbf58476d1ce4e5b9ULL;
        xoshiro_s[i] = (xoshiro_s[i] ^ (xoshiro_s[i] >> 27)) * 0x94d049bb133111ebULL;
        xoshiro_s[i] = xoshiro_s[i] ^ (xoshiro_s[i] >> 31);
    }
}

ROMANORENDER_FORCE_INLINE uint64_t rotl(const uint64_t x, int k)
{
    return (x << k) | (x >> (64 - k));
}

constexpr uint64_t tofloat64 = 0x2f80000000000004ULL;
constexpr uint32_t tofloat32 = 0x2f800004UL;

uint32_t pcg_next_uint32() noexcept
{
    uint64_t oldstate = pcg_state;
    pcg_state = oldstate * 6364136223846793005ULL + pcg_inc;
    uint32_t xorshifted = static_cast<uint32_t>(((oldstate >> 18u) ^ oldstate) >> 27u);
    uint32_t rot = oldstate >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

float pcg_next_float() noexcept
{
    return static_cast<float>(pcg_next_uint32()) * reinterpret_cast<const float&>(tofloat32);
}

uint64_t xoshiro_next_uint64() noexcept
{
    const uint64_t result = rotl(xoshiro_s[0] + xoshiro_s[3], 23) + xoshiro_s[0];
    const uint64_t t = xoshiro_s[1] << 17;

    xoshiro_s[2] ^= xoshiro_s[0];
    xoshiro_s[3] ^= xoshiro_s[1];
    xoshiro_s[1] ^= xoshiro_s[2];
    xoshiro_s[0] ^= xoshiro_s[3];

    xoshiro_s[2] ^= t;
    xoshiro_s[3] = rotl(xoshiro_s[3], 45);

    return result;
}

float xoshiro_next_float() noexcept
{
    return static_cast<float>(xoshiro_next_uint64() >> 32) * reinterpret_cast<const float&>(tofloat32);
}

void seed_pcg(uint64_t seed) noexcept
{
    pcg_state = seed;
    pcg_inc = (seed + 1) | 1;
    pcg_state = pcg_state * 6364136223846793005ULL + pcg_inc;
}

void seed_xoshiro(uint64_t seed) noexcept
{
    xoshiro_s[0] = seed;

    for(int i = 1; i < 4; i++)
    {
        xoshiro_s[i] = xoshiro_s[i - 1] + 0x9e3779b97f4a7c15ULL;
        xoshiro_s[i] = (xoshiro_s[i] ^ (xoshiro_s[i] >> 30)) * 0xbf58476d1ce4e5b9ULL;
        xoshiro_s[i] = (xoshiro_s[i] ^ (xoshiro_s[i] >> 27)) * 0x94d049bb133111ebULL;
        xoshiro_s[i] = xoshiro_s[i] ^ (xoshiro_s[i] >> 31);
    }
}

/* User seed */

uint32_t __pcg_random_uint32(uint32_t value) noexcept
{
    uint64_t state = value;
    uint64_t inc = (value | 1) ^ 0xda3e39cb94b95bdbULL;
    state = state * 6364136223846793005ULL + inc;
    uint32_t xorshifted = static_cast<uint32_t>(((state >> 18u) ^ state) >> 27u);
    uint32_t rot = state >> 59u;
    return (xorshifted >> rot) | (xorshifted << ((-rot) & 31));
}

uint64_t __xoshiro_random_uint64(uint64_t value) noexcept
{
    uint64_t s[4];
    s[0] = value;

    for(int i = 1; i < 4; i++)
    {
        s[i] = s[i - 1] + 0x9e3779b97f4a7c15ULL;
        s[i] = (s[i] ^ (s[i] >> 30)) * 0xbf58476d1ce4e5b9ULL;
        s[i] = (s[i] ^ (s[i] >> 27)) * 0x94d049bb133111ebULL;
        s[i] = s[i] ^ (s[i] >> 31);
    }

    const uint64_t result = rotl(s[0] + s[3], 23) + s[0];
    const uint64_t t = s[1] << 17;

    s[2] ^= s[0];
    s[3] ^= s[1];
    s[1] ^= s[2];
    s[0] ^= s[3];

    s[2] ^= t;
    s[3] = rotl(s[3], 45);

    return result;
}

uint32_t pcg_random_uint32(uint32_t value) noexcept { return __pcg_random_uint32(value); }

uint64_t xoshiro_random_uint64(uint64_t value) noexcept { return __xoshiro_random_uint64(value); }

ROMANORENDER_NAMESPACE_END
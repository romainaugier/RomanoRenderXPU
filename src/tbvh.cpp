#define TINYBVH_IMPLEMENTATION
#include "romanorender/tbvh.h"

namespace tbvh {

Vec4F::Vec4F(const Vec3F& a, float b) : x(a.x), y(a.y), z(a.z), w(b) {}

}
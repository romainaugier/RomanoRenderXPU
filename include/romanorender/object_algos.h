#pragma once

#define OBJECT_ALGOS_NAMESPACE_BEGIN namespace object_algos {
#define OBJECT_ALGOS_NAMESPACE_END }

#if !defined(__ROMANORENDER_OBJECT_ALGOS)
#define __ROMANORENDER_OBJECT_ALGOS

#include "romanorender/object.h"

ROMANORENDER_NAMESPACE_BEGIN

OBJECT_ALGOS_NAMESPACE_BEGIN

struct Edge 
{
    uint32_t v1 = 0xFFFFFFFF;
    uint32_t v2 = 0xFFFFFFFF;

    Edge() = default;
    Edge(const uint32_t v1, const uint32_t v2) : v1(v1), v2(v2) {}

    bool operator==(const Edge& other) const noexcept 
    {
        return this->v1 == other.v1 && this->v2 == other.v2;
    }
};

ROMANORENDER_API void subdivide(ObjectMesh* object, const uint32_t subdiv_level) noexcept;

ROMANORENDER_API void smooth_normals(ObjectMesh* object) noexcept;

OBJECT_ALGOS_NAMESPACE_END

ROMANORENDER_NAMESPACE_END

namespace std {
    template<>
    struct hash<romanorender::object_algos::Edge>
    {
        std::size_t operator()(const romanorender::object_algos::Edge& edge) const
        {
            return stdromano::hash_murmur3((void*)std::addressof(edge), sizeof(romanorender::object_algos::Edge), 0xAAAA);
            // return stdromano::hash_murmur_64(*(reinterpret_cast<int64_t*>(std::addressof(edge))));
        }
    };
}

#endif /* !defined(__ROMANORENDER_OBJECT_ALGOS) */
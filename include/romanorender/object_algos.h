#pragma once

#define OBJECT_ALGOS_NAMESPACE_BEGIN namespace object_algos {
#define OBJECT_ALGOS_NAMESPACE_END }

#if !defined(__ROMANORENDER_OBJECT_ALGOS)
#define __ROMANORENDER_OBJECT_ALGOS

#include "romanorender/object.h"

ROMANORENDER_NAMESPACE_BEGIN

OBJECT_ALGOS_NAMESPACE_BEGIN

ROMANORENDER_API void subdivide(ObjectMesh* object, const uint32_t subdiv_level) noexcept;

ROMANORENDER_API void smooth_normals(ObjectMesh* object) noexcept;

OBJECT_ALGOS_NAMESPACE_END

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_OBJECT_ALGOS) */
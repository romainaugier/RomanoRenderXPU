#pragma once

#if !defined(__ROMANORENDER_APP)
#define __ROMANORENDER_APP

#include "romanorender/scenegraph.h"

ROMANORENDER_NAMESPACE_BEGIN

ROMANORENDER_API int application(int argc, char** argv);

ROMANORENDER_API void draw_scenegraph(SceneGraph& graph) noexcept;

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_APP) */
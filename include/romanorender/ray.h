#pragma once

#if !defined(__ROMANORENDER_RAY)
#define __ROMANORENDER_RAY

#include "romanorender/romanorender.h"

ROMANORENDER_NAMESPACE_BEGIN

enum VisibilityFlag_ : uint8_t
{
    VisibilityFlag_VisiblePrimaryRays = 0x1,
    VisibilityFlag_VisibleSecondaryRays = 0x2,
    VisibilityFlag_VisibleShadowRays = 0x4,
    VisibilityFlag_VisiblsBRDFRays = 0x8,
    VisibilityFlag_VisibleBTDFRays = 0x10,
    VisibilityFlag_VisibleAllRays = VisibilityFlag_VisiblePrimaryRays |
                                    VisibilityFlag_VisibleSecondaryRays |
                                    VisibilityFlag_VisibleShadowRays,
};

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_RAY) */
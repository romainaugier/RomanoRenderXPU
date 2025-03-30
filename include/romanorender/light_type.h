#pragma once

#if !defined(__ROMANORENDER_LIGHT_TYPE)
#define __ROMANORENDER_LIGHT_TYPE

#include "romanorender/romanorender.h"

ROMANORENDER_NAMESPACE_BEGIN

enum LightType_ : uint32_t 
{
    LightType_Square,
    LightType_Dome,
    LightType_Distant,
    LightType_Circle,
    LightType_Spherical,
};

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_LIGHT_TYPE) */
#pragma once 

#if !defined(__ROMANORENDER_COLOR)
#define __ROMANORENDER_COLOR

#include "romanorender/vec4.h"

ROMANORENDER_NAMESPACE_BEGIN

ROMANORENDER_FORCE_INLINE Vec4F gamma_correct(const Vec4F& color, const float gamma = 2.2f)
{
    const float inv_gamma = maths::rcp_safef(gamma);
    Vec4F new_color = pow_vec4f(color, inv_gamma);

    new_color.w = color.w;

    return new_color;
}

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_COLOR) */
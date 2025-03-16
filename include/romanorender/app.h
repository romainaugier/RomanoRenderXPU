#pragma once

#if !defined(__ROMANORENDER_APP)
#define __ROMANORENDER_APP

#include "romanorender/scenegraph.h"

#include <GLFW/glfw3.h>

ROMANORENDER_NAMESPACE_BEGIN

ROMANORENDER_API void glfw_error_callback(int error, const char* desc) noexcept;

ROMANORENDER_API void glfw_drop_event_callback(GLFWwindow* user_ptr, int count, const char** paths) noexcept;

ROMANORENDER_API void glfw_key_event_callback(GLFWwindow* user_ptr, int key, int scancode, int action, int mods) noexcept;

ROMANORENDER_API int application(int argc, char** argv);

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_APP) */
#pragma once

#if !defined(__ROMANORENDER_APP)
#define __ROMANORENDER_APP

#include "romanorender/scenegraph.h"

#include <GLFW/glfw3.h>

ROMANORENDER_NAMESPACE_BEGIN

ROMANORENDER_API void glfw_error_callback(int error, const char* desc) noexcept;

ROMANORENDER_API void glfw_drop_event_callback(GLFWwindow* window, int count, const char** paths) noexcept;

ROMANORENDER_API void glfw_key_event_callback(GLFWwindow* window, int key, int scancode, int action, int mods) noexcept;

ROMANORENDER_API void glfw_scroll_callback(GLFWwindow* window, double x_offset, double y_offset) noexcept;

ROMANORENDER_API void glfw_mouse_button_callback(GLFWwindow* window, int button, int action, int mods) noexcept;

ROMANORENDER_API void glfw_cursor_pos_callback(GLFWwindow* window, double x_pos, double y_pos) noexcept;

ROMANORENDER_API int application(int argc, char** argv);

ROMANORENDER_NAMESPACE_END

#endif /* !defined(__ROMANORENDER_APP) */
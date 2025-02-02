# Global

include(CMakePrintHelpers)

cmake_minimum_required(VERSION 3.15)

project(RomanoRender VERSION ${VERSION})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_VERBOSE_MAKEFILE ON)

set(CMAKE_CXX_STANDARD 11)

find_package(glew REQUIRED)

find_package(glfw REQUIRED)

set(stdromano_DIR "${CMAKE_SOURCE_DIR}/stdromano/install/cmake")
find_package(stdromano REQUIRED)

set(IMGUI_INCLUDES dependencies/imgui)
include_directories(${IMGUI_INCLUDES})

set(APPNAME ${PROJECT_NAME})
set(LIBNAME "lib${PROJECT_NAME}")

add_subdirectory(dependencies)

add_subdirectory(src)
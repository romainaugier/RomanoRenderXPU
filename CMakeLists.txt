# Global

include(CMakePrintHelpers)

cmake_minimum_required(VERSION 3.15)

project(RomanoRender VERSION ${VERSION})

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_VERBOSE_MAKEFILE ON)

if(RUN_TESTS EQUAL 1)
    message(STATUS "RUN_TESTS enabled, building and running tests")

    set(TESTS_DATA_DIR "${CMAKE_SOURCE_DIR}/tests_data")
    add_definitions(-DTESTS_DATA_DIR="${TESTS_DATA_DIR}")

    enable_testing()
endif()

find_package(glew CONFIG REQUIRED)

find_package(glfw3 CONFIG REQUIRED)

set(stdromano_DIR "${CMAKE_SOURCE_DIR}/stdromano/install/cmake")
find_package(stdromano REQUIRED)

set(IMGUI_INCLUDES dependencies/imgui)
include_directories(${IMGUI_INCLUDES})

set(TINYBVH_INCLUDES tinybvh)
include_directories(${TINYBVH_INCLUDES})

set(APPNAME "${PROJECT_NAME}")
set(LIBNAME "lib${PROJECT_NAME}")

add_subdirectory(dependencies)

add_subdirectory(src)

if(RUN_TESTS EQUAL 1)
    add_subdirectory(tests)
endif()
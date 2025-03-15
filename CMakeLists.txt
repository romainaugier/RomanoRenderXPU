# Global

include(CMakePrintHelpers)

cmake_minimum_required(VERSION 3.15)

project(RomanoRender VERSION ${VERSION} LANGUAGES CXX CUDA)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

set(CMAKE_VERBOSE_MAKEFILE ON)

if(RUN_TESTS EQUAL 1)
    message(STATUS "RUN_TESTS enabled, building and running tests")

    set(TESTS_DATA_DIR "${CMAKE_SOURCE_DIR}/tests_data")
    add_definitions(-DTESTS_DATA_DIR="${TESTS_DATA_DIR}")

    enable_testing()
endif()

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

set(CUDA_ATTACH_VS_BUILD_RULE_TO_CUDA_FILE OFF)

find_package(CUDAToolkit REQUIRED)

if(WIN32)
    set(OptiX_INCLUDE_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.0.0/include")
elseif(UNIX)
    set(OptiX_INCLUDE_DIR "/home/donromano/Documents/OptiXSDK_9.0.0/include")
endif()

include_directories(${OptiX_INCLUDE_DIR})

find_package(glew CONFIG REQUIRED)

find_package(glfw3 CONFIG REQUIRED)

find_package(Alembic CONFIG REQUIRED)

find_package(imgui CONFIG REQUIRED)

set(stdromano_DIR "${CMAKE_SOURCE_DIR}/stdromano/install/cmake")
find_package(stdromano REQUIRED)

set(TINYBVH_INCLUDES tinybvh)
include_directories(${TINYBVH_INCLUDES})

set(APPNAME "${PROJECT_NAME}")
set(LIBNAME "lib${PROJECT_NAME}")
set(PTXNAME "ptx${PROJECT_NAME}")
set(PTX_OUTPUT_FILE "${CMAKE_BINARY_DIR}/src/CMakeFiles/${PTXNAME}.dir/shaders/shaders.ptx")

set(IMNODES_IMGUI_TARGET_NAME imgui::imgui)
include_directories(imnodes)
add_subdirectory(imnodes)

add_subdirectory(src)

if(RUN_TESTS EQUAL 1)
    add_subdirectory(tests)
endif()
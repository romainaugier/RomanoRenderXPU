# Src

include(target_options)
include(utils)

# Cuda ptx
add_library(${PTXNAME} OBJECT "shaders/shaders.cu")
set_target_properties(${PTXNAME} PROPERTIES CUDA_PTX_COMPILATION ON)

set(CUDA_DEFINES $<$<CXX_COMPILER_ID:MSVC>:-DCUDA_WINDOWS -D_USE_MATH_DEFINES>)

target_compile_options(${PTXNAME} PRIVATE --use_fast_math -optix-ir ${CUDA_DEFINES})

target_include_directories(${PTXNAME}
                           PUBLIC
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

# Render Library
file(GLOB_RECURSE SOURCES *.cpp)

filter_regex(EXCLUDE "(main\.cpp)|(app.*\.cpp)" SOURCES ${SOURCES})

add_library(${LIBNAME} STATIC ${SOURCES})

set_target_properties(${LIBNAME} PROPERTIES CXX_STANDARD 17
                                            PREFIX ""
                                            OUTPUT_NAME ${LIBNAME}
                                            DEBUG_POSTFIX "d"
                                            CUDA_SEPARABLE_COMPILATION ON
                                            POSITION_INDEPENDENT_CODE ON)

set_target_options(${LIBNAME})

# target_compile_definitions(${LIBNAME} PUBLIC ROMANORENDER_BUILD_SHARED)

target_include_directories(${LIBNAME}
                           PUBLIC
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_include_directories(${LIBNAME} PUBLIC ${stdromano_INCLUDE_DIRS})

target_link_libraries(${LIBNAME} PUBLIC stdromano::stdromano)
target_link_libraries(${LIBNAME} PRIVATE Alembic::Alembic)
target_link_libraries(${LIBNAME} PUBLIC GLEW::GLEW)
target_link_libraries(${LIBNAME} PUBLIC CUDA::cudart)
target_link_libraries(${LIBNAME} PRIVATE nlohmann_json::nlohmann_json)

if(UNIX)
    target_compile_options(${LIBNAME} PUBLIC "-lm")
endif()

# Render executable
add_executable(${APPNAME} main.cpp app.cpp app_widgets.cpp)

set_target_properties(${APPNAME} PROPERTIES CXX_STANDARD 17)

set_target_options(${APPNAME})

target_link_libraries(${APPNAME} PRIVATE ${LIBNAME})
target_link_libraries(${APPNAME} PRIVATE glfw)
target_link_libraries(${APPNAME} PRIVATE imgui::imgui)
target_link_libraries(${APPNAME} PRIVATE imnodes)

# Copy clang asan dll to the tests directory when building in debug mode
# along pdb files

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    get_filename_component(CL_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)

    message(STATUS "Found CL directory: ${CL_DIR}")

    set(ASAN_COPY_COMMAND_APP
        ${CMAKE_COMMAND} -E copy ${CL_DIR}/clang_rt.asan_dynamic-x86_64.dll $<TARGET_FILE_DIR:${APPNAME}>
    )

    add_custom_command(
        TARGET ${APPNAME} POST_BUILD
        COMMAND "$<$<CONFIG:Debug,RelWithDebInfo>:${ASAN_COPY_COMMAND_APP}>"
        COMMAND_EXPAND_LISTS
    )

    set(PDB_COPY_COMMAND_APP
        ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE_DIR:${LIBNAME}>/$<TARGET_FILE_BASE_NAME:${LIBNAME}>.pdb $<TARGET_FILE_DIR:${APPNAME}>)

    add_custom_command(
        TARGET ${APPNAME} POST_BUILD
        COMMAND "$<$<CONFIG:Debug,RelWithDebInfo>:${PDB_COPY_COMMAND_APP}>"
        COMMAND_EXPAND_LISTS
    )
endif()

# Copy dlls to the tests runtime directory

if(WIN32)
    add_custom_command(
        TARGET ${APPNAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_RUNTIME_DLLS:${APPNAME}>
            $<TARGET_FILE_DIR:${APPNAME}>
        COMMAND_EXPAND_LISTS
    )

    add_custom_command(
        TARGET ${APPNAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${stdromano_RUNTIME_DEPENDENCIES}
            $<TARGET_FILE_DIR:${APPNAME}>
        COMMAND_EXPAND_LISTS
    )
endif()

add_custom_command(
    TARGET ${APPNAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory        
        ${CMAKE_SOURCE_DIR}/res
        $<TARGET_FILE_DIR:${APPNAME}>/res
)

add_custom_command(
    TARGET ${APPNAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${PTX_OUTPUT_FILE}
        $<TARGET_FILE_DIR:${APPNAME}>/shaders/shaders.ptx
)
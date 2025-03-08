# Src

include(target_options)

# Render Library
file(GLOB_RECURSE SOURCES *.cpp)
list(REMOVE_ITEM SOURCES "main.cpp")

add_library(${LIBNAME} ${SOURCES})

set_target_properties(${LIBNAME} PROPERTIES CXX_STANDARD 17)
set_target_properties(${LIBNAME} PROPERTIES PREFIX "")
set_target_properties(${LIBNAME} PROPERTIES OUTPUT_NAME ${LIBNAME})
set_target_properties(${LIBNAME} PROPERTIES DEBUG_POSTFIX "d")

set_target_options(${LIBNAME})

target_compile_definitions(${LIBNAME} PUBLIC ROMANORENDER_BUILD_SHARED)

target_include_directories(${LIBNAME}
                           PUBLIC
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_include_directories(${LIBNAME} PUBLIC ${stdromano_INCLUDE_DIRS})

target_link_libraries(${LIBNAME} PUBLIC stdromano::stdromano)
target_link_libraries(${LIBNAME} PRIVATE Alembic::Alembic)
target_link_libraries(${LIBNAME} PUBLIC GLEW::GLEW)

# Render executable
add_executable(${APPNAME} main.cpp)

set_target_options(${APPNAME})

target_link_libraries(${APPNAME} PRIVATE ${LIBNAME})
target_link_libraries(${APPNAME} PRIVATE glfw)
target_link_libraries(${APPNAME} PRIVATE imgui::imgui)

# Copy clang asan dll to the tests directory when building in debug mode
# along pdb files

if(CMAKE_C_COMPILER_ID STREQUAL "MSVC")
    get_filename_component(CL_DIR ${CMAKE_C_COMPILER} DIRECTORY)

    set(ASAN_COPY_COMMAND
        ${CMAKE_COMMAND} -E copy_if_different ${CL_DIR}/clang_rt.asan_dynamic-x86_64.dll $<TARGET_FILE_DIR:${APPNAME}>
    )

    add_custom_command(
        TARGET ${APPNAME} POST_BUILD
        COMMAND "$<$<CONFIG:Debug,RelWithDebInfo>:${ASAN_COPY_COMMAND}>"
        COMMAND_EXPAND_LISTS
    )

    set(PDB_COPY_COMMAND
        ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE_DIR:${LIBNAME}>/$<TARGET_FILE_BASE_NAME:${LIBNAME}>.pdb $<TARGET_FILE_DIR:${APPNAME}>)

    add_custom_command(
        TARGET ${APPNAME} POST_BUILD
        COMMAND "$<$<CONFIG:Debug,RelWithDebInfo>:${PDB_COPY_COMMAND}>"
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
endif()
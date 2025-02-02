# Src

include(target_options)

# Render Library
file(GLOB_RECURSE SOURCES *.cpp)
list(REMOVE_ITEM SOURCES "main.cpp")

add_library(${LIBNAME} ${SOURCES})

set_target_properties(${LIBNAME} PROPERTIES CXX_STANDARD 11)
set_target_properties(${LIBNAME} PROPERTIES PREFIX "")
set_target_properties(${LIBNAME} PROPERTIES OUTPUT_NAME ${LIBNAME})
set_target_properties(${LIBNAME} PROPERTIES DEBUG_POSTFIX "d")

set_target_options(${LIBNAME})

target_include_directories(${LIBNAME}
                           PUBLIC
                           $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>
                           $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>)

target_include_directories(${LIBNAME} PUBLIC ${stdromano_INCLUDE_DIRS})

target_link_libraries(${LIBNAME} IMGUILIB)
target_link_libraries(${LIBNAME} PUBLIC stdromano::stdromano)

# Render executable
add_executable(${APPNAME} main.cpp)

set_target_options(${APPNAME})

target_link_libraries(${APPNAME} PUBLIC ${LIBNAME})

include(target_options)

file(GLOB_RECURSE NODE_FILES *.cpp)

set(NODES "")

foreach(node_file ${NODE_FILES})
    get_filename_component(NODENAME ${node_file} NAME_WLE)
    message(STATUS "Compiling romanorender node: ${NODENAME}")

    add_library(${NODENAME} SHARED ${node_file})
    set_target_options(${NODENAME})
    target_link_libraries(${NODENAME} ${LIBNAME})

    list(APPEND NODES ${NODENAME})
endforeach()
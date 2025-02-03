# ImGui

file(GLOB_RECURSE HEADERS *.h)
file(GLOB_RECURSE SOURCES *.cpp)

add_library(IMGUILIB ${HEADERS} ${SOURCES})

target_link_libraries(IMGUILIB PUBLIC GLEW::GLEW)
target_link_libraries(IMGUILIB PUBLIC glfw)
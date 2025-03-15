include(target_options)

file(GLOB_RECURSE TEST_FILES *.cpp)

foreach(test_file ${TEST_FILES})
    get_filename_component(TESTNAME ${test_file} NAME_WLE)
    message(STATUS "Adding romanorender test : ${TESTNAME}")

    add_executable(${TESTNAME} ${test_file})
    set_target_options(${TESTNAME})
    target_link_libraries(${TESTNAME} PRIVATE ${LIBNAME})
    
    add_test(${TESTNAME} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TESTNAME})
endforeach()

# Copy clang asan dll to the tests directory when building in debug mode
# along pdb files

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    get_filename_component(CL_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)

    message(STATUS "Found CL directory: ${CL_DIR}")

    set(ASAN_COPY_COMMAND
        ${CMAKE_COMMAND} -E copy_if_different ${CL_DIR}/clang_rt.asan_dynamic-x86_64.dll $<TARGET_FILE_DIR:${TESTNAME}>
    )

    add_custom_command(
        TARGET ${TESTNAME} POST_BUILD
        COMMAND "$<$<CONFIG:Debug,RelWithDebInfo>:${ASAN_COPY_COMMAND}>"
        COMMAND_EXPAND_LISTS
    )

    set(PDB_COPY_COMMAND
        ${CMAKE_COMMAND} -E copy_if_different $<TARGET_FILE_DIR:${PROJECT_NAME}>/$<TARGET_FILE_BASE_NAME:${PROJECT_NAME}>.pdb $<TARGET_FILE_DIR:${TESTNAME}>)

    add_custom_command(
        TARGET ${TESTNAME} POST_BUILD
        COMMAND "$<$<CONFIG:Debug,RelWithDebInfo>:${PDB_COPY_COMMAND}>"
        COMMAND_EXPAND_LISTS
    )
endif()

if(WIN32)
    add_custom_command(
        TARGET ${TESTNAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            $<TARGET_RUNTIME_DLLS:${TESTNAME}>
            $<TARGET_FILE_DIR:${TESTNAME}>
        COMMAND_EXPAND_LISTS
    )
endif()

add_custom_command(
    TARGET ${TESTNAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${PTX_OUTPUT_FILE}
        $<TARGET_FILE_DIR:${TESTNAME}>/shaders/shaders.ptx
)
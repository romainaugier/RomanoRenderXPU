include(target_options)

file(GLOB_RECURSE TEST_FILES *.cpp)

foreach(test_file ${TEST_FILES})
    get_filename_component(TESTNAME ${test_file} NAME_WLE)
    message(STATUS "Adding romanorender test : ${TESTNAME}")

    add_executable(${TESTNAME} ${test_file})
    set_target_options(${TESTNAME})
    set_target_properties(${TESTNAME} PROPERTIES CXX_STANDARD 17)
    target_link_libraries(${TESTNAME} PRIVATE ${LIBNAME})
    
    add_test(${TESTNAME} ${CMAKE_RUNTIME_OUTPUT_DIRECTORY}/${TESTNAME})
endforeach()

# Copy clang asan dll to the tests directory when building in debug mode
# along pdb files

if(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
    get_filename_component(CL_DIR ${CMAKE_CXX_COMPILER} DIRECTORY)

    message(STATUS "Found CL directory: ${CL_DIR}")

    set(ASAN_COPY_COMMAND
        ${CMAKE_COMMAND} -E copy ${CL_DIR}/clang_rt.asan_dynamic-x86_64.dll $<TARGET_FILE_DIR:${TESTNAME}>
    )

    add_custom_command(
        TARGET ${TESTNAME} POST_BUILD
        COMMAND "$<$<CONFIG:Debug,RelWithDebInfo>:${ASAN_COPY_COMMAND}>"
        COMMAND_EXPAND_LISTS
    )

    set(PDB_COPY_COMMAND
        ${CMAKE_COMMAND} -E copy $<TARGET_FILE_DIR:${LIBNAME}>/$<TARGET_FILE_BASE_NAME:${LIBNAME}>.pdb $<TARGET_FILE_DIR:${TESTNAME}>)

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

    add_custom_command(
        TARGET ${TESTNAME} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy
            ${stdromano_RUNTIME_DEPENDENCIES}
            $<TARGET_FILE_DIR:${TESTNAME}>
        COMMAND_EXPAND_LISTS
    )
endif()

add_custom_command(
    TARGET ${TESTNAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_directory        
        ${CMAKE_SOURCE_DIR}/res
        $<TARGET_FILE_DIR:${TESTNAME}>/res
)

add_custom_command(
    TARGET ${TESTNAME} POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
        ${PTX_OUTPUT_FILE}
        $<TARGET_FILE_DIR:${TESTNAME}>/shaders/shaders.ptx
)
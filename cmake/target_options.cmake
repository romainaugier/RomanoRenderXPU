function(set_target_options target_name)
    if(CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
        set(ROMANO_RENDER_CLANG 1)
        set(CMAKE_CXX_FLAGS "-Wall -pedantic-errors")

        target_compile_options(${target_name} PRIVATE $<$<CONFIG:Debug>:-fsanitize=leak -fsanitize=address>)
        target_compile_options(${target_name} PRIVATE $<$<CONFIG:Release,RelWithDebInfo>:-O3 -mavx2 -mfma)

        target_link_options(${target_name} PRIVATE $<$<CONFIG:Debug>:-fsanitize=address>)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set(ROMANO_RENDER_GCC 1)

        set(COMPILE_OPTIONS -D_FORTIFY_SOURCES=2 -pipe $<$<CONFIG:Debug>:-fsanitize=leak -fsanitize=address> $<$<CONFIG:Release,RelWithDebInfo>:-O3 -ftree-vectorizer-verbose=2> -mveclibabi=svml -mavx2 -mfma)

        string(JOIN ", " CUDA_COMPILE_OPTIONS "${COMPILE_OPTIONS}")
        string(REPLACE ";" "," CUDA_COMPILE_OPTIONS "${CUDA_COMPILE_OPTIONS}")

        target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${COMPILE_OPTIONS}> $<$<COMPILE_LANGUAGE:CUDA>:\"${CUDA_COMPILE_OPTIONS}\">)

        target_link_options(${target_name} PRIVATE $<$<CONFIG:Debug>:-fsanitize=address>)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
        set(ROMANO_RENDER_INTEL 1)
    elseif(CMAKE_CXX_COMPILER_ID STREQUAL "MSVC")
        set(ROMANO_RENDER_MSVC 1)
        include(find_avx)

        # 4710 is "Function not inlined", we don't care it pollutes more than tells useful information about the code
        # 5045 is "Compiler will insert Spectre mitigation for memory load if /Qspectre switch specified", again we don't care
        set(COMPILE_OPTIONS /W1 /wd4710 /wd5045 /utf-8 ${AVX_FLAGS} $<$<CONFIG:Debug>:/fsanitize=address> $<$<CONFIG:Release,RelWithDebInfo>:/O2 /GF /Ot /Oy /GT /GL /Oi /Zi /Gm- /Zc:inline /Qpar>)

        string(JOIN ", " CUDA_COMPILE_OPTIONS "${COMPILE_OPTIONS}")
        string(REPLACE ";" "," CUDA_COMPILE_OPTIONS "${CUDA_COMPILE_OPTIONS}")

        target_compile_options(${target_name} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:${COMPILE_OPTIONS}> $<$<COMPILE_LANGUAGE:CUDA>:\"${CUDA_COMPILE_OPTIONS}\">)

        # 4300 is "ignoring '/INCREMENTAL' because input module contains ASAN metadata", and we do not care
        set_target_properties(${target_name} PROPERTIES LINK_FLAGS "/ignore:4300")
    endif()

    # Provides the macro definition DEBUG_BUILD
    target_compile_definitions(${target_name} PRIVATE $<$<CONFIG:Debug>:DEBUG_BUILD>)
endfunction()
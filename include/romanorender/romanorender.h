#pragma once

#if !defined(__ROMANORENDER)
#define __ROMANORENDER

#if defined(_MSC_VER)
#define ROMANORENDER_MSVC
#pragma warning(disable:4711) /* function selected for automatic inline expansion */
#elif defined(__GNUC__)
#define ROMANORENDER_GCC
#elif defined(__clang__)
#define ROMANORENDER_CLANG
#endif /* defined(_MSC_VER) */

#if !defined(ROMANORENDER_VERSION_STR)
#define ROMANORENDER_VERSION_STR "Debug"
#endif /* !defined(ROMANORENDER_VERSION_STR) */

#include <cstddef>
#include <cstdint>
#include <cstdbool>
#include <cstdio>
#include <cstdlib>

#if INTPTR_MAX == INT64_MAX || defined(__x86_64__)
#define ROMANORENDER_X64
#define ROMANORENDER_SIZEOF_PTR 8
#elif INTPTR_MAX == INT32_MAX
#define ROMANORENDER_X86
#define ROMANORENDER_SIZEOF_PTR 4
#endif /* INTPTR_MAX == INT64_MAX || defined(__x86_64__) */

#if defined(_WIN32)
#define ROMANORENDER_WIN
#if !defined(WIN32_LEAN_AND_MEAN)
#define WIN32_LEAN_AND_MEAN
#endif /* !defined(WIN32_LEAN_AND_MEAN) */
#if defined(ROMANORENDER_X64)
#define ROMANORENDER_PLATFORM_STR "WIN64"
#else
#define ROMANORENDER_PLATFORM_STR "WIN32"
#endif /* defined(ROMANORENDER_x64) */
#elif defined(__linux__)
#define ROMANORENDER_LINUX
#if defined(ROMANORENDER_X64)
#define ROMANORENDER_PLATFORM_STR "LINUX64"
#else
#define ROMANORENDER_PLATFORM_STR "LINUX32"
#endif /* defined(ROMANORENDER_X64) */
#elif defined(__APPLE__)
#define ROMANORENDER_APPLE
#if defined(ROMANORENDER_X64)
#define ROMANORENDER_PLATFORM_STR "APPLE64"
#else
#define ROMANORENDER_PLATFORM_STR "APPLE32"
#endif /* defined(ROMANORENDER_X64) */
#elif defined(__OpenBSD__)
#define ROMANORENDER_OPENBSD
#if defined(ROMANORENDER_X64)
#define ROMANORENDER_PLATFORM_STR "OPENBSD64"
#else
#define ROMANORENDER_PLATFORM_STR "OPENBSD32"
#endif /* defined(ROMANORENDER_X64) */
#elif defined(__NetBSD__) 
#define ROMANORENDER_NETBSD
#elif defined(__FreeBSD__) 
#define ROMANORENDER_FREEBSD
#elif defined(__DragonFly__)
#define ROMANORENDER_DRAGONFLY
#endif /* defined(_WIN32) */

#define ROMANORENDER_BYTE_ORDER_UNDEFINED 0
#define ROMANORENDER_BYTE_ORDER_LITTLE_ENDIAN __ORDER_LITTLE_ENDIAN__
#define ROMANORENDER_BYTE_ORDER_BIG_ENDIAN __ORDER_BIG_ENDIAN__

#define ROMANORENDER_BYTE_ORDER __BYTE_ORDER__

#define ROMANORENDER_RESTRICT restrict

#if defined(ROMANORENDER_WIN)
#if defined(ROMANORENDER_MSVC)
#define ROMANORENDER_EXPORT __declspec(dllexport)
#define ROMANORENDER_IMPORT __declspec(dllimport)
#elif defined(ROMANORENDER_GCC) || defined(ROMANORENDER_CLANG)
#define ROMANORENDER_EXPORT __attribute__((dllexport))
#define ROMANORENDER_IMPORT __attribute__((dllimport))
#endif /* defined(ROMANORENDER_MSVC) */
#elif defined(ROMANORENDER_LINUX)
#define ROMANORENDER_EXPORT __attribute__((visibility("default")))
#define ROMANORENDER_IMPORT
#endif /* defined(ROMANORENDER_WIN) */

#if defined(ROMANORENDER_MSVC)
#define ROMANORENDER_FORCE_INLINE __forceinline
#define ROMANORENDER_LIB_ENTRY
#define ROMANORENDER_LIB_EXIT
#elif defined(ROMANORENDER_GCC)
#define ROMANORENDER_FORCE_INLINE inline __attribute__((always_inline)) 
#define ROMANORENDER_LIB_ENTRY __attribute__((constructor))
#define ROMANORENDER_LIB_EXIT __attribute__((destructor))
#elif defined(ROMANORENDER_CLANG)
#define ROMANORENDER_FORCE_INLINE __attribute__((always_inline))
#define ROMANORENDER_LIB_ENTRY __attribute__((constructor))
#define ROMANORENDER_LIB_EXIT __attribute__((destructor))
#endif /* defined(ROMANORENDER_MSVC) */

#if defined(ROMANORENDER_GCC) || defined(ROMANORENDER_CLANG)
#define	ROMANORENDER_LIKELY(x)	__builtin_expect((x) != 0, 1)
#define	ROMANORENDER_UNLIKELY(x)	__builtin_expect((x) != 0, 0)
#else
#define	ROMANORENDER_LIKELY(x) (x)
#define	ROMANORENDER_UNLIKELY(x) (x)
#endif /* defined(ROMANORENDER_GCC) || defined(ROMANORENDER_CLANG) */

#if defined(ROMANORENDER_BUILD_SHARED)
#define ROMANORENDER_API ROMANORENDER_EXPORT
#else
#define ROMANORENDER_API ROMANORENDER_IMPORT
#endif /* defined(ROMANORENDER_BUILD_SHARED) */

#define ROMANORENDER_NAMESPACE_BEGIN namespace romanorender {
#define ROMANORENDER_NAMESPACE_END }

#if defined(ROMANORENDER_WIN)
#define ROMANORENDER_FUNCTION __FUNCTION__
#elif defined(ROMANORENDER_GCC) || defined(ROMANORENDER_CLANG)
#define ROMANORENDER_FUNCTION __PRETTY_FUNCTION__
#endif /* ROMANORENDER_WIN */

#define ROMANORENDER_STATIC_FUNCTION static

#define CONCAT_(prefix, suffix)     prefix##suffix
#define CONCAT(prefix, suffix)      CONCAT_(prefix, suffix)

#define ROMANORENDER_ASSERT(expr, message) if(!(expr)) { std::fprintf(stderr, "Assertion failed in file %s at line %d: %s", __FILE__, __LINE__, message); std::abort(); }

#define ROMANORENDER_STATIC_ASSERT(expr)                \
    struct CONCAT(__outscope_assert_, __COUNTER__)      \
    {                                                   \
        char                                            \
        outscope_assert                                 \
        [2*(expr)-1];                                   \
                                                        \
    } CONCAT(__outscope_assert_, __COUNTER__)

#define ROMANORENDER_NOT_IMPLEMENTED std::fprintf(stderr, "Called function " ROMANORENDER_FUNCTION " that is not implemented (%s:%d)", __FILE__, __LINE__); std::exit(1)

#if defined(ROMANORENDER_MSVC)
#define ROMANORENDER_PACKED_STRUCT(__struct__) __pragma(pack(push, 1)) __struct__ __pragma(pack(pop))
#elif defined(ROMANORENDER_GCC) || defined(ROMANORENDER_CLANG)
#define ROMANORENDER_PACKED_STRUCT(__struct__) __struct__ __attribute__((__packed__))
#else
#define ROMANORENDER_PACKED_STRUCT(__struct__) __struct__
#endif /* defined(ROMANORENDER_MSVC) */

#if defined(ROMANORENDER_MSVC)
#define dump_struct(s) 
#elif defined(ROMANORENDER_CLANG)
#define dump_struct(s) __builtin_dump_struct(s, printf)
#elif defined(ROMANORENDER_GCC)
#define dump_struct(s) 
#endif /* defined(ROMANORENDER_MSVC) */

#if defined(DEBUG_BUILD)
#define ROMANORENDER_DEBUG 1
#else
#define ROMANORENDER_DEBUG 0
#endif /* defined(DEBUG_BUILD) */

#endif /* !defined(__ROMANORENDER) */
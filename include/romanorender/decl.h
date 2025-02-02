#pragma once

#ifdef _MSC_VER 

#define FORCEINLINE __forceinline

#else __GNUC__

#define FORCEINLINE __attribute__((always_inline))

#endif
// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_CORE_TYPES_H_
#define AER_CORE_TYPES_H_

#include <stdint.h>
#include "aer/core/defs.h"

namespace aer {

/// Signed integers
typedef int8_t          I8;
typedef int16_t         I16;
typedef int32_t         I32;
typedef int64_t         I64;

/// Unsigned integers
typedef uint8_t         U8;
typedef uint16_t        U16;
typedef uint32_t        U32;
typedef uint64_t        U64;

/// Fast integers (at least 32-bit, potentially more)
typedef uint_fast32_t   U32F;
typedef int_fast32_t    I32F;

/// Pointers
#if AER_64
typedef U64             UPTR;
typedef I64             IPTR;
#else
typedef U32             UPTR;
typedef I32             IPTR;
#endif

/// Floating point
typedef float           F32;
typedef double          F64;

/// Unsigned simplification
typedef unsigned char   uchar;
typedef unsigned short  ushort;
typedef unsigned int    uint;
typedef unsigned long   ulong;

}  // namespace aer


#endif  // AER_CORE_TYPES_H_

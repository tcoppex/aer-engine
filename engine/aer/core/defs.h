// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_CORE_DEFS_H_
#define AER_CORE_DEFS_H_

#include <cassert>
#include <cstdio>
#include <cstddef>


// ===========================
// + Defines the building OS +
// ===========================
#if defined(__APPLE__) || defined(MACOSX)
#   define AER_MACOSX
#elif defined(_WIN32)
#   define AER_WINDOWS
#else
#   define AER_LINUX
#endif

// ==========================
// + Size of memory address +
// ==========================
#if defined(_M_X64) || defined(_LP64)
#   define AER_64               1
#else
#   define AER_64               0
#endif

// ================================================
// + Delete an object and reset its value to null +
// ================================================
#ifndef AER_SAFE_FREE
#   define AER_SAFE_FREE(x)     if (x) {free(x); x = nullptr;}
#endif

#ifndef AER_SAFE_DELETE
#   define AER_SAFE_DELETE(x)   if (x) {delete x; x = nullptr;}
#endif

#ifndef AER_SAFE_DELETEV
#   define AER_SAFE_DELETEV(x)  if (x) {delete [] x; x = nullptr;}
#endif

// ===============================================
// + Transform a set of characters as a C string +
// ===============================================
#ifndef AER_STRINGIFY
#   define AER_STRINGIFY(x)     (#x)
#endif

// ====================
// + Custom assertion +
// ====================
#ifndef AER_ASSERT
#   define AER_ASSERT(x)        assert(x)
#endif

// ===============
// + DEBUG macro +
// ===============
#ifndef NDEBUG
#   define AER_DEBUG   1
#else
#   define AER_DEBUG   0
#endif


// ==================
// + Checking macro +
// ==================
#ifndef AER_CHECK
# if AER_DEBUG
#   define AER_CHECK(x)         if (!(x)) {fprintf(stderr,"%s (%d) : \x1b[1;31m %s failed.\x1b[0m\n", __FILE__, __LINE__, #x);}
# else
#   define AER_CHECK(x)         (x)
# endif
#endif

// =========================
// + DEBUG warning message +
// =========================
#ifndef AER_WARNING
# if AER_DEBUG
#   define AER_WARNING(msg)     {fprintf(stderr, "[WARNING] %s\n", msg);}
# else
#   define AER_WARNING(msg)
# endif
#endif

// ===================
// + DEBUG only code +
// ===================
#ifndef AER_DEBUG_CODE
# if AER_DEBUG
#   define AER_DEBUG_CODE(x)    x
# else
#   define AER_DEBUG_CODE(x)
# endif
#endif


// =============================
// on debug mode
// remove the const attribute for
// some methods to save stats
// ============================
#ifndef AER_DEBUG_CONST
# if AER_DEBUG
#   define AER_DEBUG_CONST
# else
#   define AER_DEBUG_CONST      const
# endif
#endif


#ifndef AER_MEGABYTE
#   define AER_MEGABYTE         (1u<<20u)
#endif

#ifndef AER_ARRAYSIZE
#   define AER_ARRAYSIZE(v)     static_cast<unsigned long>(sizeof(v)/sizeof(v[0]))
#endif

// ================================================================
// + Disallow the copy constructor and operator= functions        +
// + This should be used in the private: declarations for a class +
// ================================================================
#define DISALLOW_COPY_AND_ASSIGN(TypeName)    \
  TypeName(const TypeName&) = delete;         \
  void operator=(const TypeName&) = delete


#endif  // AER_CORE_DEFS_H_

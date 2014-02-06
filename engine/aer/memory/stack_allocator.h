// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AER_MEMORY_STACK_ALLOCATOR_H_
#define AER_MEMORY_STACK_ALLOCATOR_H_

#include <cstdlib>
#include "aer/common.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// A stack memory allocator data structure.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class StackAllocator
{
 public:
  typedef U32 Marker;

  explicit 
  StackAllocator() : 
    data_(nullptr),
    bytesize_(0u),
    top_(0u)
  {}
  
  ~StackAllocator() {
    release();
  }
  
  bool init(const U32 bytesize);
  void release();
  
  /// Allocates a new block of bytesize from the top;
  /// returns nullptr if there is no enough memory
  void* allocate(const U32 bytesize);

  /// Aligned allocation. 'alignment' must be a power of 2
  /// MUST be freed with freeAlignedToMarker()
  void* allocate_aligned(const U32 bytesize, const U32 alignment);


  template<typename T>
  T* allocate(const U32 bytesize) {
    return reinterpret_cast<T*>(allocate(bytesize));
  }

  template<typename T>
  T* allocate_aligned(const U32 bytesize, const U32 alignment) {
    return reinterpret_cast<T*>(allocate_aligned(bytesize, alignment));
  }

  /// Set the stack top to a previous position
  void free_to(const Marker marker);

  void free_aligned_to(const Marker marker);

  /// Return the stack size in bytes
  U32 size() const { 
    return bytesize_;
  }

  /// Return a marker to the current stack top
  Marker marker() const { 
    return top_;
  }

  /// Clear the stack, ie. set the marker to zero
  void clear() { 
    top_ = 0u;
  }


 private:
  void*  data_;
  U32    bytesize_;
  Marker top_;


  DISALLOW_COPY_AND_ASSIGN(StackAllocator);
};

}  // namespace aer

#endif  // AER_MEMORY_STACK_ALLOCATOR_H_

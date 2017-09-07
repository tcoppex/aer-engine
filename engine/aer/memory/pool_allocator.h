// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AER_MEMORY_POOL_ALLOCATOR_H_
#define AER_MEMORY_POOL_ALLOCATOR_H_

/*
For she is the key to open my mind to see / 
The energy that radiates from the gates of heavenly bliss.
*/

#include "aer/common.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// A linkedlist memory allocator data structure.
/// A pool allocator is a virtual space where many chunks
/// of memory are allocated once as a large block.
/// Useful for operation necessiting many alloc / free per
/// frames.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class PoolAllocator {
 public:
  PoolAllocator();
  ~PoolAllocator();

  void initialize(const U32 chunkcount, const U32 chunksize);
  void release();

  void* allocate();
  void  deallocate(void* ptr);

  U32 capacity() const { return capacity_; }
  U32 size()     const { return size_; }


 private:
  void setupFreeChunkList();

  void *data_;                  // allocated memory
  void *freelist_ptr_;          // head to the list of free chunks

  U32 alignement_padding_;      // padding to align original data (in bytes)
  U32 capacity_;                // total capacity of the pool
  U32 size_;                    // actual number of allocate elements
  U32 chunksize_;

  DISALLOW_COPY_AND_ASSIGN(PoolAllocator);
};

} // aer

#endif  // AER_MEMORY_POOL_ALLOCATOR_H_

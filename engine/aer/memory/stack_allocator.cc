// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#include "aer/memory/stack_allocator.h"


namespace aer {

bool StackAllocator::init(const U32 bytesize) {
  AER_ASSERT(nullptr == data_);//

  bytesize_ = bytesize;
  data_ = calloc(bytesize_, 1u);
  return nullptr != data_;
}

void StackAllocator::release() {
  AER_SAFE_FREE(data_);
}

void* StackAllocator::allocate(const U32 bytesize) {
  AER_ASSERT(nullptr != data_);

  if (top_ + bytesize > bytesize_) {
    return nullptr;
  }

  UPTR address = reinterpret_cast<UPTR>(data_) + top_;
  void *data_ptr = reinterpret_cast<void*>(address);
  top_ += bytesize;

  return data_ptr;
}

void* StackAllocator::allocate_aligned(const U32 bytesize, const U32 alignment) {
  AER_ASSERT(nullptr != data_);

  // Handles only alignment of power of two greater than 2
  AER_ASSERT(alignment > 2u);
  //AER_ASSERT(IsPowerOfTwo(alignment));

  // Total amount of memory to allocate
  U32 expanded_size = bytesize + alignment;

  // Allocate an unaligned block and convert the address
  UPTR raw_address = reinterpret_cast<UPTR>(allocate(expanded_size));

  if (raw_address == 0u) {
    return nullptr;
  }

  // Calculates the adjustment by masking off the lower bits of the address
  UPTR mask         = (alignment - 1u);
  UPTR misalignment = (raw_address & mask);
  UPTR adjustment   = alignment - misalignment;

  // Calculate the adjusted address
  UPTR aligned_address = raw_address + adjustment;
  top_ += adjustment;

  // Store the adjustment in the four bytes preceding the adjusted address
  // returned
  UPTR *adjustment_ptr = reinterpret_cast<UPTR*>(aligned_address - 4u);
  *adjustment_ptr = adjustment;

  return reinterpret_cast<void*>(aligned_address);
}

void StackAllocator::free_to(const Marker marker) {
  AER_ASSERT(marker < top_);
  top_ = marker;
}

void StackAllocator::free_aligned_to(const Marker marker) {
  AER_ASSERT((marker >= 4u) && (marker < top_));

  UPTR adjustment = (reinterpret_cast<UPTR*>(data_))[marker-4u];
  Marker raw_marker = marker - adjustment;
  free_to(raw_marker);
}

}  // namespace aer

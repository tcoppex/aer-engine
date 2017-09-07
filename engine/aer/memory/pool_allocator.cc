// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#include "aer/memory/pool_allocator.h"


namespace aer {

PoolAllocator::PoolAllocator() : 
    data_(nullptr),
    freelist_ptr_(nullptr),
    alignement_padding_(0u),
    capacity_(0u),
    size_(0u),
    chunksize_(0u) {
}

PoolAllocator::~PoolAllocator() {
  if (data_) {
    release();
  }
}

void PoolAllocator::initialize(const U32 chunkcount, const U32 chunksize) {
  AER_ASSERT(data_ == nullptr);

  // Needed to store free chunk list pointer
  AER_ASSERT(chunksize >= sizeof(UPTR));

  // For base alignment
  // AER_ASSERT(isPowerOfTwo(chunksize));

  capacity_  = chunkcount;
  chunksize_ = chunksize;

  // Add +1 to capacity for base memory alignment
  void *ptr = calloc(capacity_+1u, chunksize_);

  // Align base memory on chunk size
  UPTR raw_address    = UPTR(ptr);
  UPTR mask           = chunksize_ - 1u;
  UPTR misalignment   = raw_address & mask;
  alignement_padding_ = chunksize - misalignment;

  data_ = reinterpret_cast<void*>(raw_address + alignement_padding_);

  setupFreeChunkList();
}

void PoolAllocator::setupFreeChunkList() {
  // Set the list of free chunks
  UPTR *nextaddress_ptr = reinterpret_cast<UPTR*>(data_);

  for (U32 i = 0u; i < capacity_-1u; ++i) {
    *nextaddress_ptr = UPTR(nextaddress_ptr) + chunksize_;
    nextaddress_ptr  = reinterpret_cast<UPTR*>(*nextaddress_ptr);
  }
  *nextaddress_ptr = 0;

  // Head of the list
  freelist_ptr_ = data_;
}

void PoolAllocator::release() {
  AER_ASSERT(nullptr != data_);

  freelist_ptr_ = nullptr;
  void *ptr = reinterpret_cast<void*>(UPTR(data_) - alignement_padding_);
  free(ptr);
}

void* PoolAllocator::allocate() {
  AER_ASSERT(0u != capacity_);

  if (size_ >= capacity_) {
    return nullptr;
  }
  ++size_;

  // Get the first free cell
  UPTR *address = reinterpret_cast<UPTR*>(freelist_ptr_);
  UPTR next     = *address;

  // Update the free chunk list (Head = Head->next)
  freelist_ptr_ = reinterpret_cast<void*>(next);

  return address;
}

void PoolAllocator::deallocate(void *ptr) {
  AER_ASSERT(size_ > 0);

  --size_;

  UPTR *dst = reinterpret_cast<UPTR*>(ptr);
  *dst = (nullptr != freelist_ptr_) ? *(reinterpret_cast<UPTR*>(freelist_ptr_))
                                 : 0u;

  // ptr becomes the new list head
  freelist_ptr_ = ptr;
}

}  // namespace aer

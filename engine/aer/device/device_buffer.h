// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_DEVICE_DEVICE_BUFFER_H_
#define AER_DEVICE_DEVICE_BUFFER_H_

#include "aer/common.h"
#include "aer/device/device_resource.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
///
/// Wrapper around OpenGL buffer object.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class DeviceBuffer : public DeviceResource {
 public:
  static
  void Unbind(GLenum target) {
    glBindBuffer(target, 0u);
  }


  DeviceBuffer() :
    target_(GL_ARRAY_BUFFER),
    bJustAllocated_(false),
    bMapped_(false),
    bBinded_(false)
  {}

  void generate() {
    AER_ASSERT(!is_generated());
    glGenBuffers(1, &id_);
  }

  void release() {
    if (is_generated()) {
      glDeleteBuffers(1, &id_);
      id_ = 0u;
    }
  }

  void bind(GLenum target) {
    AER_ASSERT(is_generated());
    target_ = target;
    glBindBuffer(target_, id_);
    bBinded_ = true;
  }

  void unbind() {
    Unbind(target_);
    bBinded_ = false;
  }

  void allocate(UPTR bytesize, GLenum usage) {
    AER_ASSERT(!bMapped_);

    // Creates mutable storage
    bytesize_ = bytesize;
    usage_    = usage;
    glBufferData(target_, bytesize_, NULL, usage_);
    bJustAllocated_ = true;
  }

  IPTR upload(IPTR offset, UPTR bytesize, const void *data) {
    AER_ASSERT(!bMapped_);
    AER_ASSERT(bBinded_);

    /// Tell the device to discard previous buffer data
    if (!bJustAllocated_ && (offset == 0u) && (bytesize == bytesize_)) {
      allocate(bytesize_, usage_);
    }
    glBufferSubData(target_, offset, bytesize, data);
    return offset + bytesize;
  }

  //void clear(GLenum internalformat, const void* data);

  void copy(const DeviceBuffer &src,
            const IPTR src_offset,
            const IPTR dst_offset,
            const UPTR bytesize)
  {
    AER_ASSERT(!bMapped_);
    AER_ASSERT(bytesize < bytesize_);
    glCopyBufferSubData(src.id(), id_, src_offset, dst_offset, bytesize);
  }

  template<typename T>
  void map(T** pptr, GLenum access) {
    AER_ASSERT(!bMapped_);
    bMapped_ = true;
    *pptr = reinterpret_cast<T*>(glMapBuffer(target_, access));
  }

  template<typename T>
  void map_range(void **pptr, IPTR offset, UPTR length, GLenum access) {
    AER_ASSERT(!bMapped_);
    bMapped_ = true;
    *pptr = reinterpret_cast<T*>(glMapBufferRange(target_, offset, length, access));
  }

  template<typename T>
  void unmap(T** pptr) {
    *pptr = NULL;
    bMapped_ = false;
    glUnmapBuffer(target_);
  }

  GLenum target() const {
    return target_;
  }

  UPTR bytesize() const {
    return bytesize_;
  }

  bool is_mapped() const {
    return bMapped_;
  }


 private:
  GLenum target_;
  GLenum usage_;
  UPTR bytesize_;
  bool bJustAllocated_;
  bool bMapped_;
  bool bBinded_;
};
  
}  // namespace aer

#endif  // AER_DEVICE_DEVICE_BUFFER_H_

// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_DEVICE_DEVICE_BUFFER_H_
#define AER_DEVICE_DEVICE_BUFFER_H_

#include "aer/common.h"
#include "aer/device/device_resource.h"

// =============================================================================
namespace aer {
// =============================================================================

/**
 * @class DeviceBuffer
 * @brief Wrapper around OpenGL buffer object
*/
class DeviceBuffer : public DeviceResource {
public:
  // ---------------------------------------------------------------------------
  /// @name Static methods
  // ---------------------------------------------------------------------------

  static
  void Unbind(GLenum target) {
    glBindBuffer(target, 0u);
  }

  // ---------------------------------------------------------------------------
  /// @name onstructor
  // ---------------------------------------------------------------------------

  DeviceBuffer() :
    target_(GL_ARRAY_BUFFER),
    bJustAllocated_(false),
    bBinded_(false)
  {}

  // ---------------------------------------------------------------------------
  /// @name DeviceResource methods
  // ---------------------------------------------------------------------------

  void generate() override {
    AER_ASSERT(!is_generated());
    glGenBuffers(1, &id_);
  }

  void release() override {
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

  // ---------------------------------------------------------------------------
  /// @name DeviceBuffer methods
  // ---------------------------------------------------------------------------

  ///
  void allocate(UPTR bytesize, GLenum usage) {
    AER_ASSERT(!is_mapped());

    // Creates mutable storage
    bytesize_ = bytesize;
    glBufferData(target_, bytesize_, NULL, usage);
    bJustAllocated_ = true;
  }
  
  ///
  IPTR upload(IPTR offset, UPTR bytesize, const void *data) {
    AER_ASSERT(!is_mapped());
    AER_ASSERT(bBinded_);

    /// Tell the device to discard previous buffer data
    if (!bJustAllocated_ && (offset == 0u) && (bytesize == bytesize_)) {
      allocate(bytesize_, usage());
    }
    glBufferSubData(target_, offset, bytesize, data);
    return offset + bytesize;
  }

  //void clear(GLenum internalformat, const void* data);

  ///
  void copy(const DeviceBuffer &src,
            const IPTR src_offset,
            const IPTR dst_offset,
            const UPTR bytesize)
  {
    AER_ASSERT(!is_mapped());
    AER_ASSERT(bytesize < bytesize_);
    glCopyBufferSubData(src.id(), id_, src_offset, dst_offset, bytesize);
  }

  ///
  template<typename T>
  void map(T** pptr, GLenum access) {
    AER_ASSERT(!is_mapped());
    *pptr = reinterpret_cast<T*>(glMapBuffer(target_, access));
  }

  ///
  template<typename T>
  void map_range(void **pptr, IPTR offset, UPTR length, GLenum access) {
    AER_ASSERT(!is_mapped());
    *pptr = reinterpret_cast<T*>(glMapBufferRange(target_, offset, length, access));
  }

  ///
  template<typename T>
  void unmap(T** pptr) {
    *pptr = NULL;
    glUnmapBuffer(target_);
  }

  // ---------------------------------------------------------------------------
  /// @name Getters
  // ---------------------------------------------------------------------------

  ///
  GLenum target() const {
    return target_;
  }
  
  ///
  UPTR bytesize() const {
    return bytesize_;
  }
  
  ///
  GLenum usage() const {
    AER_ASSERT(bBinded_);
    GLint data;
    glGetBufferParameteriv(target_, GL_BUFFER_USAGE, &data);
    return GLenum(data);
  }

  ///
  bool is_mapped() const {
    AER_ASSERT(bBinded_);
    GLint data;
    glGetBufferParameteriv(target_, GL_BUFFER_MAPPED, &data);
    return data == GL_TRUE;
  }


private:
  // ---------------------------------------------------------------------------
  /// @name Attributes
  // ---------------------------------------------------------------------------

  GLenum  target_;
  UPTR    bytesize_; //
  bool    bJustAllocated_;
  bool    bBinded_;
};

// =============================================================================
}  // namespace aer
// =============================================================================

#endif  // AER_DEVICE_DEVICE_BUFFER_H_

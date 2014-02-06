// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_DEVICE_TEXTURE_BUFFER_H_
#define AER_DEVICE_TEXTURE_BUFFER_H_

#include "aer/common.h"
#include "aer/device/texture.h"
#include "aer/device/device_buffer.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
class TextureBuffer : public Texture {
 public:
  TextureBuffer() :
    Texture(GL_TEXTURE_BUFFER)
  {}

  void set_buffer(GLenum internalformat, const DeviceBuffer &buffer) {
    glTexBuffer(target_, internalformat, buffer.id());
  }

  void set_buffer_range(GLenum internalformat,
                        const DeviceBuffer &buffer,
                        IPTR offset,
                        UPTR bytesize) {
    glTexBufferRange(target_, internalformat, buffer.id(), offset, bytesize);
  }
};


/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
/// Simple buffer + texture binding
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
struct TBO_t {
  aer::DeviceBuffer  buffer;
  aer::TextureBuffer texture;

  ~TBO_t() {
    buffer.release();
    texture.release();
  }
};

}  // namespace aer

#endif  // AER_DEVICE_TEXTURE_BUFFER_H_

// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_DEVICE_TEXTURE_H_
#define AER_DEVICE_TEXTURE_H_

#include "aer/common.h"
#include "aer/device/device_resource.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// Wrapper around OpenGL texture object.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class Texture : public DeviceResource {
 public:
  static
  void Activate(U32 unit) {
    glActiveTexture(GL_TEXTURE0 + unit);
  }

  static
  void Unbind(GLenum target, U32 unit=0u) {
    Activate(unit);
    glBindTexture(target, 0u);
  }

  static
  void UnbindAll(GLenum target, I32 count) {
    for (I32 i=count-1; i>=0; --i) {
      Unbind(target, i);
    }
  }

  static
  void GetTextureFormatInfo(GLenum internalformat, GLenum &format, GLenum &type) {
    format = GL_RED;
    type   = GL_UNSIGNED_BYTE;

    if (internalformat == GL_DEPTH24_STENCIL8)
    {
      format = GL_DEPTH_STENCIL;
      type   = GL_UNSIGNED_INT_24_8;
    }
    else if (internalformat == GL_DEPTH_COMPONENT   ||
             internalformat == GL_DEPTH_COMPONENT16 ||
             internalformat == GL_DEPTH_COMPONENT24 ||
             internalformat == GL_DEPTH_COMPONENT32F)
    {
      format = GL_DEPTH_COMPONENT;
      type   = GL_UNSIGNED_INT;
    }
  }


  Texture(GLenum target) :
    target_(target),
    texture_unit_(0u)
  {}

  void generate() {
    AER_ASSERT(!is_generated());
    glGenTextures(1, &id_);
  }

  void release() {
    if (is_generated()) {
      glDeleteTextures(1, &id_);
      id_ = 0u;
    }
  }

  virtual void bind(U32 unit) {
    texture_unit_ = unit;
    Activate(unit);
    glBindTexture(target_, id_);
  }

  virtual void bind() {
    bind(texture_unit_);
  }

  virtual void unbind() {
    Unbind(target_, texture_unit_);
  }


  // Return the texture target
  virtual GLenum target() const {
    return target_;
  }

  /// Set the default texture unit activate with bind and unbind
  void set_texture_unit(U32 unit) {
    texture_unit_ = unit;
  }

 protected:
  GLenum  target_;
  U32     texture_unit_;
};
  
}  // namespace aer

#endif  // AER_DEVICE_TEXTURE_H_

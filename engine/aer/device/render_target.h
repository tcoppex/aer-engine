// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_DEVICE_RENDER_TARGET_H_
#define AER_DEVICE_RENDER_TARGET_H_

#include "aer/common.h"
#include "aer/device/device_resource.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// Interface to Framebuffer object .
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class RenderTarget : public DeviceResource {
 public:
  static
  bool CheckStatus() {
    return GL_FRAMEBUFFER_COMPLETE == glCheckFramebufferStatus(GL_FRAMEBUFFER);
  }

  static
  void Unbind() {
    glBindFramebuffer(GL_FRAMEBUFFER, 0u);
  }


  RenderTarget() :
    DeviceResource()
  {}

        
  void generate() {
    AER_ASSERT(!is_generated());
    glGenFramebuffers(1u, &id_);
  }

  void release() {
    if (is_generated()) {
      glDeleteFramebuffers(1u, &id_);
      id_ = 0u;
    }
  }

  void bind(GLenum target) {
    AER_ASSERT(is_generated());
    target_ = target;
    glBindFramebuffer(target_, id_);
  }

  void unbind(){
    Unbind();
  }

  bool drawable() const {
    return (target_ == GL_DRAW_FRAMEBUFFER) ||
           (target_ == GL_FRAMEBUFFER);
  }

  bool readable() const {
    return (target_ == GL_READ_FRAMEBUFFER) ||
           (target_ == GL_FRAMEBUFFER);
  }

 protected:
  GLenum target_;
};
  
}  // namespace aer

#endif  // AER_DEVICE_RENDER_TARGET_H_

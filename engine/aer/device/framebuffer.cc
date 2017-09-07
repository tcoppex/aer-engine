// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#include "aer/device/framebuffer.h"
#include "aer/device/texture.h"
#include "aer/device/texture_2d.h"
#include "aer/utils/logger.h"


// =============================================================================
namespace aer {
// =============================================================================

void Framebuffer::bind(GLenum target) {
  RenderTarget::bind(target);

  if (drawable()) {
    DrawBuffers(kAttachmentMax, attachments_);
  }
}

// -----------------------------------------------------------------------------

void Framebuffer::attach_color(const Texture *color_tex_ptr,
                               GLenum attachment) {

  AER_ASSERT(nullptr != color_tex_ptr);

#define ATTACHMENT_ID(x)   ((x) - GL_COLOR_ATTACHMENT0)
  const U32 attach_id = ATTACHMENT_ID(attachment);
#undef ATTACHMENT_ID

  if (attach_id >= kAttachmentMax) {
    Logger::Get().error("invalid framebuffer color attachment id");
  }

  if (GL_NONE == attachments_[attach_id]) {
    num_colors_ += 1u;
  } else {
    Logger::Get().warning("a framebuffer color attachment was overriden");
  }

  colors_ptr_[attach_id]  = color_tex_ptr;
  attachments_[attach_id] = attachment;

  glFramebufferTexture(GL_FRAMEBUFFER,
                       attachment,
                       color_tex_ptr->id(),
                       0);
}

// -----------------------------------------------------------------------------

void Framebuffer::attach_special(const Texture2D *special_tex_ptr,
                                 GLenum attachment) {
  AER_ASSERT(nullptr != special_tex_ptr);

  GLenum internalformat = special_tex_ptr->storage_info().internalformat;
  bool validity = CheckSpecialFormat(internalformat, attachment);

  if (!validity) {
    Logger::Get().error("invalid texture format for this attachment");
  }

  special_ptr_ = special_tex_ptr;
  glFramebufferTexture(GL_FRAMEBUFFER,
                       attachment,
                       special_tex_ptr->id(),
                       0);
}

// -----------------------------------------------------------------------------

void Framebuffer::blit(Framebuffer *src_fbo, 
                       const Vector4 &src_coords,
                       const Vector4 &dst_coords,
                       GLbitfield mask,
                       GLenum filter)
{
  if (src_fbo) {
    src_fbo->bind(GL_READ_FRAMEBUFFER);
  }

  bind(GL_DRAW_FRAMEBUFFER);
  glBlitFramebuffer(src_coords.x, src_coords.y, src_coords.z, src_coords.w,
                    dst_coords.x, dst_coords.y, dst_coords.z, dst_coords.w,
                    mask, filter);
  unbind();

  if (src_fbo) {
    src_fbo->unbind();
  }
}

// -----------------------------------------------------------------------------

bool Framebuffer::CheckSpecialFormat(GLenum internalFormat,
                                     GLenum attachment) {
  if (internalFormat == GL_DEPTH_COMPONENT   || 
      internalFormat == GL_DEPTH_COMPONENT16 || 
      internalFormat == GL_DEPTH_COMPONENT24 || 
      internalFormat == GL_DEPTH_COMPONENT32F) {
    return (attachment == GL_DEPTH_ATTACHMENT);
  }

  if (internalFormat == GL_DEPTH24_STENCIL8) {
    return (attachment == GL_DEPTH_STENCIL_ATTACHMENT);
  }

  return false;
}

// =============================================================================
}  // namespace aer
// =============================================================================

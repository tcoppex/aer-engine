// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_DEVICE_FRAMEBUFFER_H_
#define AER_DEVICE_FRAMEBUFFER_H_

#include <vector>

#include "aer/common.h"
#include "aer/device/render_target.h"


namespace aer {

class Texture;
class Texture2D;

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
///
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class Framebuffer : public RenderTarget {
 public:
  static void DrawBuffers(U32 count, GLenum attachments[]) {
    glDrawBuffers(count, attachments);
  }


  Framebuffer();

  /// Overrided, call DrawBuffers on default setup
  void bind(GLenum target=GL_FRAMEBUFFER);

  /// Attach a Color texture as output
  void attach_color(const Texture *color_tex_ptr, 
                    GLenum attachment);

  /// Attach a Depth or Stencil texture as output
  void attach_special(const Texture2D *special_tex_ptr,
                      GLenum attachment);


  /// Passes
  //U32 add_pass(ColorAttachmentSet_t attachments);
  //void set_active_pass(U32 pass_id);

  /// Blit a subregion from another buffer.
  /// If src_fbo is NULL, sample from the main framebuffer.
  void blit(Framebuffer *src_fbo, 
            const Vector4 &src_coords,
            const Vector4 &dst_coords,
            GLbitfield mask,
            GLenum filter);

 private:
  static const U32 kAttachmentMax = 8u;
  static bool CheckSpecialFormat(GLenum internalFormat, GLenum attachment);

  typedef GLenum ColorAttachmentSet_t[kAttachmentMax];

  /// Textures output
  U32 num_colors_;
  const Texture* colors_ptr_[kAttachmentMax];
  const Texture2D* special_ptr_;

  /// Attachments target
  ColorAttachmentSet_t attachments_;

  /// Passes
  //std::vector<ColorAttachmentSet_t> passes_;
  //aer::U32 active_pass_id_;

  bool bInitialized_;
};
  
}  // namespace aer

#endif  // AER_DEVICE_FRAMEBUFFER_H_

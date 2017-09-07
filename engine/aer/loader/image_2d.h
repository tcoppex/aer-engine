// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_LOADER_IMAGE2D_H_
#define AER_LOADER_IMAGE2D_H_

#include "FreeImage.h"

#include "aer/common.h"
#include "aer/core/opengl.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
///  Simple FreeImage wrapper to load 2D image stored as 
///  unsigned char used for creating OpenGL textures
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
class Image2D {
 public:
  Image2D() : 
      target_(GL_INVALID_ENUM),
      internalformat_(0),
      width_(0), 
      height_(0),
      format_(GL_INVALID_ENUM),
      type_(GL_INVALID_ENUM),
      image_(NULL)
  {}

  virtual ~Image2D() { 
    clean();
  }

  void clean() {
    if (image_ != NULL) {
      FreeImage_Unload(image_);
      image_ = NULL;
    }
  }

  bool load(const char *filename);


  /// Getters
  GLenum  target()          const { return target_; }
  GLint   internalformat()  const { return internalformat_; }
  U32     width()           const { return width_; }
  U32     height()          const { return height_; }
  GLenum  format()          const { return format_; }
  GLenum  type()            const { return type_; }
  
  void* data() const { 
    return FreeImage_GetBits(image_);
  }


  private:
    Image2D(const Image2D&);
    const Image2D& operator=(const Image2D&) const;

    bool setDefaultAttributes(FIBITMAP *dib);


    U32 bytesPerPixel;

    GLenum  target_;  
    GLint   internalformat_;
    U32     width_;
    U32     height_;
    GLenum  format_;
    GLenum  type_;

    FIBITMAP *image_;
};

}  // namespace aer

#endif  // AER_LOADER_IMAGE2D_H_

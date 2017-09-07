// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#include "aer/loader/image_2d.h"


namespace aer {

bool Image2D::load(const char *filename) {
  FREE_IMAGE_FORMAT format_ = FreeImage_GetFileType(filename,0);
  image_ = FreeImage_Load(format_, filename);
  
  if (image_ == nullptr) {
    return false;
  }
  
  if (setDefaultAttributes(image_) == false) {    
    // Convert to 32 bits
    image_ = FreeImage_ConvertTo32Bits(image_);

    if (setDefaultAttributes(image_) == false) {
      fprintf(stderr, "ImageLoader : \"%s\" can't be loaded [conversion failed].\n", filename);
      return false;
    }
  }
  return true;
}

bool Image2D::setDefaultAttributes(FIBITMAP *dib) { 
  if (FIT_BITMAP != FreeImage_GetImageType(dib)) {
    return false;
  }

  unsigned int bpp = FreeImage_GetBPP(dib);
  switch (bpp) {      
    case 8u:
      internalformat_ = format_ = GL_RED;
    break;
    
    case 16u:
      internalformat_ = format_ = GL_RG;
    break;
    
    case 24u:
      internalformat_ = GL_RGB;
      format_ = GL_BGR;
    break;
    
    case 32u:
      internalformat_ = GL_RGBA;
      format_ = GL_BGRA;
    break;
    
    default:
      return false;
  }

  bytesPerPixel = bpp / 8u;
  width_  = FreeImage_GetWidth(dib);
  height_ = FreeImage_GetHeight(dib);
  target_ = GL_TEXTURE_2D;
  type_   = GL_UNSIGNED_BYTE;

  return true;
}

}  // namespace aer

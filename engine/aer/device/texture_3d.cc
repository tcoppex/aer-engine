// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------


#include "aer/device/texture_3d.h"


namespace aer {

void Texture3D::allocate(GLenum internalformat,
                         U32 width,
                         U32 height,
                         U32 depth,
                         U32 levels) {
  // Get the appropriate format and type from internalformat
  GLenum format, type;
  Texture::GetTextureFormatInfo(internalformat, format, type);

  // Allocate memory
  for (U32 i = 0u; i < levels; ++i) {
    U32 w = glm::max(1u,  width >> i);
    U32 h = glm::max(1u, height >> i);
    U32 z = glm::max(1u,  depth >> i);
    glTexImage3D(GL_TEXTURE_3D, i, internalformat, w, h, z, 0, format, type, NULL);
    //if (w == h == z == 1u) { break; }
  }

  // Update infos
  storage_.levels         = levels;
  storage_.internalformat = internalformat;
  storage_.width          = width;
  storage_.height         = height;
  storage_.depth          = depth;
  storage_.format         = format;
  storage_.type           = type;
  bAllocated_ = true;
}

void Texture3D::resize(aer::U32 width, aer::U32 height, aer::U32 depth) {
  allocate(storage_.internalformat, width, height, depth, storage_.levels);
}

void Texture3D::upload(U32 level,
                       U32 x, U32 y, U32 z,
                       U32 width, U32 height, U32 depth, 
                       GLenum format, GLenum type,
                       const void* data) {
  AER_ASSERT(bAllocated_);
  AER_ASSERT(level < storage_.levels);

  storage_.format = format;
  glTexSubImage3D(GL_TEXTURE_3D,
                  level, x, y, z, width, height, depth, format, type, data);
  storage_.type = type;
}

}  // namespace aer

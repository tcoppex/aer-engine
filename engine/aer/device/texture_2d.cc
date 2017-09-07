// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#include "aer/device/texture_2d.h"
#include "aer/device/sampler.h"



namespace aer {

void Texture2D::UnbindImage(U32 image_unit) {
  glBindImageTexture(image_unit, 0, 0, GL_FALSE, 0, GL_WRITE_ONLY, GL_R32F);
}


void Texture2D::bind(U32 unit) {
  Texture::bind(unit);

  if (sampler_ptr_) {
    AER_CHECK(is_sampler_compatible(*sampler_ptr_));
    (*sampler_ptr_).bind(texture_unit_);
  }
  bBound_ = true;
}

void Texture2D::bind() {
  bind(texture_unit_);
}

void Texture2D::unbind() {
  Texture::unbind();

  if (sampler_ptr_) {
    (*sampler_ptr_).unbind(texture_unit_);
  }
  bBound_ = false;
}


void Texture2D::bind_image(U32 image_unit, GLenum access, I32 level) {
  glBindImageTexture(image_unit, id_, level, GL_FALSE, 0, access, storage_.internalformat);
}

void Texture2D::unbind_image(U32 image_unit) {
  UnbindImage(image_unit);
}

void Texture2D::allocate(GLenum internalformat,
                         U32 width,
                         U32 height,
                         U32 levels,
                         bool immutable)
{
  AER_ASSERT(bBound_);

  // Get the appropriate format and type from internalformat
  GLenum format, type;
  Texture::GetTextureFormatInfo(internalformat, format, type);

  // Texture use sampling parameter from sampler object, so we specify
  // their internal sampling to be nearest (no mipmapping)
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);//
  glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);//

  if (!immutable) {
    // Allocate memory
    for (U32 i = 0u; i < levels; ++i) {
      const U32 w = glm::max(1u,  width >> i);
      const U32 h = glm::max(1u, height >> i);
      glTexImage2D(GL_TEXTURE_2D, i, internalformat, w, h, 0, format, type, nullptr);
      //if (w == h == 1u) { break; }
    }
  } else {
    // TODO : Create a special method call for immutable creation
    /// OpenGL 4.3+
    /// Needed to be used as Image object by compute shaders
    glTexStorage2D(GL_TEXTURE_2D, levels, internalformat, width, height);
  }

  // Update infos
  storage_.levels         = levels;
  storage_.internalformat = internalformat;
  storage_.width          = width;
  storage_.height         = height;
  storage_.format         = format;
  storage_.type           = type;
  bAllocated_ = true;

  CHECKGLERROR();
}

void Texture2D::resize(U32 width, U32 height) {
  AER_ASSERT(bBound_);
  allocate(storage_.internalformat, width, height, storage_.levels);
}

void Texture2D::upload(U32 level,
                       U32 x,
                       U32 y,
                       U32 width,
                       U32 height,
                       GLenum format,
                       GLenum type,
                       const void* data)
{
  AER_ASSERT(bBound_);
  AER_ASSERT(bAllocated_);
  AER_ASSERT(level < storage_.levels);

  storage_.format = format;
  glTexSubImage2D(GL_TEXTURE_2D, 
                  level, x, y, width, height, format, type, data);
  storage_.type = type;
}

void Texture2D::generate_mipmap() {
  AER_ASSERT(bBound_);

  glGenerateMipmap(GL_TEXTURE_2D);
  bUseMipMap_ = true;
}

bool Texture2D::set_sampler_ptr(const Sampler *sampler_ptr) {
  sampler_ptr_ = sampler_ptr;

  if (sampler_ptr_) {
    return is_sampler_compatible(*sampler_ptr);
  }

  return true;
}

bool Texture2D::is_sampler_compatible(const Sampler &sampler) const {
  bool status = true;

  if (sampler.use_mipmap_filter()) {
    status = status && bUseMipMap_;
  }

  return status;
}

  
}  // namespace aer

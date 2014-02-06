// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_DEVICE_TEXTURE_2D_H_
#define AER_DEVICE_TEXTURE_2D_H_

#include "aer/common.h"
#include "aer/device/texture.h"
#include "aer/device/sampler.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// Wrapper around OpenGL 2D texture object.
///
/// Notes :
/// -----------
/// Sampler parameters cannot be specified texture-wised
/// directly by design, use a sampler object instead.
/// -----------
/// Texture specific parameters (ie. Swizzle, mipmap range &
/// stencil texturing) are not handled directly, yet.
/// -----------
/// One problem is that object should be mutable to be used
/// (ie. bind / unbind are not const). It may change in the 
/// future.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class Texture2D : public Texture {
 public:
  /// Store information on texture storage
  struct StorageInfo_t {
    U32     levels;
    GLenum  internalformat;
    U32     width;
    U32     height;
    GLenum  format;
    GLenum  type;
  };

  static
  void UnbindImage(U32 image_unit);

  Texture2D() :
    Texture(GL_TEXTURE_2D),
    sampler_ptr_(NULL),
    bBound_(false),
    bUseMipMap_(false),
    bAllocated_(false)
  {}


  /// Set the default texture unit then bind the texture and its sampler to it
  void bind(U32 unit);

  /// Bind the texture and its sampler to the default texture unit
  void bind();

  /// Unbind the texture & its sampler from its texture unit
  void unbind();


  ///  + + + + Binding aware methods  + + + +
  /// Allocate data for all levels of the 2D texture
  void allocate(GLenum internalformat, U32 width, U32 height, U32 levels = 1u);

  void resize(aer::U32 width, aer::U32 height);

  // Upload pixels data for the specified level of the 2D texture
  void upload(U32 level, U32 x, U32 y, U32 width, U32 height, GLenum format, GLenum type,
              const void* data);

  // Upload the whole image for the first level
  void upload(GLenum format, GLenum type, const void* data) {
    upload(0u, 0u, 0u, storage_.width, storage_.height, format, type, data);
  }

  //void upload_compressed();

  void generate_mipmap();
  /// + + + + + + + + + + + + + + + + + + + +


  /// Bind/Unbind texture as image for shaders I/O operations
  void bind_image(U32 image_unit, GLenum access, I32 level = 0);
  void unbind_image(U32 image_unit);


  /// Link a sampler to the texture, use NULL to unlink the sampler
  bool set_sampler_ptr(const Sampler *sampler_ptr);

  /// Check if the sampler is compatible with the texture
  bool is_sampler_compatible(const Sampler &sampler) const;

  /// Return true if the texture has been allocated
  bool is_allocated() const {
    return bAllocated_;
  }

  const StorageInfo_t& storage_info() const { return storage_; }

 private:
  const Sampler *sampler_ptr_;

  bool    bBound_;
  bool    bUseMipMap_;
  bool    bAllocated_;

  StorageInfo_t storage_;
};
  
}  // namespace aer

#endif  // AER_DEVICE_TEXTURE_2D_H_

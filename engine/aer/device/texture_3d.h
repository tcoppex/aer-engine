// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_DEVICE_TEXTURE_3D_H_
#define AER_DEVICE_TEXTURE_3D_H_

#include "aer/common.h"
#include "aer/device/texture.h"
#include "aer/device/sampler.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// Wrapper around OpenGL 3D texture object.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class Texture3D : public Texture {
 public:
  /// Store information on texture storage
  struct StorageInfo_t {
    U32     levels;
    GLenum  internalformat;
    U32     width;
    U32     height;
    U32     depth;
    GLenum  format;
    GLenum  type;
  };

  Texture3D() :
    Texture(GL_TEXTURE_3D),
    bAllocated_(false)
  {}


  ///  + + + + Binding aware methods  + + + +
  /// Allocate data for all levels of the texture
  void allocate(GLenum internalformat,
                U32 width,
                U32 height,
                U32 depth,
                U32 levels = 1u);

  void resize(aer::U32 width, aer::U32 height, aer::U32 depth);

  /// Upload pixels data for the specified level of the texture
  void upload(U32 level,
              U32 x, U32 y, U32 z, 
              U32 width, U32 height, U32 depth, 
              GLenum format, GLenum type,
              const void* data);

  /// Upload the whole image for the first level
  void upload(GLenum format, GLenum type, const void* data) {
    upload(0u,
           0u, 0u, 0u,
           storage_.width, storage_.height, storage_.depth,
           format, type,
           data);
  }
  /// + + + + + + + + + + + + + + + + + + + +


  /// Return true if the texture has been allocated
  bool is_allocated() const {
    return bAllocated_;
  }

  const StorageInfo_t& storage_info() const {
    return storage_;
  }


 private:
  bool bAllocated_;
  StorageInfo_t storage_;
};
  
}  // namespace aer

#endif  // AER_DEVICE_TEXTURE_3D_H_

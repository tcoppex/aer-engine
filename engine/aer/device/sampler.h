// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2014 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_DEVICE_SAMPLER_H_
#define AER_DEVICE_SAMPLER_H_

#include "aer/common.h"
#include "aer/device/device_resource.h"

// =============================================================================
namespace aer {
// =============================================================================

/**
 * @class Sampler
 * @brief Wrapper around OpenGL sampler object
 *
 * Describes how a texture / image is sampled on the device
*/
class Sampler : public DeviceResource { 
public:
  // ---------------------------------------------------------------------------
  /// @name Static methods
  // ---------------------------------------------------------------------------

  static
  void Unbind(U32 unit) {
    glBindSampler(unit, 0u);
  }

  static
  void UnbindAll(U32 count) {
    for (U32 i = 0; i < count; ++i) {
      Unbind(i);
    }
  }

  // ---------------------------------------------------------------------------
  /// @name Constructor
  // ---------------------------------------------------------------------------

  Sampler() :
    DeviceResource(),
    bUseMipmapFilter_(true)
  {}

  // ---------------------------------------------------------------------------
  /// @name DeviceResource methods
  // ---------------------------------------------------------------------------

  void generate() override {
    AER_ASSERT(!is_generated());
    glGenSamplers(1, &id_);
  }

  void release() override {
    if (is_generated()) {
      glDeleteSamplers(1, &id_);
      id_ = 0u;
    }
  }

  void bind(U32 unit) const {
    glBindSampler(unit, id_);
  }

  void unbind(U32 unit) const {
    Unbind(unit);
  }

  // ---------------------------------------------------------------------------
  /// @name Getters
  // ---------------------------------------------------------------------------

  bool use_mipmap_filter() const { 
    return bUseMipmapFilter_; 
  }


  // ---------------------------------------------------------------------------
  /// @name Setters
  // ---------------------------------------------------------------------------

  /// Set minification filter
  void set_min_filter(GLint filter) {
    AER_ASSERT((filter == GL_NEAREST) || (filter == GL_LINEAR));
    glSamplerParameteri(id_, GL_TEXTURE_MIN_FILTER, filter);
    bUseMipmapFilter_ = false;
  }

  /// Set magnification filter
  void set_mag_filter(GLint filter) {
    AER_ASSERT((filter == GL_NEAREST) || (filter == GL_LINEAR));
    glSamplerParameteri(id_, GL_TEXTURE_MAG_FILTER, filter);
  }

  /// Set both minification and magnification filters
  void set_filters(GLint minification, GLint magnification) {
    set_min_filter(minification);
    set_mag_filter(magnification);
  }

  /// Set minification filter if mipmapping is used
  /// filter_map specified the filtering inside the texture map
  /// filter_levels specified the filtering between mipmap levels
  void set_mipmap_min_filter(GLint filter_map, GLint filter_levels) {
    AER_ASSERT(   (filter_map == GL_NEAREST) || (filter_map == GL_LINEAR)   );
    AER_ASSERT((filter_levels == GL_NEAREST) || (filter_levels == GL_LINEAR));

    GLint filter;
    if (filter_map == GL_NEAREST) {
      filter = (filter_levels==GL_NEAREST) ? GL_NEAREST_MIPMAP_NEAREST
                                           : GL_NEAREST_MIPMAP_LINEAR;
    } else {
      filter = (filter_levels==GL_NEAREST) ? GL_LINEAR_MIPMAP_NEAREST
                                           : GL_LINEAR_MIPMAP_LINEAR;
    }
    glSamplerParameteri(id_, GL_TEXTURE_MIN_FILTER, filter);
    bUseMipmapFilter_ = true;
  }

  /// Set the anisotropy filtering level
  /// it requires GL_EXT_texture_filter_anisotropic, which is non core
  /// but present in most implementations.
  /// Note : anisotropic filter does not replace mipmapping filters
  void set_anisotropy_level(F32 level) {
    F32 max_level = opengl::GetF(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT);
    AER_ASSERT((level >= 1.0f) && (level <= max_level));
    level = glm::clamp(level, 1.0f, max_level);
    glSamplerParameterf(id_, GL_TEXTURE_MAX_ANISOTROPY_EXT, level);
  }

  /// Range affecting the mipmap level selection
  void set_LOD_range(const Vector2i &range) {
    glSamplerParameteri(id_, GL_TEXTURE_MIN_LOD, range.x);
    glSamplerParameteri(id_, GL_TEXTURE_MAX_LOD, range.y);
  }

  /// Bias used to adjust coarsely the mipmap level selection
  void set_LOD_bias(F32 bias) {
    glSamplerParameterf(id_, GL_TEXTURE_LOD_BIAS, bias);
  }

  /// Set the comparison mode
  /// used with depth texture to implement shadow mapping
  void set_compare_mode(GLint mode) {
    AER_ASSERT((mode == GL_COMPARE_REF_TO_TEXTURE) || (mode == GL_NONE));
    glSamplerParameteri(id_, GL_TEXTURE_COMPARE_MODE, mode);
  }

  /// Set comparison function
  void set_compare_func(GLint func) {
    glSamplerParameteri(id_, GL_TEXTURE_COMPARE_FUNC, func);
  }

  /// set the wrap mode for the three direction
  void set_wraps(GLint wrap_s, GLint wrap_t=GL_REPEAT, GLint wrap_r=GL_REPEAT) {
    glSamplerParameteri(id_, GL_TEXTURE_WRAP_S, wrap_s);
    glSamplerParameteri(id_, GL_TEXTURE_WRAP_T, wrap_t);
    glSamplerParameteri(id_, GL_TEXTURE_WRAP_R, wrap_r);
  }

  /// border color used when wrapping is set to GL_CLAMP_TO_BORDER
  void set_border_color(const Vector4& color) {
    glSamplerParameterfv(id_, GL_TEXTURE_BORDER_COLOR, &(color[0]));
  }


  // Requires ARB_seamless_cubemap_per_texture, not yet in core
  void enable_seamless_cubemap(bool state) {
#ifdef GL_TEXTURE_CUBE_MAP_SEAMLESS
    glSamplerParameteri(id_, GL_TEXTURE_CUBE_MAP_SEAMLESS, GLint(state));
#else
    AER_WARNING("extension (AMD/ARB)_seamless_cubemap_per_texture required");
#endif
  }

private:
  // ---------------------------------------------------------------------------
  /// @name Attributes
  // ---------------------------------------------------------------------------

  bool bUseMipmapFilter_;
};

// =============================================================================
}  // namespace aer
// =============================================================================

#include "aer/device/default_samplers.h"

#endif  // AER_DEVICE_SAMPLER_H_

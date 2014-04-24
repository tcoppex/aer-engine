// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_DEVICE_SHADER_H_
#define AER_DEVICE_SHADER_H_

#include <string>
#include "aer/common.h"
#include "aer/core/opengl.h"

// =============================================================================
namespace aer {
// =============================================================================

/**
 * @class ShaderView
 * @brief Represent a view on shader object, used to access a shader reference's 
 *        objects data without modifying it
 *
 * @note This is not a DeviceResource object
 * @see Shader
*/
class ShaderView {
public:
  /// Return a shader bitfield from a shader type
  static GLbitfield TypeToBitfield(GLenum type);

  /// Decrement the reference counter of the shader
  /// in the ShaderManager
  virtual void release() = 0;

  /// Name of the OpenGL object
  virtual U32 id() const = 0;

  /// Type of the shader
  virtual GLenum type() const = 0;

  /// @return true if the OpenGL object exist
  virtual bool is_created() const = 0;

  /// @return true if the shader was set to be dirty
  /// A shader is dirty when source was change without compilation
  virtual bool is_dirty() const = 0;

  /// @return true if the shader was recently update
  /// A shader is state as updated by the ShaderManager when it was
  /// recently recompiled in order for the program using it to relink
  virtual bool is_updated() const = 0;

  /// @return the bitfield equivalent of the shader type
  virtual GLbitfield bitfield() const = 0;
};

// =============================================================================

/**
 * @class Shader
 * @brief Handler on an OpenGL shader object
*/
class Shader : public ShaderView {
 public:
  Shader(const std::string &refname) :
    refname_(refname),
    id_(0u),
    type_(GL_NONE),
    bDirty_(false),
    bUpdated_(false)
  {}

  /// Create a shader object
  void create(GLenum type) {
    AER_ASSERT(!is_created());
    id_ = glCreateShader(type);
  }

  void release() override;

  /// Replace the source code associated to the shader object
  void set_source(const char* source) {
    AER_ASSERT(is_created());
    glShaderSource(id_, 1, (const GLchar**)&source, NULL);
  }
  
  /// Compile a shader object
  /// @return true if the compilation succeeds
  bool compile();

  /// Change the update state when recompiled to let program objects 
  /// using the shader acknowledge it.
  void set_update_state(bool state) {
    bUpdated_ = state;
  }

  // TODO
  //bool set_binary(GLenum binaryFormat, const void *binary, const U32 length);
  //void get_source(const U32 buffersize, char *dst_buffer);


  // ---------------------------------------------------------------------------
  /// @name Overrided methods
  // ---------------------------------------------------------------------------  

  U32 id() const override {
    return id_;
  }
 
  GLenum type() const override {
    return type_;
  }
 
  bool is_created() const override {
    return 0u != id_;
  }
  
  bool is_dirty() const override {
    return bDirty_;
  }
  
  bool is_updated() const override {
    return bUpdated_;
  }

  GLbitfield bitfield() const override {
    return TypeToBitfield(type_);
  }


protected:
  // ---------------------------------------------------------------------------
  /// @name Attributes
  // ---------------------------------------------------------------------------

  std::string refname_; //

  U32    id_;
  GLenum type_;
  bool   bDirty_;
  bool   bUpdated_;
};

// =============================================================================
}  // namespace aer
// =============================================================================

#endif  // AER_DEVICE_SHADER_H_

// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_DEVICE_SHADER_H_
#define AER_DEVICE_SHADER_H_

#include "aer/common.h"
#include "aer/core/opengl.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
///
/// Represent a view on shader object, use to access a
/// a shader reference's objects data without modifying it.
///
/// Note : this is NOT a DeviceResource object.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
class ShaderView {
 public:
  /// Return a shader bitfield from a shader type
  static
  GLbitfield TypeToBitfield(GLenum type);

  ShaderView() :
    id_(0u),
    type_(GL_NONE),
    bDirty_(false),
    bUpdated_(false)
  {}

  ~ShaderView() {
    release();
  }

  /// Decrement the reference counter of the shader
  /// in the ShaderManager
  void release();

  /// Name of the OpenGL object
  U32 id()          const { return id_; }

  /// Type of the shader
  GLenum type()     const { return type_; }

  /// Return true if the OpenGL object exist
  bool is_created() const { return 0u != id_; }

  /// Return true if the shader was set to be dirty
  /// A shader is dirty when source was change without compilation
  bool is_dirty()   const { return bDirty_; }

  /// Return true if the shader was recently update
  /// A shader is state as updated by the ShaderManager when it was
  /// recently recompiled in order for the program using it to relink
  bool is_updated() const { return bUpdated_; }

  /// Return the bitfield equivalent of the shader type
  GLbitfield bitfield() const { return TypeToBitfield(type_); }


 protected:
  U32    id_;
  GLenum type_;
  bool   bDirty_;
  bool   bUpdated_;
};


/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
///
/// Handler on an OpenGL shader object.
/// Can be used to modify the shader states.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
class Shader : public ShaderView {
 public:
  /// Create a shader object
  void create(GLenum type) {
    AER_ASSERT(!is_created());
    id_ = glCreateShader(type);
  }

  /// Replace the source code associated to the shader object
  void set_source(const char* source) {
    AER_ASSERT(is_created());
    glShaderSource(id_, 1, (const GLchar**)&source, NULL);
  }
  
  /// Compile a shader object
  bool compile();

  /// Change the update state when recompiled to let program objects 
  /// using the shader acknowledge it.
  void set_update_state(bool state) {
    bUpdated_ = state;
  }


  // TODO
  //bool set_binary(GLenum binaryFormat, const void *binary, const U32 length);
  //void get_source(const U32 buffersize, char *dst_buffer);
};

}  // namespace aer

#endif  // AER_DEVICE_SHADER_H_

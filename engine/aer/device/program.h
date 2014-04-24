// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_DEVICE_PROGRAM_H_
#define AER_DEVICE_PROGRAM_H_

#include <vector>

#include "aer/common.h"
#include "aer/core/opengl.h"
#include "aer/device/shader.h"

// =============================================================================
namespace aer {
// =============================================================================

struct UniformLocation_t;


/**
 * @class Program
 * @brief Wrapper around OpenGL program object
 *
 *
 * @note This is not a DeviceResource object
*/
class Program {
public:
  static
  void Deactivate() {
    glUseProgram(0u);
  }

  Program() :
    id_(0u),
    bitfield_(0u),
    bDirty_(false)
  {}

  ~Program() {
    release();
  }

  /// Create a program object
  inline bool create();

  /// Create a program object from a single shader
  /// It automatically makes it separable
  inline bool create(ShaderView* shader);

  /// Release the program object
  inline void release();

  //void load_binary(..);

  inline bool link();
  inline bool link(bool bSeparate);

  // Note : this methods make the program 'dirty' (ie. needs relink)
  // --------------------------------------------------------
  /// Add a shader to attach to the program object
  inline void add_shader(ShaderView* shader);

  /// Load & add a shader using the ShaderProxy
  //inline void load_shader(const char* id);
  //inline void load_shader(const char* id, GLenum type);

  /// bind a user-defined varying out variable to a fragment shader color 
  /// number.
  inline void bind_frag_data_location(const U32 color_number,
                                      const char* name);
  
  /// Bind a user-defined varying out variable to a fragment shader color 
  /// number and index
  inline void bind_frag_data_location_index(const U32 color_number,
                                            const U32 index,
                                            const char* name);

  /// Specify values to record in transform feedback buffers
  inline void transform_feedback_varyings(const U32 count,
                                          const char **varyings,
                                          GLenum buffer_mode);

  /// Parameters (should be set before linking)
  inline void set_separable(bool bSeparable);
  // --------------------------------------------------------

  /// Make the program current
  inline void activate();

  /// Make the program not current
  inline void deactivate();


  // ---------------------------------------------------------------------------
  /// @name Getters
  // ---------------------------------------------------------------------------

  inline U32 id() const;

  inline bool has_stages(GLbitfield stages) const;

  inline bool is_created()   const;
  inline bool is_dirty()     const;
  inline bool is_separable() const;


  // ---------------------------------------------------------------------------
  /// @name Direct State Access uniforms handling
  // ---------------------------------------------------------------------------

  /// Specify the value of a uniform variable for a specified program object
  inline UniformLocation_t uniform_location(const char *name);

  // Scalars
  inline void set_uniform(const UniformLocation_t &loc, I32 v) const;
  inline void set_uniform(const UniformLocation_t &loc, U32 v) const;
  inline void set_uniform(const UniformLocation_t &loc, F32 v) const;

  // Vectors
  inline void set_uniform(const UniformLocation_t &loc, const Vector2 &v) const;
  inline void set_uniform(const UniformLocation_t &loc, const Vector3 &v) const;
  inline void set_uniform(const UniformLocation_t &loc, const Vector4 &v) const;
  inline void set_uniform(const UniformLocation_t &loc, const Vector2i &v) const;
  inline void set_uniform(const UniformLocation_t &loc, const Vector3i &v) const;
  inline void set_uniform(const UniformLocation_t &loc, const Vector4i &v) const;
  
  // Matrices
  inline void set_uniform(const UniformLocation_t &loc, const Matrix3x3 &v) const;
  inline void set_uniform(const UniformLocation_t &loc, const Matrix4x4 &v) const;

  // Buffers
  inline void set_uniform(const UniformLocation_t &loc, const I32 *v,       U32 count) const;
  inline void set_uniform(const UniformLocation_t &loc, const U32 *v,       U32 count) const;
  inline void set_uniform(const UniformLocation_t &loc, const F32 *v,       U32 count) const;
  inline void set_uniform(const UniformLocation_t &loc, const Vector2 *v,   U32 count) const;
  inline void set_uniform(const UniformLocation_t &loc, const Vector3 *v,   U32 count) const;
  inline void set_uniform(const UniformLocation_t &loc, const Vector4 *v,   U32 count) const;
  inline void set_uniform(const UniformLocation_t &loc, const Matrix3x3 *v, U32 count) const;
  inline void set_uniform(const UniformLocation_t &loc, const Matrix4x4 *v, U32 count) const;

  template<typename T>
  void set_uniform(const char *uname, const T& v) {
    set_uniform(uniform_location(uname), v);
  }

  template<typename T>
  void set_uniform(const char *uname, const T* v, U32 count) {
    set_uniform(uniform_location(uname), v, count);
  }

  // -- Subroutine Uniforms
  /// Return the location of the named subroutine uniform
  //inline I32 subroutine_uniform_location(const GLenum stage, const char *name);

  /// Return the location of the named subroutine
  //inline I32 subroutine_index(const GLenum stage, const char *name);

  /// Set a buffer of matching subroutine uniform -> subroutine index
  //inline void uniform_subroutines(const GLenum stage, U32 size, const U32* indices);


  // -- Uniforms Blocks
  /// Assign a binding point to an active uniform block
  //inline void uniform_block_binding(U32 block_index, U32 block_binding);

  // -- Shader Storage Blocks
  /// Change an active shader storage block binding
  //inline void shader_storage_block_binding(U32 block_index, U32 block_binding);


  // [TODO]
  // -infos with glGetProgramStageiv / glGetProgramResource


private:
  typedef std::vector<ShaderView*> ShaderSet_t;

  // ---------------------------------------------------------------------------
  /// @name Attributes
  // ---------------------------------------------------------------------------

  GLuint        id_;
  GLbitfield    bitfield_;
  ShaderSet_t   shaders_;
  bool          bDirty_;


  DISALLOW_COPY_AND_ASSIGN(Program);
};

// -----------------------------------------------------------------------------

/**
 * @struct UniformLocation_t
 * @brief Wrapper around uniform location index
*/
struct UniformLocation_t {
  UniformLocation_t(I32 loc)
    : index(loc) 
  {}

  bool is_valid() const {
    return index >= 0;
  }

  I32 index = -1;
};

// =============================================================================
}  // namespace aer
// =============================================================================

#include "aer/device/program-inl.h"

#endif  // AER_DEVICE_PROGRAM_H_



#if 0
/// On subroutines uniform setting

    // Get subroutine info
    GLint suLoc = glGetSubroutineUniformLocation( pgm.getId(), GL_COMPUTE_SHADER, "suHBAO");
    AER_CHECK(suLoc == 0);
    GLuint ssaoX_id = glGetSubroutineIndex( pgm.getId(), GL_COMPUTE_SHADER, "HBAO_X");
    GLuint ssaoY_id = glGetSubroutineIndex( pgm.getId(), GL_COMPUTE_SHADER, "HBAO_Y");
    
    //---
    // the program must be current
    glUniformSubroutinesuiv(GL_COMPUTE_SHADER, 1, &ssaoX_id);


    subroutine array = [ s0_foo3 , s1_foo1, s2_foo1, s3_foo0, s4_foo2]
    -> indice i correspond to the i-th subroutine
    -> array[i] is the indice of the chosen routine

#endif
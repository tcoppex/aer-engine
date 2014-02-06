// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#include "aer/device/shader.h"


namespace aer {


GLbitfield ShaderView::TypeToBitfield(GLenum type) {
#define SHADER_CASE(t)   case t: return t##_BIT
  switch (type) {
    SHADER_CASE(GL_VERTEX_SHADER);
    SHADER_CASE(GL_TESS_CONTROL_SHADER);
    SHADER_CASE(GL_TESS_EVALUATION_SHADER);
    SHADER_CASE(GL_GEOMETRY_SHADER);
    SHADER_CASE(GL_FRAGMENT_SHADER);
    SHADER_CASE(GL_COMPUTE_SHADER);
    default:
      AER_ASSERT("Unknown shader type");
      return 0;
  }
#undef SHADER_CASE
}


void ShaderView::release() {
  AER_ASSERT(is_created());

  //AER_WARNING("Must call the ShaderManager.");
  U32 refcount = 0u;//ShaderManager::Get().release(this);
  if (refcount == 0u) {
    glDeleteShader(id_);
    id_ = 0u;
  }
}


/// - - - - - - - - - - - - - - - - - - - - - - - - - - - 


bool Shader::compile() {
  AER_ASSERT(is_created());
  
  GLint status;
  glCompileShader(id_);
  glGetShaderiv(id_, GL_COMPILE_STATUS, &status);
  return GL_TRUE == status;
}


}  // namespace aer

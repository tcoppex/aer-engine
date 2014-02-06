// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_DEVICE_PROGRAM_INL_H_
#define AER_DEVICE_PROGRAM_INL_H_

#include "glm/gtc/type_ptr.hpp"   // for value_ptr
#include "aer/utils/logger.h"
//#include "aer/loader/shader_proxy.h"


namespace aer {

bool Program::create() {
  AER_ASSERT(!is_created());
  id_ = glCreateProgram();
  return id_ > 0u;
}

bool Program::create(ShaderView const* shader) {
  AER_ASSERT(!is_created() && "program already created");
  bool status = create();
  add_shader(shader);
  return status && link(true);
}

void Program::release() {
  if (!is_created()) {
    return;
  }

  AER_WARNING("shader references not released by Program::release");
  
  /*
  for (auto shader : shaders_) {
    // it seems ugly but it's really not
    //Shader* s = reinterpret_cast<Shader*>(const_cast<ShaderView*>(shader));
    Shader *s = (Shader*)(shader);
    ShaderProxy::Get().release(s);
  }
  shaders_.clear();
  */

  glDeleteProgram(id_);
  id_ = 0u;
}

bool Program::link() {
  GLint status;
  glLinkProgram(id_);
  glGetProgramiv(id_, GL_LINK_STATUS, &status);
  AER_WARNING("linking log todo");
  bool bResult = (GL_TRUE == status);
  if (!bResult) {
    AER_WARNING("bwabwa linqwing eewoar");
  }
  bDirty_ = !bResult;
  return bResult;
}

bool Program::link(bool bSeparate) {
  set_separable(bSeparate);
  link();
}

// --------------------------------------------------------
void Program::add_shader(ShaderView const* shader) {
  bitfield_ |= shader->bitfield();
  shaders_.push_back(shader);

  glAttachShader(id_, shader->id());
  //glDeleteShader(shader->id()); //

  bDirty_ = true;
}

void Program::bind_frag_data_location(const U32 color_number,
                                      const char* name) {
  AER_ASSERT(has_stages(GL_FRAGMENT_SHADER_BIT));
  glBindFragDataLocation(id_, color_number, name);
  bDirty_ = true;
}

void Program::bind_frag_data_location_index(const U32 color_number,
                                            const U32 index,
                                            const char* name) {
  AER_ASSERT(has_stages(GL_FRAGMENT_SHADER_BIT));
  glBindFragDataLocationIndexed(id_, color_number, index, name);
  bDirty_ = true;
}

void Program::transform_feedback_varyings(const U32 count,
                                          const char **varyings,
                                          GLenum buffer_mode) {
  glTransformFeedbackVaryings(id_, count, varyings, buffer_mode);
  bDirty_ = true;
}

void Program::set_separable(bool bSeparable) {
  GLint status = (bSeparable) ? GL_TRUE : GL_FALSE;
  glProgramParameteri(id_, GL_PROGRAM_SEPARABLE, status);
  bDirty_ = true;
}
// --------------------------------------------------------


void Program::activate() {
  glUseProgram(id_);

  // update program dirtyness
  for (auto shader : shaders_) {
    bDirty_ |= shader->is_dirty();
  }
  AER_CHECK(!is_dirty());//
}

void Program::deactivate() {
  Deactivate();
}


U32 Program::id() const {
  return id_;
}

bool Program::has_stages(GLbitfield stages) const {
  return bitfield_ & stages;
}

bool Program::is_created() const {
  return 0u != id_;
}

bool Program::is_dirty() const {
  return bDirty_;
}

bool Program::is_separable() const {
  GLint status;
  glGetProgramiv(id_, GL_PROGRAM_SEPARABLE, &status);
  return GL_TRUE == status;
}



UniformLocation_t Program::uniform_location(const char *name) {
  UniformLocation_t loc(glGetUniformLocation(id_, name));
  AER_DEBUG_CODE(
    if (!loc.is_valid()) {
      std::string n(name);
      AER_WARNING("[debug] invalid uniform name : \""+n+"\"");
    }
  )
  return loc;
}


void Program::set_uniform(const UniformLocation_t &loc, I32 v) const {
  glUniform1i(loc.index, v);
}

void Program::set_uniform(const UniformLocation_t &loc, U32 v) const {
  glUniform1ui(loc.index, v);
}

void Program::set_uniform(const UniformLocation_t &loc, F32 v) const  {
  glUniform1f(loc.index, v);
}

void Program::set_uniform(const UniformLocation_t &loc, const Vector2 &v) const {
  glUniform2fv(loc.index, 1u, glm::value_ptr(v));
}

void Program::set_uniform(const UniformLocation_t &loc, const Vector3 &v) const {
  glUniform3fv(loc.index, 1u, glm::value_ptr(v));
}

void Program::set_uniform(const UniformLocation_t &loc, const Vector4 &v) const {
  glUniform4fv(loc.index, 1u, glm::value_ptr(v));
}

void Program::set_uniform(const UniformLocation_t &loc, const Vector2i &v) const {
  glUniform2iv(loc.index, 1u, glm::value_ptr(v));
}

void Program::set_uniform(const UniformLocation_t &loc, const Vector3i &v) const {
  glUniform3iv(loc.index, 1u, glm::value_ptr(v));
}

void Program::set_uniform(const UniformLocation_t &loc, const Vector4i &v) const {
  glUniform4iv(loc.index, 1u, glm::value_ptr(v));
}

void Program::set_uniform(const UniformLocation_t &loc, const Matrix3x3 &v) const {
  glUniformMatrix3fv(loc.index, 1, GL_FALSE, glm::value_ptr(v));
}

void Program::set_uniform(const UniformLocation_t &loc, const Matrix4x4 &v) const {
  glUniformMatrix4fv(loc.index, 1, GL_FALSE, glm::value_ptr(v));
}

void Program::set_uniform(const UniformLocation_t &loc, const I32 *v, U32 count) const {
  glUniform1iv(loc.index, count, v);
}

void Program::set_uniform(const UniformLocation_t &loc, const U32 *v, U32 count) const {
  glUniform1uiv(loc.index, count, v);
}

void Program::set_uniform(const UniformLocation_t &loc, const F32 *v, U32 count) const {
  glUniform1fv(loc.index, count, v);
}

void Program::set_uniform(const UniformLocation_t &loc, const Vector2 *v, U32 count) const {
  glUniform2fv(loc.index, count, glm::value_ptr(*v));
}

void Program::set_uniform(const UniformLocation_t &loc, const Vector3 *v, U32 count) const {
  glUniform3fv(loc.index, count, glm::value_ptr(*v));
}

void Program::set_uniform(const UniformLocation_t &loc, const Vector4 *v, U32 count) const {
  glUniform4fv(loc.index, count, glm::value_ptr(*v));
}

void Program::set_uniform(const UniformLocation_t &loc, const Matrix3x3 *v, U32 count) const {
  glUniformMatrix3fv(loc.index, 1, GL_FALSE, glm::value_ptr(*v));
}

void Program::set_uniform(const UniformLocation_t &loc, const Matrix4x4 *v, U32 count) const {
  glUniformMatrix4fv(loc.index, 1, GL_FALSE, glm::value_ptr(*v));
}

}  // namespace aer

#endif  // AER_DEVICE_PROGRAM_INL_H_

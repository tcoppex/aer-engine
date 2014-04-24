// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#include <cstdlib>

#include "aer/loader/shader_proxy.h"
#include "aer/utils/logger.h"

// =============================================================================
namespace aer {
// =============================================================================

ShaderProxy* ShaderProxy::sInstance = nullptr;
char ShaderProxy::sShaderPath[] = {'\0'};

// -----------------------------------------------------------------------------

Shader* ShaderProxy::get(const std::string& id) {
  size_t pos = id.find_last_of('.');
  std::string key = id.substr(pos+1u);
  ExtensionToTypeMapIterator_t it = extension_to_type_.find(key);
  if (it == extension_to_type_.end()) {
    return nullptr;
  }
  return get(id, it->second);
}

// -----------------------------------------------------------------------------

Shader* ShaderProxy::get(const std::string& id, GLenum type) {
  type_ = type;
  return ResourceProxy<Shader>::get(id);
}

// -----------------------------------------------------------------------------

Shader* ShaderProxy::load(const std::string& id) {
  if (sShaderPath[0] == '\0') {
    Logger::Get().error("no shader path specified to ShaderProxy");
  }
  
  const GLchar* source = glswGetShader(id.c_str());
  if (nullptr == source) {
    Logger::Get().error("shader \"" + id + "\" not found, check @ \"" + sShaderPath);
  }

  Shader *shader = new Shader(id);
  AER_ASSERT(nullptr != shader);

  shader->create(type_);
  shader->set_source(source);

  if (!shader->compile()) {
    glGetShaderInfoLog(shader->id(), AER_ARRAYSIZE(log_buffer), nullptr, log_buffer);
    Logger::Get().error(id + " : \n" + log_buffer);
  }  
  shader->set_update_state(true);

  AER_DEBUG_CODE(
  fprintf(stderr, "%s : compilation succeed !\n", id.c_str());
  )

  return shader;
}

// =============================================================================
}  // namespace aer
// =============================================================================

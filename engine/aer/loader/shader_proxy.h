// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_LOADER_SHADER_PROXY_H_
#define AER_LOADER_SHADER_PROXY_H_

#include <cstdio>
#include <string>
#include <unordered_map>

#include "glsw/glsw.h"
#include "aer/common.h"
#include "aer/memory/resource_proxy.h"
#include "aer/device/shader.h"


namespace aer {


/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
///
/// Manages shader resources.
/// It detects automatically the shader type based on its
/// extension, but users can specify it manually if needed.
///
/// The class is singletonize manually.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
class ShaderProxy : public ResourceProxy<Shader> {

//------------ Singleton

 public:
  static void Initialize() {
    AER_ASSERT(sInstance == nullptr);
    glswInit();
    AER_ASSERT(0 == glswGetError());
    sInstance = new ShaderProxy();
  }

  static void Deinitialize() {
    AER_ASSERT(sInstance != nullptr);
    glswShutdown();
    AER_SAFE_DELETE(sInstance);
  }

  static ShaderProxy& Get() {
    AER_ASSERT(sInstance != nullptr);
    return *sInstance;
  }

 private:
  static ShaderProxy* sInstance;
  static char sShaderPath[256u];

  ShaderProxy() {
    init();
  }

  void init() {
    set_shader_types();
  }


//------------ ResourceProxy

 public:
  void set_shader_path(const char *shader_path) {
    snprintf(sShaderPath, AER_ARRAYSIZE(sShaderPath), "%s", shader_path);
    glswSetPath(sShaderPath, ".glsl");
  }

  void add_directive_token(const char *token, const char *directive) {
    glswAddDirectiveToken(token, directive);
  }

  void release_compiler() {
    glReleaseShaderCompiler();
  }


  /// Override get to detect shader type automatically
  Shader* get(const std::string& id);

  /// Specialize GET to set the shader type
  Shader* get(const std::string& id, GLenum type);


 private:
  void set_shader_types() {
    extension_to_type_["VS"]  = GL_VERTEX_SHADER;
    extension_to_type_["TCS"] = GL_TESS_CONTROL_SHADER;
    extension_to_type_["TES"] = GL_TESS_EVALUATION_SHADER;
    extension_to_type_["GS"]  = GL_GEOMETRY_SHADER;
    extension_to_type_["FS"]  = GL_FRAGMENT_SHADER;
    extension_to_type_["CS"]  = GL_COMPUTE_SHADER;

    extension_to_type_["Vertex"]          = GL_VERTEX_SHADER;
    extension_to_type_["TessControl"]     = GL_TESS_CONTROL_SHADER;
    extension_to_type_["TessEvaluation"]  = GL_TESS_EVALUATION_SHADER;
    extension_to_type_["Geometry"]        = GL_GEOMETRY_SHADER;
    extension_to_type_["Fragment"]        = GL_FRAGMENT_SHADER;
    extension_to_type_["Compute"]         = GL_COMPUTE_SHADER;
  }

  /// Load override
  virtual Shader* load(const std::string& id) override;


  typedef std::unordered_map<std::string, GLenum> ExtensionToTypeMap_t;
  typedef ExtensionToTypeMap_t::iterator          ExtensionToTypeMapIterator_t;
  
  ExtensionToTypeMap_t extension_to_type_;    // Map to detected shader type
                                              // from extension
  
  char log_buffer[1024u];                     // Buffer to log compilation error
  
  GLenum type_;                               // saved argument from get()
};


}  // namespace aer

#endif  // AER_LOADER_SHADER_PROXY_H_

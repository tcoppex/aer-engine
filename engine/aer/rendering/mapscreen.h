// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_RENDERING_MAPSCREEN_H_
#define AER_RENDERING_MAPSCREEN_H_

#include "aer/common.h"
#include "aer/core/opengl.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// A simple static class used to send a triangle to the
/// GPU.
/// The shaders provided allow the triangle to be mapped 
/// to the screen as a fullscreen quad (eg. for postprocess)
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class MapScreen {
 public:
  /// Vertex shader used to map the triangle to the screen as a fullscreen quad
  static
  const char* GetVertexShaderSrc() {
    return AER_STRINGIFY(
      \x23 version 330\n

      out vec2 vTexCoord;

      void main() {
        vTexCoord.s = (gl_VertexID << 1) & 2;
        vTexCoord.t = gl_VertexID & 2;
        gl_Position = vec4(2.0f*vTexCoord - 1.0f, 0.0f, 1.0f);
      }
    );
  }

  /// Fragment shader used to map a texture to the triangle as a quad
  static
  const char* GetFragmentShaderSrc() {
    return AER_STRINGIFY(
      \x23 version 330\n

      in vec2 vTexCoord;
      out vec4 fragColor;
      uniform sampler2D uTex;

      void main() {
        fragColor = texture(uTex, vTexCoord);
      }
    );
  }

  static
  void Draw() {
    static VAO_t sVAO;
    sVAO.bind();
    glDrawArrays(GL_TRIANGLES, 0, 3);
    sVAO.unbind();
  }


 private:
  struct VAO_t
  {
    VAO_t() : id(0u) { 
      glGenVertexArrays(1, &id);
    }

    ~VAO_t() { 
      if (id) {
        glDeleteVertexArrays(1, &id); 
      }
    }

    void bind()   { glBindVertexArray(id); }
    void unbind() { glBindVertexArray(0u); }

    GLuint id;
  };
};

}  // namespace aer

#endif  // AER_RENDERING_MAPSCREEN_H_

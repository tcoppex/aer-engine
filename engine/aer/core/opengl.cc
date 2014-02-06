// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#include "aer/core/opengl.h"


namespace aer {
namespace opengl {

/// Initialization -------------------------------------------------------------

bool Initialize() {
  glewExperimental = GL_TRUE;

  GLenum result = glewInit();  
  if (GLEW_OK != result) {
    fprintf(stderr, "Error: %s\n", glewGetErrorString(result));
    return false;
  }

#if AER_DEBUG
  fprintf(stderr, "----------------------------------------------------\n");
  fprintf(stderr, "[ Debug ]\n");
  fprintf(stderr, "Device : %s\n",         GetRenderer());
  fprintf(stderr, "GLEW version : %s\n",   glewGetString(GLEW_VERSION));
  fprintf(stderr, "OpenGL version : %s\n", GetVersion());
  fprintf(stderr, "GLSL version : %s\n",   GetGLSLVersion());
  fprintf(stderr, "----------------------------------------------------\n\n");
#endif

  // Void the error handler (sometimes not initialized)
  glGetError();
  
  return true;
}


/// Errors handling ------------------------------------------------------------

const char* GetErrorString(GLenum err) {
  switch (err)
  {
    // [GetError]
    case GL_NO_ERROR:
      return AER_STRINGIFY(GL_NO_ERROR);
      
    case GL_INVALID_ENUM:
      return AER_STRINGIFY(GL_INVALID_ENUM);
      
    case GL_INVALID_VALUE:
      return AER_STRINGIFY(GL_INVALID_VALUE);
      
    case GL_INVALID_OPERATION:
      return AER_STRINGIFY(GL_INVALID_OPERATION);
      
    case GL_STACK_OVERFLOW:
      return AER_STRINGIFY(GL_STACK_OVERFLOW);
      
    case GL_STACK_UNDERFLOW:
      return AER_STRINGIFY(GL_STACK_UNDERFLOW);
      
    case GL_OUT_OF_MEMORY:
      return AER_STRINGIFY(GL_OUT_OF_MEMORY);
      
    case GL_TABLE_TOO_LARGE:
      return AER_STRINGIFY(GL_TABLE_TOO_LARGE);
      
              
    // [CheckFramebufferStatus]
    case GL_FRAMEBUFFER_COMPLETE:
      return AER_STRINGIFY(GL_FRAMEBUFFER_COMPLETE);
      
    case GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT:
      return AER_STRINGIFY(GL_FRAMEBUFFER_INCOMPLETE_ATTACHMENT);
      
    case GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT:
      return AER_STRINGIFY(GL_FRAMEBUFFER_INCOMPLETE_MISSING_ATTACHMENT);
      
    case GL_FRAMEBUFFER_UNSUPPORTED:
      return AER_STRINGIFY(GL_FRAMEBUFFER_UNSUPPORTED);
      
      
    default:
      return "GetErrorString : Unknown constant";
  }  
    
  return "";
}

void CheckError(const char* file, const int line, const char* errMsg, bool bExitOnFail) {
  GLenum err = glGetError();

  if (err != GL_NO_ERROR) {
    fprintf(stderr,
            "OpenGL error @ \"%s\" [%d] : %s [%s].\n",
            file, line, errMsg, GetErrorString(err));

    if (bExitOnFail) {
      exit(EXIT_FAILURE);
    }
  }
}


/// Device information ---------------------------------------------------------

I32 GetI(GLenum pname) {
  GLint data;
  glGetIntegerv(pname, &data);
  return static_cast<I32>(data);
}

F32 GetF(GLenum pname) {
  GLfloat data;
  glGetFloatv(pname, &data);
  return static_cast<F32>(data);
}

bool GetB(GLenum pname) {
  GLboolean status;
  glGetBooleanv(pname, &status);
  return status == GL_TRUE;
}

const uchar* GetVendor() {
  return glGetString(GL_VENDOR);
}

const uchar* GetRenderer() {
  return glGetString(GL_RENDERER);
}

const uchar* GetVersion() {
  return glGetString(GL_VERSION);
}

const uchar* GetGLSLVersion() {
  return glGetString(GL_SHADING_LANGUAGE_VERSION);
}

const I32 GetMinorVersion() {
  return GetI(GL_MINOR_VERSION);
}

const I32 GetMajorVersion() {
  return GetI(GL_MAJOR_VERSION);
}

const I32 GetAvailableMemory() {
  return GetI(GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX); 
}

const I32 GetCurrentMemory() {
  return GetI(GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX);
}


/// States management ----------------------------------------------------------

StatesInfo PopStates() {
  StatesInfo info;

  info.bDepthTest   = GetB(GL_DEPTH_TEST);
  info.bStencilTest = GetB(GL_STENCIL_TEST);
  info.bBlend       = GetB(GL_BLEND);
  info.bCullFace    = GetB(GL_CULL_FACE);
  info.bMultisample = GetB(GL_MULTISAMPLE);

  GLboolean colormask[4];
  glGetBooleanv(GL_COLOR_WRITEMASK, colormask);
  info.bRGBAMask[0] = (GL_TRUE == colormask[0]);
  info.bRGBAMask[1] = (GL_TRUE == colormask[1]);
  info.bRGBAMask[2] = (GL_TRUE == colormask[2]);
  info.bRGBAMask[3] = (GL_TRUE == colormask[3]);

  info.bDepthMask   = GetB(GL_DEPTH_WRITEMASK);
  info.bStencilMask = GetB(GL_STENCIL_WRITEMASK);

  GLint vp[4];
  glGetIntegerv(GL_VIEWPORT, vp);
  info.viewport = Vector4i(vp[0], vp[1], vp[2], vp[3]);

  info.cull_face_mode = GetI(GL_CULL_FACE_MODE);

  return info;
}

void PushStates(const StatesInfo &info) {
#define SET_STATE(b, cap)    (b) ? glEnable(cap) : glDisable(cap)
  SET_STATE(info.bDepthTest,    GL_DEPTH_TEST);
  SET_STATE(info.bStencilTest,  GL_STENCIL_TEST);
  SET_STATE(info.bBlend,        GL_BLEND);
  SET_STATE(info.bCullFace,     GL_CULL_FACE);
  SET_STATE(info.bMultisample,  GL_MULTISAMPLE);
#undef SET_STATE

  const bool *rgba = info.bRGBAMask;
  glColorMask(rgba[0], rgba[1], rgba[2], rgba[3]);

  glDepthMask(info.bDepthMask);
  glStencilMask(info.bStencilMask);

  const Vector4i &vp = info.viewport;
  glViewport(vp.x, vp.y, vp.z, vp.w);

  glCullFace(info.cull_face_mode);
}

}  // namespace opengl

}  // namespace aer

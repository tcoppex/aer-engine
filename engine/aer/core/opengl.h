// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_CORE_OPENGL_H_
#define AER_CORE_OPENGL_H_

#include "GL/glew.h"
#include "aer/common.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
/// Set of simple utility functions to handle OpenGL
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
namespace opengl {

/// Initialization -------------------------------------------------------------

bool Initialize();


/// Errors handling ------------------------------------------------------------

const char* GetErrorString(GLenum err);

void CheckError(const char* file, const int line, const char* errMsg, bool bExitOnFail);

#if AER_DEBUG
# define CHECKGLERROR(msg)  aer::opengl::CheckError(__FILE__, __LINE__, "" msg, true)
#else
# define CHECKGLERROR(msg)
#endif


/// Device information ---------------------------------------------------------

I32  GetI(GLenum pname);
F32  GetF(GLenum pname);
bool GetB(GLenum pname);

const uchar*  GetVendor();
const uchar*  GetRenderer();
const uchar*  GetVersion();
const uchar*  GetGLSLVersion();

const I32     GetMinorVersion();
const I32     GetMajorVersion();

// http://developer.download.nvidia.com/opengl/specs/GL_NVX_gpu_memory_info.txt
#ifndef GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX
# define GPU_MEMORY_INFO_TOTAL_AVAILABLE_MEMORY_NVX    0x9048
#endif
#ifndef GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX
# define GPU_MEMORY_INFO_CURRENT_AVAILABLE_VIDMEM_NVX  0x9049
#endif

const I32 GetAvailableMemory();
const I32 GetCurrentMemory();

// some constants
#if 0
const I32 MaxClipDistances()       { return GetI(GL_MAX_CLIP_DISTANCES); }
const I32 MaxCubeMapTextureSize()  { return GetI(GL_MAX_CUBE_MAP_TEXTURE_SIZE); }
const I32 MaxColorAttachments()    { return GetI(GL_MAX_COLOR_ATTACHMENTS); }
const I32 MaxDrawBuffer()          { return Geti(GL_MAX_DRAW_BUFFERS); }
const I32 MaxTextureImageUnits()   { return Geti(GL_MAX_TEXTURE_IMAGE_UNITS); }
const I32 MaxTextureSize()         { return Geti(GL_MAX_TEXTURE_SIZE); }
const I32 MaxTextureAnisotropy()   { return Geti(GL_MAX_TEXTURE_MAX_ANISOTROPY_EXT); }
const I32 MaxVertexAttribs()       { return Geti(GL_MAX_VERTEX_ATTRIBS); }
#endif

/// States management ----------------------------------------------------------

struct StatesInfo {
  bool bDepthTest;
  bool bStencilTest;
  bool bBlend;      
  bool bCullFace;
  bool bMultisample;

  bool bRGBAMask[4];
  bool bDepthMask;
  bool bStencilMask;

  Vector4i  viewport;
  GLint cull_face_mode;
};

StatesInfo PopStates();
void       PushStates(const StatesInfo &states);

}  // namespace opengl

}  // namespace aer

#endif  // AER_CORE_OPENGL_H_

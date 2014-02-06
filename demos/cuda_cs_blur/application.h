// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef CUDA_CS_BLUR_APPLICATION_H_
#define CUDA_CS_BLUR_APPLICATION_H_

#include "aer/aer.h"
#include "aer/app/application.h"
#include "aer/device/program.h"
#include "aer/device/texture_2d.h"
#include "aer/device/framebuffer.h"
#include "aer/rendering/shape.h"
#include "aer/view/free_camera.h"

#ifdef USE_CUDA
# include <cuda_gl_interop.h>
# include "cuda_cs_blur/cuda_tools.h"
#endif


class Application : public aer::Application {
 public:
  static const aer::U32 kDefaultRes = 512u;
  static const aer::U32 kBlockDim = 16u;
  static const aer::I32 kDefaultKernelRadius = 8;

  // Note : Kernel radius can be set to kBlockDim max but setting this
  //        will affect shared memory occupancy, optimized here.
  //        (testable by doubling the max kernel radius below)
  static const aer::I32 kMaxKernelRadius = kDefaultKernelRadius;


  Application(int argc, char* argv[]);
  ~Application();


 private:
  void init() override;
  void init_textures();
  void init_shaders();
  void init_scene();

  void frame() override;
  void render_scene();
  void map_screen();

  void help();

  void postprocess_ComputeShader();
  void postprocess_CUDA();


  aer::FreeCamera *mCamera;    

  // FrameBuffer variable used to render the
  // scene into a texture
  struct {
    aer::Framebuffer fbo;
    aer::Texture2D   texRGBA;
    aer::Texture2D   texDEPTH;
  } mFirstPass;


  // Variables used for transferring result between GPU's kernel 
  // and OpenGL
  struct {
#ifdef USE_CUDA
    cudaGraphicsResource_t texSRC_CUDAResource;
    cudaGraphicsResource_t texDST_CUDAResource;
    CUDATimer              cudaTimer;
#endif
    aer::Texture2D         texDST;
    GLuint                 query;
  } mInterop;

  // Program Shaders
  struct {
    aer::Program scene;        // Render default scene
    aer::Program compute;      // Compute Shader kernel
    aer::Program mapscreen;    // Map final texture to screen
  } mProgram;

  // Scene object
  aer::SphereRaw mSceneObject;
  
  // Params
  aer::U32 mKernelRadius;

  bool mbDisplayStats;
  bool mbUseCUDA;
};


#endif  // CUDA_CS_BLUR_APPLICATION_H_
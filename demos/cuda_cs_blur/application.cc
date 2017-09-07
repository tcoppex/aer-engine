// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#include "cuda_cs_blur/application.h"

#include "aer/aer.h"
#include "aer/rendering/mapscreen.h"

#ifdef USE_CUDA
# include "cuda_cs_blur/kernel_postprocess.cuh"
#endif


Application::Application(int argc, char* argv[])
  : aer::Application(argc, argv),
    mCamera(nullptr),
    mKernelRadius(kDefaultKernelRadius),
    mbDisplayStats(false),
    mbUseCUDA(false)
{}

Application::~Application() {
  AER_SAFE_DELETE(mCamera);

#ifdef USE_CUDA
  cudaGraphicsUnregisterResource(mInterop.texSRC_CUDAResource);
  cudaGraphicsUnregisterResource(mInterop.texDST_CUDAResource);
#endif

  glDeleteQueries(1, &mInterop.query);
}

void Application::init() {
  /// Window
  aer::Display_t display(kDefaultRes, kDefaultRes);
  create_window(display, "CUDA / OpenGL Compute Shader blur benchmark");

  //window().set_verticalsync(true);
  set_fps_control(true);
  set_fps_limit(60u);


  /// Camera
  aer::View view(aer::Vector3(0.0f, 0.0f, 30.0f),
                 aer::Vector3(0.0f, 0.0f, 0.0f),
                 aer::Vector3(0.0f, 1.0f, 0.0f));

  aer::Frustum frustum(glm::radians(60.0f), 1.0f, 0.1f, 1000.0f);

  mCamera = new aer::FreeCamera(view, frustum);
  mCamera->set_motion_factor(0.20f);
  mCamera->set_rotation_factor(0.15f);
  mCamera->enable_motion_noise(true);

  /// OpenGL settings
  glClearColor(0.5f, 0.15f, 0.15f, 1.0f);
  glEnable(GL_DEPTH_TEST);

  glGenQueries(1, &mInterop.query); //

  init_textures();
  init_shaders();
  init_scene();

  help();
}

void Application::init_textures() {
  /// Default sampler used by every textures
  aer::DefaultSampler::NearestClampled().bind(0);


  ///FirstPass buffer textures (RGBA + DEPTH)
  mFirstPass.texRGBA.generate();
  mFirstPass.texRGBA.bind();
  mFirstPass.texRGBA.allocate(GL_RGBA8, kDefaultRes, kDefaultRes);

  mFirstPass.texDEPTH.generate();
  mFirstPass.texDEPTH.bind();
  mFirstPass.texDEPTH.allocate(GL_DEPTH_COMPONENT24, kDefaultRes, kDefaultRes);

  /// Kernel's output Texture
  mInterop.texDST.generate();
  mInterop.texDST.bind();
  mInterop.texDST.allocate(GL_RGBA8, kDefaultRes, kDefaultRes);

  /// !! WARNING !!
  /// In ComputeShaders, image object must use an Immutable storage, specified with
  /// TexStorage2D
  /// ('texDST.allocate' is kept here because it sets parameters internally)
  glTexStorage2D(GL_TEXTURE_2D, 1, GL_RGBA8, kDefaultRes, kDefaultRes);

#ifdef USE_CUDA
  // Register the texture by CUDA to be later bound as a read-only CUDA texture
  CHECK_CUDA(
  cudaGraphicsGLRegisterImage(&mInterop.texSRC_CUDAResource, 
                              mFirstPass.texRGBA.id(),
                              GL_TEXTURE_2D,
                              cudaGraphicsRegisterFlagsReadOnly)
  );

  // Register the texture by CUDA to be later bound to a CUDA Surface reference
  CHECK_CUDA(
  cudaGraphicsGLRegisterImage(&mInterop.texDST_CUDAResource,
                              mInterop.texDST.id(),
                              GL_TEXTURE_2D,
                              cudaGraphicsRegisterFlagsSurfaceLoadStore)
  );
#endif

  /// FirstPass FBO
  mFirstPass.fbo.generate();
  mFirstPass.fbo.bind();
  mFirstPass.fbo.attach_color  (&mFirstPass.texRGBA,  GL_COLOR_ATTACHMENT0);
  mFirstPass.fbo.attach_special(&mFirstPass.texDEPTH, GL_DEPTH_ATTACHMENT);
  AER_CHECK(aer::Framebuffer::CheckStatus());
  mFirstPass.fbo.unbind();

  
  aer::Texture::Unbind(GL_TEXTURE_2D);
  CHECKGLERROR();
}

void CheckProgramStatus(GLuint program) {
  GLint status = 0;

  glGetProgramiv(program, GL_LINK_STATUS, &status);
  if (status != GL_TRUE) {
    char buffer[1024];
    glGetProgramInfoLog(program, 1024, 0, buffer);
    fprintf(stderr, "%s\n", buffer);
  }
}

void Application::init_shaders() {
  aer::ShaderProxy &sp = aer::ShaderProxy::Get();
  sp.set_shader_path(DATA_DIRECTORY "shaders/");
  sp.add_directive_token("*", "#version 420 core");
  sp.add_directive_token("CS", "#extension GL_ARB_compute_shader : enable");

  //--
  
  // Set static kernel's parameters
  char glslToken[64];
  sprintf(glslToken, "#define BLOCK_DIM %d", kBlockDim);
  sp.add_directive_token("CS", glslToken);
  sprintf(glslToken, "#define MAX_RADIUS %d", kMaxKernelRadius);
  sp.add_directive_token("CS", glslToken);
  
  // Passthrough program to render the scene
  mProgram.scene.create();
    mProgram.scene.add_shader(sp.get("PassThrough.VS"));
    mProgram.scene.add_shader(sp.get("PassThrough.FS"));
  AER_CHECK(mProgram.scene.link());

  // Compute Shader kernel to blur input image
  AER_CHECK(mProgram.compute.create(sp.get("PostProcess.CS")));
  CheckProgramStatus(mProgram.compute.id());

  // Map a texture to the screen
  mProgram.mapscreen.create();
    mProgram.mapscreen.add_shader(sp.get("MapScreen.VS"));
    mProgram.mapscreen.add_shader(sp.get("MapScreen.FS"));
  AER_CHECK(mProgram.mapscreen.link());

  CHECKGLERROR();
}

void Application::init_scene() {
  mSceneObject.init(4.0f, 32u);
  CHECKGLERROR();
}

void Application::frame() {
  const aer::EventsHandler &ev = aer::EventsHandler::Get();


  if (ev.key_pressed(aer::Keyboard::Escape)) {
    quit();
  }

  if (ev.key_pressed(aer::Keyboard::H)) {
    help();
  }

  if (ev.key_down(aer::Keyboard::T)) {
    mbDisplayStats ^= true;
  }
  
  bool bKernelRadiusUpdated = false;
  
  if (ev.key_pressed(aer::Keyboard::Add)) {
    mKernelRadius = std::min(aer::I32(mKernelRadius+1), aer::I32(kMaxKernelRadius));
    bKernelRadiusUpdated = true;
  }

  if (ev.key_pressed(aer::Keyboard::Subtract)) {
    mKernelRadius = std::max(aer::I32(mKernelRadius-1), aer::I32(0));
    bKernelRadiusUpdated = true;
  }

#ifdef USE_CUDA
  if (ev.key_pressed(aer::Keyboard::Space)) {
    mbUseCUDA = !mbUseCUDA;
    printf("Kernel : %s\n", (mbUseCUDA) ? "CUDA" : "GLSL Compute Shader");
  }
#endif

  if (bKernelRadiusUpdated) {
    printf("Kernel radius : %d\n", mKernelRadius);
  }

  // --------

  mCamera->update();

  mFirstPass.fbo.bind(GL_DRAW_FRAMEBUFFER);
  render_scene();
  mFirstPass.fbo.unbind();

  map_screen();
}

void Application::render_scene() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  aer::Program &pgm = mProgram.scene;
  pgm.activate();
  {
    const aer::Matrix4x4 &mvp = mCamera->view_projection_matrix();
    pgm.set_uniform("uModelViewProjMatrix", mvp);
    mSceneObject.draw();
  }
  pgm.deactivate();

  CHECKGLERROR();
}

void Application::map_screen() {
  /// Apply post-process in a GPU kernel
  if (mbUseCUDA) {
    postprocess_CUDA();    
  } else {
    postprocess_ComputeShader();  
  }

  /// Render output image
  glDisable(GL_DEPTH_TEST);
  aer::Program &pgm = mProgram.mapscreen;
  pgm.activate();
  {
    mInterop.texDST.bind(0);
    pgm.set_uniform("uSceneTex", 0);
    aer::MapScreen::Draw();
    mInterop.texDST.unbind();
  }
  pgm.deactivate();
  glEnable(GL_DEPTH_TEST);

  CHECKGLERROR();
}

void Application::help() {
#define NEW_LINE  "\n" \

  fprintf(stdout, NEW_LINE
  "----------------------------------------------------------------------" NEW_LINE
  "This demo shows GLSL 4.2 Compute Shader capabilities." NEW_LINE
  "A 16*16 blur filter is apply to a real-time scene as a post-process stage." NEW_LINE
  "A CUDA kernel is also available to compares performances.\n" NEW_LINE
  "Controls:" NEW_LINE
  "[Z-Q-S-D + mouse] or [SixAxis pad] : control the camera." NEW_LINE
  "[+/-] : change kernel radius." NEW_LINE
  "[T] : display GPU kernel execution time." NEW_LINE
  "[Space] : toggle CUDA or GLSL kernel." NEW_LINE
  "[H] : display this help." NEW_LINE
  "[ESCAPE] : quit the application." NEW_LINE
  "----------------------------------------------------------------------" NEW_LINE
  "\n");

#undef NEW_LINE
}

void Application::postprocess_ComputeShader() {
  glBeginQuery(GL_TIME_ELAPSED, mInterop.query);

  aer::Program &pgm = mProgram.compute;
  pgm.activate();
    // Bind source texture in TEXTURE_UNIT 0
    mFirstPass.texRGBA.bind(0u);
    pgm.set_uniform("uSrcTex", 0);

    // Bind destination texture to IMAGE_UNIT 0
    mInterop.texDST.bind_image(0u, GL_WRITE_ONLY);
    pgm.set_uniform("uDstImg", 0);

    pgm.set_uniform("uRadius", aer::I32(mKernelRadius));

    // Launch kernel
    glDispatchCompute(kDefaultRes/kBlockDim, kDefaultRes/kBlockDim, 1);

    // Unbind src & dst textures
    mFirstPass.texRGBA.unbind();
    mInterop.texDST.unbind_image(0u);
  pgm.deactivate();

  glEndQuery(GL_TIME_ELAPSED);


  if (mbDisplayStats) {
    GLint nanosecond = 0;
    glGetQueryObjectiv(mInterop.query, GL_QUERY_RESULT, &nanosecond);
    aer::F32 ms = nanosecond/1000000.0f;
    fprintf(stderr, "%.3f ms\n", ms);
    mbDisplayStats = false;
  }

  CHECKGLERROR();
}


void Application::postprocess_CUDA() {
#ifdef USE_CUDA
  mInterop.cudaTimer.Start();

  cudaArray *d_inArray = nullptr;
  CHECK_CUDA(cudaGraphicsMapResources(1, &mInterop.texSRC_CUDAResource));
  CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&d_inArray, mInterop.texSRC_CUDAResource, 0, 0));

  cudaArray *d_outArray = nullptr;
  CHECK_CUDA(cudaGraphicsMapResources(1, &mInterop.texDST_CUDAResource));
  CHECK_CUDA(cudaGraphicsSubResourceGetMappedArray(&d_outArray, mInterop.texDST_CUDAResource, 0, 0));

  const dim3 gridDim(kDefaultRes/kBlockDim, kDefaultRes/kBlockDim);
  const dim3 blockDim(kBlockDim, kBlockDim);
  const size_t tileWidth = (kBlockDim + 2*mKernelRadius);
  const size_t smemSize = tileWidth * tileWidth * 4 * sizeof(unsigned char);

  launch_cuda_kernel(gridDim, blockDim, smemSize, 
                     d_inArray, d_outArray, kDefaultRes, tileWidth, mKernelRadius);


  CHECK_CUDA(cudaGraphicsUnmapResources(1, &mInterop.texSRC_CUDAResource));
  CHECK_CUDA(cudaGraphicsUnmapResources(1, &mInterop.texDST_CUDAResource));

  mInterop.cudaTimer.Stop();

  // Note : CUDATimer is ineffective when screen vertical sync is enabled.
  if (mbDisplayStats) {
    aer::F32 ms = mInterop.cudaTimer.Elapsed();
    fprintf(stderr, "%.3f ms\n", ms);
    mbDisplayStats = false;
  }

  CHECKCUDAERROR();
#endif
}

// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#include "aura/application.h"

#include "aer/aer.h"
#include "aer/rendering/mapscreen.h"

// =============================================================================

Application::Application(int argc, char* argv[]) :
  aer::Application(argc, argv),
  mCamera(nullptr),
  mAOTexturePtr(nullptr)
{}

// -----------------------------------------------------------------------------

Application::~Application() {
  AER_SAFE_DELETE(mCamera);
}

// -----------------------------------------------------------------------------

void Application::init() {
  srand(time(NULL));

  /// Window
  aer::Display_t display(1280, 720);
  display.msaa_level = 0u;
  create_window(display, "aura");

  // Enable fps control
  set_fps_control(true);
  set_fps_limit(60u);

  /// Camera
  aer::View view(aer::Vector3(0.0f, 10.0f, 7.0f),
                 aer::Vector3(0.0f, 9.0f, 0.0f),
                 aer::Vector3(0.0f, 1.0f, 0.0f));

  aer::F32 ratio = window().display().aspect_ratio();
  aer::Frustum frustum(glm::radians(60.0f), ratio, 0.1f, 1000.0f);

  mCamera = new aer::FreeCamera(view, frustum);
  mCamera->set_motion_factor(0.20f);
  mCamera->set_rotation_factor(0.15f);
  mCamera->enable_motion_noise(true);

  /// OpenGL settings
  glClearColor(0.25f, 0.25f, 0.25f, 1.0f);

  /// The post-process textures & FBO must be set as MULTISAMPLE
  /// for this to works
  if (display.msaa_level > 1u) {
    glEnable(GL_MULTISAMPLE);
  } else {
    glDisable(GL_MULTISAMPLE);
  }

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);

  glEnable(GL_CULL_FACE);
  glCullFace(GL_BACK);

  // Application settings
  init_textures();
  init_shaders();
  init_scene();

  mPass.hbao.init(&mBufferingPass.texDEPTH);

  help();
}

// -----------------------------------------------------------------------------

void Application::init_textures() { 
  const aer::U32 width  = window().display().width;
  const aer::U32 height = window().display().height;

  // - Buffer textures
  mBufferingPass.texRGBA.generate();
  mBufferingPass.texRGBA.bind();
  mBufferingPass.texRGBA.allocate(GL_RGBA8, width, height, 1u, true);

  mBufferingPass.texDEPTH.generate();
  mBufferingPass.texDEPTH.bind();
  mBufferingPass.texDEPTH.allocate(GL_DEPTH_COMPONENT24, width, height, 1u, true);
  aer::Texture::Unbind(GL_TEXTURE_2D);

  // - FBO
  mBufferingPass.FBO.generate();
  mBufferingPass.FBO.bind();
  mBufferingPass.FBO.attach_color  (&mBufferingPass.texRGBA,  GL_COLOR_ATTACHMENT0);
  mBufferingPass.FBO.attach_special(&mBufferingPass.texDEPTH, GL_DEPTH_ATTACHMENT);
  AER_CHECK(aer::Framebuffer::CheckStatus());
  mBufferingPass.FBO.unbind();

  CHECKGLERROR();
}

// -----------------------------------------------------------------------------

void Application::init_shaders() {
  aer::ShaderProxy &sp = aer::ShaderProxy::Get();
  sp.set_shader_path(DATA_DIRECTORY"/shaders/");
  sp.add_directive_token("*", "#version 420 core");
  sp.add_directive_token("CS", "#extension GL_ARB_compute_shader : enable");

  //--

  mProgram.scene.create();
    mProgram.scene.add_shader(sp.get("FillBuffer.VS"));
    mProgram.scene.add_shader(sp.get("FillBuffer.FS"));
  AER_CHECK(mProgram.scene.link());

  mProgram.mapscreen.create();
    mProgram.mapscreen.add_shader(sp.get("MapScreen.VS"));
    mProgram.mapscreen.add_shader(sp.get("MapScreen.FS"));
  AER_CHECK(mProgram.mapscreen.link());
}

// -----------------------------------------------------------------------------

void Application::init_scene() {
  mScene.character.init();
  mScene.skydome.init();

  const aer::F32 plane_size = 32.0f;
  mScene.floorPlane.init(plane_size, plane_size, 8u);

  CHECKGLERROR();
}

// -----------------------------------------------------------------------------

void Application::frame() {
  const aer::EventsHandler &ev  = aer::EventsHandler::Get();
        aer::GlobalClock &clock = aer::GlobalClock::Get();


  /// Exit the application
  if (ev.key_pressed(aer::Keyboard::Escape)) {
    quit();
  }

  /// Display FPS
  if (ev.key_down(aer::Keyboard::F)) {
    fprintf(stderr,"fps : %u (%.3f ms)\r", clock.fps(), clock.delta_time());
  }

  /// Pause / Resume the clock
  if (ev.key_pressed(aer::Keyboard::Space) || 
      ev.joystick_button_pressed(aer::Joystick::Start)) {
    (clock.is_paused()) ? clock.resume() : clock.pause();
  }

  /// While 'W' down, render scene in wireframe mode
  if (ev.key_down(aer::Keyboard::W) ||
      ev.joystick_button_down(aer::Joystick::Cross)) {
    glPolygonMode(GL_FRONT_AND_BACK, GL_LINE);
  } else {
    glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);
  }

  // 'H' display the help
  if (ev.key_pressed(aer::Keyboard::H)) {
    help();
  }

  bool bSSAO = true;
  if (ev.key_down(aer::Keyboard::A)) {
    bSSAO = false;
  }


  //-------------

  mCamera->update();
  mScene.character.update();

  //-------------

  aer::opengl::StatesInfo states = aer::opengl::PopStates();

  if (!bSSAO) {
    render_scene(*mCamera);
  } else {
    render_deferred(*mCamera);
  }

  aer::opengl::PushStates(states);

  CHECKGLERROR();
}

// -----------------------------------------------------------------------------

void Application::render_scene(const aer::Camera &camera) {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  /// Skydome
  mScene.skydome.render(camera);

  /// Character
  mScene.character.render(camera);

  /// Basic scene
  const aer::Matrix4x4 &viewProj = camera.view_projection_matrix();
  aer::Matrix4x4 mvp;

  mProgram.scene.activate();
    mvp = viewProj;
    mProgram.scene.set_uniform("uModelViewProjMatrix", mvp);
    mProgram.scene.set_uniform("uColor", aer::Vector3(0.42f, 0.4f, 0.35f));
    mScene.floorPlane.draw();
  mProgram.scene.deactivate();
  glMemoryBarrier(GL_ALL_BARRIER_BITS);

  CHECKGLERROR();
}

// -----------------------------------------------------------------------------

void Application::render_deferred(const aer::Camera &camera) {
    //1) Render scene to the FBO
  mBufferingPass.FBO.bind(GL_DRAW_FRAMEBUFFER);
  render_scene(*mCamera);
  mBufferingPass.FBO.unbind();

  glDepthMask(GL_FALSE);
  glDisable(GL_DEPTH_TEST);

  // 2) Compute SSAO
  mPass.hbao.process(&mAOTexturePtr, mCamera->frustum());

  // 3) Render final texture to the screen
  postprocess();
}

// -----------------------------------------------------------------------------

void Application::postprocess() {
  AER_ASSERT(mAOTexturePtr != nullptr);
  
  glPolygonMode(GL_FRONT_AND_BACK, GL_FILL);

  aer::DefaultSampler::LinearClampled().bind(0);
  aer::DefaultSampler::LinearClampled().bind(1);

  mProgram.mapscreen.activate();
    mAOTexturePtr->bind(0);
    mProgram.mapscreen.set_uniform("uAOTex", 0);

    mBufferingPass.texRGBA.bind(1);
    mProgram.mapscreen.set_uniform("uSceneTex", 1);

    aer::MapScreen::Draw();
  mProgram.mapscreen.deactivate();

  aer::Texture::UnbindAll(GL_TEXTURE_2D, 2);
  aer::Sampler::UnbindAll(2);

  CHECKGLERROR();
}

// -----------------------------------------------------------------------------

void Application::help() {
#define NEW_LINE  "\n" \

  fprintf(stdout, NEW_LINE
  "----------------------------------------------------------------------" NEW_LINE
  "Aura is a technical demo demonstrating the capabilities of OpenGL 4.3." NEW_LINE
  //"It consists of a full deferred renderer with post-processing stages," NEW_LINE
  //"some great rendering effects, and an animation pipeline.\n" NEW_LINE
  "(Work In Progress)\n" NEW_LINE
  "Controls:" NEW_LINE
  "[Z-Q-S-D + mouse] or [SixAxis pad] : control the camera." NEW_LINE
  "[Space] or [SixAxis start] : pause / resume the clock." NEW_LINE
  "[A] : toggle HBAO." NEW_LINE
  "[M] : switch between Dual Quaternion & Linear blend skinning." NEW_LINE
  "[F] : display FPS." NEW_LINE
  "[W] : wireframe mode." NEW_LINE
  "[H] : display this help." NEW_LINE
  "[ESCAPE] : quit the application." NEW_LINE
  "----------------------------------------------------------------------" NEW_LINE
  "\n");

#undef NEW_LINE
}

// =============================================================================

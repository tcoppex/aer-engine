// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2014 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#include "hair/application.h"


// =============================================================================

Application::Application(int argc, char* argv[]) :
  aer::Application(argc, argv),
  mCamera(nullptr)
{}

// -----------------------------------------------------------------------------

Application::~Application() {
  AER_SAFE_DELETE(mCamera);
}

// -----------------------------------------------------------------------------

void Application::init() {
  /// Window
  aer::Display_t display(kDefaultWidth, kDefaultHeight);
  display.msaa_level = 4u;
  create_window(display, "Hair Simulation with tesselation shader");

  // Enable fps control
  set_fps_control(true);
  set_fps_limit(60u);


  /// Camera
  aer::View view(aer::Vector3(-40.0f, 30.0f, 40.0f),
                 aer::Vector3(0.0f, 0.0f, 0.0f),
                 aer::Vector3(0.0f, 1.0f, 0.0f));

  aer::F32 ratio = window().display().aspect_ratio();
  aer::Frustum frustum(60.0f, ratio, 0.1f, 500.0f);

  mCamera = new aer::FreeCamera(view, frustum);
  mCamera->set_motion_factor(0.20f);
  mCamera->set_rotation_factor(0.15f);
  mCamera->enable_motion_noise(false);

  /// OpenGL settings
  glClearColor(0.25f, 0.25f, 0.25f, 1.0f);

  if (display.msaa_level > 1u) {
    glEnable(GL_MULTISAMPLE);
  } else {
    glDisable(GL_MULTISAMPLE);
  }

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);

  // Application settings
  aer::ShaderProxy &sp = aer::ShaderProxy::Get();
  sp.set_shader_path(DATA_DIRECTORY"/shaders/");
  sp.add_directive_token("*", "#version 420 core");

  mHairSim.init();

  help();
}

// -----------------------------------------------------------------------------

void Application::frame() {
  const auto &ev = aer::EventsHandler::Get();

  if (ev.key_pressed(aer::Keyboard::Escape)) {
    quit();
  }

  mCamera->update();
  mHairSim.update();

  render_scene();
}

// -----------------------------------------------------------------------------

void Application::render_scene() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  mHairSim.render(*mCamera);
  CHECKGLERROR();
}

// -----------------------------------------------------------------------------

void Application::help() {
#define NEW_LINE  "\n" \

  fprintf(stdout, NEW_LINE
  "----------------------------------------------------------------------" NEW_LINE
  "Hair generation & simulation using Tesselation Shader.\n" NEW_LINE
  
  "Controls:" NEW_LINE
  "[Z-Q-S-D + mouse] or [SixAxis pad] : control the camera." NEW_LINE
  "[Up / Down] : change the number of control points tesselated." NEW_LINE
  "[Left / Right] : change the number of lines tesselated." NEW_LINE
  "[C] : enable / disable hair curling." NEW_LINE
  "[ESCAPE] : quit the application." NEW_LINE
  "----------------------------------------------------------------------" NEW_LINE
  "\n");

#undef NEW_LINE
}

// =============================================================================

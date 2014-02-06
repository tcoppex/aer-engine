// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#include "marching_cube/application.h"

#include "aer/aer.h"
//#include "aer/rendering/mapscreen.h"


Application::Application(int argc, char* argv[]) :
  aer::Application(argc, argv),
  mCamera(nullptr)
{}

Application::~Application() {
  AER_SAFE_DELETE(mCamera);
}


void Application::init() {
  /// Window
  aer::Display_t display(1280, 720);
  create_window(display, "marching cube");

  // Enable fps control
  set_fps_control(true);
  set_fps_limit(60u);


  /// Camera
  aer::View view(aer::Vector3(0.0f, 1.0f, 45.0f),
                 aer::Vector3(0.0f, 0.0f, 0.0f),
                 aer::Vector3(0.0f, 1.0f, 0.0f));

  aer::F32 ratio = window().display().aspect_ratio();
  aer::Frustum frustum(60.0f, ratio, 0.01f, 1000.0f);

  mCamera = new aer::FreeCamera(view, frustum);
  mCamera->set_motion_factor(0.20f);
  mCamera->set_rotation_factor(0.15f);
  mCamera->enable_motion_noise(false);

  /// OpenGL settings
  glClearColor(0.74f, 0.64f, 0.54f, 1.0f);

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);

  //glEnable(GL_CULL_FACE);
  //glCullFace(GL_BACK);


  // Application settings
  aer::ShaderProxy &sp = aer::ShaderProxy::Get();
  sp.set_shader_path(DATA_DIRECTORY "shaders/");
  sp.add_directive_token("*", "#version 430");


  mRenderer.init();
  mRenderer.generate(aer::Vector3(8, 4, 8)); //

  help();
}

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

  //---------------------

  mCamera->update();

  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  mRenderer.render(*mCamera);


  CHECKGLERROR();
}

void Application::help() {
  const char* filename = DATA_DIRECTORY "help_message";
  FILE *fd = fopen(filename, "r");

  if (!fd) {
    fprintf(stderr, "Can't find \"%s\"\n", filename);
    return;
  }

  fputc('\n', stdout);
  int c;
  for (int c = fgetc(fd); c != EOF; c = fgetc(fd)) {
    fputc(c, stdout);
  }
  fclose(fd);

  return;
}

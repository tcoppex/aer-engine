// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#include "marcheur/application.h"

#include "aer/aer.h"


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
  display.msaa_level = 4u;
  create_window(display, "marcheur");

  // Enable fps control
  set_fps_control(true);
  set_fps_limit(60u);

  /// Camera
  aer::View view(aer::Vector3(5.0f, 3.0f, 4.0f),
                 aer::Vector3(0.0f, 2.0f, 0.0f),
                 aer::Vector3(0.0f, 1.0f, 0.0f));

  aer::F32 ratio = window().display().aspect_ratio();
  aer::Frustum frustum(glm::radians(60.0f), ratio, 0.1f, 1000.0f);

  mCamera = new aer::FreeCamera(view, frustum);
  mCamera->set_motion_factor(0.20f);
  mCamera->set_rotation_factor(0.15f);
  //mCamera->enable_motion_noise(true);

  /// OpenGL settings
  glClearColor(0.25f, 0.25f, 0.25f, 1.0f);

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
  srand(time(NULL));
  init_textures();
  init_shaders();
  init_scene();

  help();
}

void Application::init_textures() { 
  //const aer::U32 width  = window().display().width;
  //const aer::U32 height = window().display().height;

  CHECKGLERROR();
}

void Application::init_shaders() {
  aer::ShaderProxy &sp = aer::ShaderProxy::Get();
  sp.set_shader_path(DATA_DIRECTORY"/shaders/");
  sp.add_directive_token("*", "#version 420 core");
  sp.add_directive_token("CS", "#extension GL_ARB_compute_shader : enable");

  //--

  mProgram.scene.create();
    mProgram.scene.add_shader(sp.get("Noise.Include", GL_VERTEX_SHADER));
    mProgram.scene.add_shader(sp.get("Terrain.VS"));
    mProgram.scene.add_shader(sp.get("Terrain.FS"));
  AER_CHECK(mProgram.scene.link());

  mProgram.moon.create();
    mProgram.moon.add_shader(sp.get("Noise.Include", GL_VERTEX_SHADER));
    mProgram.moon.add_shader(sp.get("Moon.VS"));
    mProgram.moon.add_shader(sp.get("Moon.FS"));
  AER_CHECK(mProgram.moon.link());
}


void Application::init_scene() {
  mScene.character.init();
  mScene.floorPlane.init(32.0f, 64.0f, 24u);

  //-------
  aer::Mesh &mesh = mScene.moon_plane;

  mesh.init(1, false);
  mesh.set_primitive_mode(GL_TRIANGLE_STRIP);
  mesh.set_vertex_count(4u);
  mesh.begin_update();
  aer::DeviceBuffer &vbo = mesh.vbo();
  vbo.bind(GL_ARRAY_BUFFER);
  {
    aer::F32 data[] = {
        -0.5f, -0.5f,
        -0.5f, +0.5f,
        +0.5f, -0.5f,
        +0.5f, +0.5f
    };

    aer::U32 buffersize = sizeof(data);
    vbo.allocate(buffersize, GL_STATIC_READ);

    glBindVertexBuffer(0, vbo.id(), 0, 2*sizeof(data[0]));
    glVertexAttribFormat(0, 2, GL_FLOAT, GL_FALSE, 0);
    glVertexAttribBinding(0, 0);
    glEnableVertexAttribArray(0);
    vbo.upload(0, buffersize, data);
  }
  vbo.unbind();
  mesh.end_update();
  //-------

  CHECKGLERROR();
}


//---------------------------------------------
//---------------------------------------------


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

  render_scene(*mCamera);

  CHECKGLERROR();
}

void Application::render_scene(const aer::Camera &camera) {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

  /// Character
  mScene.character.render(camera);

  /// Basic scene
  const aer::Matrix4x4 &viewProj = camera.view_projection_matrix();
  aer::Matrix4x4 mvp;

  mProgram.scene.activate();
    mvp = viewProj;
    mProgram.scene.set_uniform("uModelViewProjMatrix", mvp);
    aer::F32 t = aer::GlobalClock::Get().application_time(aer::SECOND);
    mProgram.scene.set_uniform("uTime", t);
    mProgram.scene.set_uniform("uColor", aer::Vector3(0.5f, 0.31f, 0.02f));
    mScene.floorPlane.draw();
  mProgram.scene.deactivate();


  const aer::Vector3 base = aer::Vector3(0.0f, 0.0f, -1.0f);
  aer::Vector3 direction  = glm::normalize(aer::Vector3(-1.0f, 0.7f, -0.5f));
  //aer::F32 angle    = acos(glm::dot(base, direction));
  aer::Vector3 axis = glm::cross(base, direction);
  aer::Matrix4x4 mrot = glm::rotate(glm::mat4(1.0f), 100.f, axis);

  mProgram.moon.activate();
    mvp = viewProj * mrot * glm::translate(glm::mat4(1.0f), glm::vec3(0.0f, 0.0f, -30.0f));
    mProgram.moon.set_uniform("uModelViewProjMatrix", mvp);
    mProgram.moon.set_uniform("uColor", aer::Vector3(1.0f, 0.0f, 0.0f));
    mScene.moon_plane.draw();
  mProgram.moon.deactivate();

glDisable(GL_CULL_FACE);

  CHECKGLERROR();
}


void Application::help() {
#define NEW_LINE  "\n" \

  fprintf(stdout, NEW_LINE
  "----------------------------------------------------------------------" NEW_LINE
/*
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
*/
  "----------------------------------------------------------------------" NEW_LINE
  "\n");

#undef NEW_LINE
}

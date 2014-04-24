// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2014 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#include "ik_demo/application.h"


Application::Application(int argc, char* argv[]) :
  aer::Application(argc, argv),
  mCamera(nullptr)
{}

Application::~Application() {
  AER_SAFE_DELETE(mCamera);
}

void Application::init() {
  /// Window
  aer::Display_t display(kDefaultHeight, kDefaultHeight);
  create_window(display, "IK Demo");

  // Enable fps control
  set_fps_control(true);
  set_fps_limit(60u);


  /// Camera
  aer::View view(aer::Vector3(0.0f, 0.0f, 10.0f),
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

  glEnable(GL_DEPTH_TEST);
  glDepthFunc(GL_LEQUAL);

  // Application settings
  aer::ShaderProxy &sp = aer::ShaderProxy::Get();
  sp.set_shader_path(DATA_DIRECTORY"/shaders/");
  sp.add_directive_token("*", "#version 420 core");

  mProgram.create();
    mProgram.add_shader(sp.get("PassThrough.VS"));
    mProgram.add_shader(sp.get("PassThrough.FS"));
  AER_CHECK(mProgram.link());


  mSphere.init(0.1f, 8u);
  setup_ikchain();

  help();
}

void Application::setup_ikchain() {
  mTarget = aer::Vector3(2.0f, 1.0f, 0.0f);

  aer::Vector3 rot_axis = aer::Vector3(0.0f, 0.0f, 1.0f);
  
  aer::BaseIKNode *node(nullptr);
  aer::BaseIKNode *root = mIKChain.set_root(aer::Vector3(0.0f), rot_axis);

  node = mIKChain.add_node(aer::Vector3(1.0f, 0.0f, 0.0f), rot_axis, 
                           aer::IK_JOINT, root);
  node = mIKChain.add_node(aer::Vector3(0.0f, -2.0f, 0.0f), rot_axis, 
                           aer::IK_END_EFFECTOR, node);


  node = mIKChain.add_node(aer::Vector3(-1.0f, 0.0f, 0.0f), rot_axis, 
                           aer::IK_JOINT, root);
  node = mIKChain.add_node(aer::Vector3(0.0f, 2.0f, 0.0f), rot_axis, 
                           aer::IK_END_EFFECTOR, node);


  mIKSolver.init(&mIKChain);
}

void Application::frame() {
  const auto &ev = aer::EventsHandler::Get();

  if (ev.key_pressed(aer::Keyboard::Escape)) {
    quit();
  }

  mCamera->update();

  aer::Vector3 targets[2] = {mTarget, mTarget};

  //for (int i=0;i<20;++i)
  mIKSolver.update(targets, 2);

  render_scene();
}

void Application::render_scene() {
  glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
  
  const aer::Matrix4x4 &viewproj = mCamera->view_projection_matrix();
  aer::Matrix4x4 mvp;

  mProgram.activate();

  aer::F32 tick = aer::GlobalClock::Get().application_time()/20.0f;
  mTarget = aer::Vector3(glm::rotate(glm::mat4(1.0f), tick, aer::Vector3(0.0f,0.0f,1.0f)) * 
                         aer::Vector4(2.0f, 1.0f, 0.0f,1.0f));
  // Render Target
  mvp = viewproj * glm::translate(glm::mat4(1.0f), mTarget);


  mProgram.set_uniform("uColor", aer::Vector3(1.0f, 0.0f, 0.0f));
  mProgram.set_uniform("uModelViewProjMatrix", mvp);
  mSphere.draw();

  // Render "joints"
  mProgram.set_uniform("uColor", aer::Vector3(0.0f, 0.0f, 1.0f));
  for (auto n = mIKChain.begin(); n != mIKChain.end(); ++n) {
    aer::Vector3 pos = (*n)->position_ws();
    mvp = viewproj * glm::translate(glm::mat4(1.0f), pos);
    mProgram.set_uniform("uModelViewProjMatrix", mvp);
    
    mSphere.draw();

    //printf("%f %f %f\n", pos.x, pos.y, pos.z);
  }

  mProgram.deactivate();

  CHECKGLERROR();
}

void Application::help() {
#define NEW_LINE  "\n" \

  fprintf(stdout, NEW_LINE
  "----------------------------------------------------------------------" NEW_LINE
  "Test the inverse kinematic solver.\n" NEW_LINE
  
  "Controls:" NEW_LINE
  "[Z-Q-S-D + mouse] or [SixAxis pad] : control the camera." NEW_LINE
  "[H] : display this help." NEW_LINE
  "[ESCAPE] : quit the application." NEW_LINE
  "----------------------------------------------------------------------" NEW_LINE
  "\n");

#undef NEW_LINE
}

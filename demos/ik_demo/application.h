// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2014 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef IK_DEMO_APPLICATION_H_
#define IK_DEMO_APPLICATION_H_

#include "aer/aer.h"
#include "aer/app/application.h"
#include "aer/device/program.h"
#include "aer/physic/ik_solver.h"
#include "aer/rendering/shape.h"
#include "aer/view/free_camera.h"

#include "ik_demo/base_ik_chain.h"


class Application : public aer::Application {
 public:
  static const aer::U32 kDefaultWidth  = 1280u;
  static const aer::U32 kDefaultHeight = 720u;
 
  Application(int argc, char* argv[]);
  ~Application();


 private:    
  void init() override;
  void setup_ikchain();

  void frame() override;
  void render_scene();

  void help();


  aer::FreeCamera *mCamera;
  aer::Program    mProgram;
  aer::SphereRaw  mSphere;

  aer::Vector3      mTarget;
  aer::BaseIKChain  mIKChain;
  aer::IKSolver     mIKSolver;
};

#endif  // IK_DEMO_APPLICATION_H_

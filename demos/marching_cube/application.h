// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef MARCHING_CUBE_APPLICATION_H_
#define MARCHING_CUBE_APPLICATION_H_

#include "aer/aer.h"
#include "aer/app/application.h"
#include "aer/view/free_camera.h"

#include "marching_cube/marching_cube_renderer.h"


/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
///
///
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
class Application : public aer::Application {
 public:
  Application(int argc, char* argv[]);
  ~Application();


 private:
  void init() override;
  void frame() override;

  void help();

  aer::FreeCamera *mCamera;
  MarchingCubeRenderer mRenderer;
};

#endif  // MARCHING_CUBE_APPLICATION_H_

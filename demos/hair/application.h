// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef HAIR_APPLICATION_H_
#define HAIR_APPLICATION_H_

#include "aer/aer.h"
#include "aer/app/application.h"
#include "aer/view/free_camera.h"
#include "hair/hair_simulation.h"

// =============================================================================

/**
 * @class Application
 */
class Application : public aer::Application {
 public:
  static const aer::U32 kDefaultWidth  = 1280u;
  static const aer::U32 kDefaultHeight = 720u;
 
  Application(int argc, char* argv[]);
  ~Application();

 private:    
  void init() override;
  void frame() override;
  void render_scene();

  void help();

  aer::FreeCamera *mCamera;
  HairSimulation mHairSim;
};

// =============================================================================

#endif  // HAIR_APPLICATION_H_

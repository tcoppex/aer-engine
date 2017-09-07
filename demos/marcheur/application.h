// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef MARCHEUR_APPLICATION_H_
#define MARCHEUR_APPLICATION_H_

#include "aer/aer.h"
#include "aer/app/application.h"
#include "aer/view/free_camera.h"
#include "aer/rendering/shape.h"

#include "marcheur/character.h"



class Application : public aer::Application {
 public:
  Application(int argc, char* argv[]);
  ~Application();


 private:
  void init() override;
  void init_textures();
  void init_shaders();
  void init_scene();

  void frame() override;
  void render_scene(const aer::Camera &camera);

  void help();


  aer::FreeCamera *mCamera;

  struct {
    Character   character;
    aer::Plane  floorPlane;
    aer::Mesh   moon_plane;
  } mScene;

  struct {
    aer::Program scene;
    aer::Program moon;
  } mProgram;
};

#endif  // MARCHEUR_APPLICATION_H_

// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AURA_APPLICATION_H_
#define AURA_APPLICATION_H_

#include "aer/aer.h"
#include "aer/app/application.h"
#include "aer/view/free_camera.h"
#include "aer/rendering/shape.h"

#include "aura/hbao_pass.h"
#include "aura/character.h"
#include "aura/skydome.h"



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
  void render_scene();
  void map_screen();

  void help();


  aer::FreeCamera *mCamera;

  struct {
    Character   character;
    SkyDome     skydome;
    aer::Plane  floorPlane;
    aer::SphereRaw  sphere;
  } mScene;

  struct {
    aer::Framebuffer FBO;
    aer::Texture2D   texRGBA;
    aer::Texture2D   texDEPTH;
  } mBufferingPass;

  struct {
    aer::Program scene;
    aer::Program mapscreen;
  } mProgram;

  struct {
    HBAOPass hbao;
  } mPass;

  aer::Texture2D *mAOTexturePtr;
};

#endif  // AURA_APPLICATION_H_

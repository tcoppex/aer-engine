// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2014 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef SHADOWMAP_APPLICATION_H_
#define SHADOWMAP_APPLICATION_H_

#include "aer/aer.h"
#include "aer/app/application.h"
#include "aer/device/program.h"
#include "aer/device/framebuffer.h"
#include "aer/rendering/shape.h"
#include "aer/view/free_camera.h"


// A simple class to test Variance shadow mapping
// There is still some visual glitch to get ride off
class Application : public aer::Application {
 public:
   Application(int argc, char* argv[]);
   ~Application();

 private:
  void init() override;
  void init_scene();
  void init_shadow();

  void frame() override;

  void render(const aer::Camera &camera);
  void draw_scene(const aer::Camera &camera, aer::Program &pgm);
  void blur_shadow();
  void render_final(const aer::Camera &camera);
  void render_shadow();

  
  aer::FreeCamera *mCamera;

  struct {
    aer::Plane      floorPlane;
    aer::Cube       cube;
    aer::SphereRaw  sphere;
    aer::Matrix4x4  cubeModel;
    aer::Matrix4x4  sphereModel;
  } mScene;

  struct {
    aer::Program shadow;  // create the shadow map
    aer::Program blur;    // blur values
    aer::Program scene;   // render the shadowmap
  } mProgram;

  struct {
    aer::Camera      projector;     // projection parameters
    aer::Framebuffer FBO;           // RenderTarget for shadowmap
    aer::Texture2D   texSHADOW;     // Raw VSM shadowmap
    aer::Texture2D   texDEPTH;      // FBO's depth buffer
    aer::Texture2D   texBLUR;       // Blurred shadowmap
    aer::Sampler     sampler;       // texBLUR sampler
  } mShadowPass;
};


#endif // SHADOWMAP_APPLICATION_H_
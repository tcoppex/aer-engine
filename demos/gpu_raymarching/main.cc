// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#include <cstdlib>

#include "aer/aer.h"
#include "aer/app/application.h"
#include "aer/device/program.h"
#include "aer/rendering/mapscreen.h"


class Application : public aer::Application {
 public:
  static const aer::U32 kWidth  = 800u;
  static const aer::U32 kHeight = 600u;

  Application(int argc, char* argv[]) :
    aer::Application(argc, argv) 
  {}

 private:
  void init() override {
    create_window(aer::Display_t(kWidth, kHeight), "GPU Raymarching");
    help();

    glDisable(GL_DEPTH_TEST);
    glDepthMask(GL_FALSE);

    auto& sp = aer::ShaderProxy::Get();
    sp.set_shader_path(DATA_DIRECTORY"/shaders/");
    sp.add_directive_token("*", "#version 420 core");

    pgm_.create();
    pgm_.add_shader(sp.get("MapScreen.VS"));
    pgm_.add_shader(sp.get("MapScreen.FS"));
    AER_CHECK(pgm_.link());

    CHECKGLERROR();
  }

  void frame() override {
    pgm_.activate();    
    aer::F32 tick = aer::GlobalClock::Get().application_time(aer::SECOND);
    pgm_.set_uniform("uTime", tick);

    auto& disp = window().display();
    pgm_.set_uniform("uResolution", aer::Vector2(disp.width, disp.height));

    aer::MapScreen::Draw();
    pgm_.deactivate();

    CHECKGLERROR();
  }

  void help() {
    fprintf(stdout, "###\nBender J. Rodriguez preparing some mischiefs..\n###\n");
  }

  aer::Program  pgm_;
};


int main(int argc, char *argv[]) {
  Application(argc, argv).run();
  return EXIT_SUCCESS;
}

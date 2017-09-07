// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_APP_APPLICATION_H_
#define AER_APP_APPLICATION_H_

#include "aer/common.h"
#include "aer/app/window.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
/// 
/// Application layer used to automatized the main
/// subsystems.
/// 
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class Application {
 public:
  Application(int argc, char* argv[]);
  virtual ~Application();

  // create a window
  Window& create_window(const Display_t &display, const char *title="");

  // launch the mainloop
  void run();

 protected:
  // limit the frame per second
  void fps_control();

  // signal to terminate the application
  void quit();


  /// Getters
  Window& window();
  U32     fps_limit()       const { return fps_limit_; }
  bool    has_fps_control() const { return bEnableFPSControl_; }

  /// Setters
  void set_fps_limit(const U32 fps) { fps_limit_ = fps; }
  void set_fps_control(bool state)  { bEnableFPSControl_ = state; }


 private:
  // callback to initialize the application
  virtual void init();

  // callback to handle events
  virtual void events();

  // callback to process one frame
  virtual void frame();


  Window* window_;                  // Window used by the application

  U32     fps_limit_;               // framerate limit for FPS control
  bool    bEnableFPSControl_;       // when true, enable FPS control
  bool    bExit_;                   // when true, the application terminate


  DISALLOW_COPY_AND_ASSIGN(Application);
};

}  // namespace aer

#endif  // AER_APP_APPLICATION_H_

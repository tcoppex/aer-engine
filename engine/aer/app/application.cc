// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#include "aer/app/application.h"

#include "aer/app/events_handler.h"
#include "aer/utils/global_clock.h"
#include "aer/utils/logger.h"
#include "aer/loader/shader_proxy.h"
#include "aer/loader/texture_2d_proxy.h"


namespace {

void DeinitializeSingletons() {
  aer::GlobalClock::Deinitialize();
  aer::Logger::Deinitialize();
  aer::EventsHandler::Deinitialize();
  aer::ShaderProxy::Deinitialize();
  aer::Texture2DProxy::Deinitialize();
}

void InitializeSingletons() {
  aer::GlobalClock::Initialize();
  aer::Logger::Initialize();
  aer::EventsHandler::Initialize();
  aer::ShaderProxy::Initialize();
  aer::Texture2DProxy::Initialize();

  atexit(DeinitializeSingletons);
}

}  // namespace

namespace aer {

Application::Application(int argc, char* argv[]) : 
  window_(nullptr),
  fps_limit_(60u),
  bEnableFPSControl_(false),
  bExit_(false) 
{}

Application::~Application() {
  AER_SAFE_DELETE(window_);
}

Window& Application::create_window(const Display_t &display, const char *title) {
  window_ = new Window(display, title);
  return *window_;
}

void Application::run() {
  InitializeSingletons();
  init();

  if (nullptr == window_) {
    Logger::Get().error("application launche with no window");
  }


  GlobalClock  &clock = GlobalClock::Get();

  /// Mainloop
  while (!bExit_) {

    // First clock update can occurs very long time after the app initialization.
    // Therefore we stabilize the dt the first few frames. XXX
    if (clock.framecount_total() < 4) {
      aer::F64 dt = 1000.0f/(fps_limit_);
      clock.stabilize_delta_time(dt);
    }

    clock.update();

    EventsHandler::Get().update(window());
    events();

    frame();

    (*window_).swap();

    if (bExit_) {
      break;
    }

    if (has_fps_control()) {
      fps_control();
    }
  }
}

// to override
void Application::init() {
}

// to override
void Application::events() {
  EventsHandler &events = EventsHandler::Get();

  if (events.has_closed()) {
    quit();
  }

  if (events.has_resized()) {
    Vector2i size = events.window_size();
    window_->resize(size.x, size.y);
    glViewport(0, 0, size.x, size.y);
  }
}

// to override
void Application::frame() {
}

void Application::fps_control() {
  const GlobalClock &gc = GlobalClock::Get();
  double fps_time = 1000.0 / static_cast<double>(fps_limit_);
  double elapsed = gc.frame_elapsed_time();

#ifdef AER_UNIX
  if (elapsed < fps_time) {
    usleep(1000.0 * (fps_time - elapsed));
  }
#else
  while (elapsed < fps_time) {
    elapsed = gc.frame_elapsed_time();
  }
#endif
}

void Application::quit() {
  bExit_ = true;
}


Window& Application::window() {
  AER_ASSERT(window_ != nullptr);
  return *window_;
}

}  // namespace aer

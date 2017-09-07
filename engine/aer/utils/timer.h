// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_UTILS_TIMER_H_
#define AER_UTILS_TIMER_H_

#include "aer/common.h"
#include "aer/utils/global_clock.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
/// [abstract] Measure time intervals
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
struct TimerInterface_t {
  virtual void Start()  = 0;
  virtual void Stop()   = 0;
  virtual F32 Elapsed() = 0;
};


/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
/// Measure time spend on the host
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
struct CPUTimer_t : TimerInterface_t {
  CPUTimer_t()
    : start_time(0.0),
      stop_time(0.0),
      bStarted(false)
  {}

  void Start() {
    start_time = static_cast<F32>(GlobalClock::Get().relative_time());
    bStarted = true;
  }

  void Stop() {
   AER_ASSERT(bStarted);
   stop_time = static_cast<F32>(GlobalClock::Get().relative_time());
   bStarted = false;
  }

  F32 Elapsed() {
    return stop_time - start_time ;
  }

  F32 start_time;
  F32 stop_time;
  bool bStarted;
};


/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
/// Measure time spend on the device
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
struct GPUTimer_t : TimerInterface_t {
  GPUTimer_t()
    : query_id(0u),
      bStarted(true)
  {}

  ~GPUTimer_t() {
    if (query_id) {
      glDeleteQueries(1u, &query_id);
    }
  }

  void Start() {
    if (!query_id) {
      glGenQueries(1u, &query_id);
    }
    glBeginQuery(GL_TIME_ELAPSED, query_id);
    bStarted = true;
  }

  void Stop() {
    AER_ASSERT(bStarted);
    glEndQuery(GL_TIME_ELAPSED);
    bStarted = false;
  }

  F32 Elapsed() {
    GLint nanosecond = 0;
    glGetQueryObjectiv(query_id, GL_QUERY_RESULT, &nanosecond);
    F32 ms = nanosecond / 1000000.0f;
    return ms;
  }

  GLuint query_id;
  bool bStarted;
};

}  // namespace aer

#endif  // AER_UTILS_GLOBAL_CLOCK_H_

// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#include "aer/utils/global_clock.h"

#ifdef AER_WINDOWS
# include <windows.h>
#else
# include <sys/time.h>
#endif


namespace aer {

GlobalClock::GlobalClock()
  : start_time_(0.0),
    delta_time_(0.0),
    frame_time_(0.0),
    last_fps_time_(0.0),
    application_time_(0.0),
    application_delta_time_(0.0),
    time_scale_(1.0),
    fps_(-1),
    framecount_(0u),
    framecount_total_(0u),
    default_unit_(MILLISECOND),
    bPaused_(true)
{
  converse_table_[NANOSECOND]   = 1.0e-9;
  converse_table_[MICROSECOND]  = 1.0e-6;
  converse_table_[MILLISECOND]  = 1.0e-3;
  converse_table_[SECOND]       = 1.0;
  converse_table_[DEFAULT]      = converse_table_[default_unit_];

  // initialize the clock
  start_time_ = absolute_time();

  resume();
}


void GlobalClock::update() {
  ++framecount_;
  ++framecount_total_;

  F64 lastFrameTime = frame_time_;
  frame_time_ = relative_time(MILLISECOND);
  delta_time_ = frame_time_ - lastFrameTime;

  if ((frame_time_ - last_fps_time_) >= 1000.0) {
    last_fps_time_ = frame_time_;
    fps_ = framecount_;
    framecount_ = 0u;
  }

  if (!is_paused()) {
    application_delta_time_ = time_scale_ * delta_time_;
    application_time_ += application_delta_time_;
  } else {
    application_delta_time_ = 0.0;
  }
}

void GlobalClock::stabilize_delta_time(const aer::F64 dt) {
  frame_time_ = relative_time(MILLISECOND) - dt;
  delta_time_ = dt;

  if (!is_paused()) {
    application_delta_time_ = time_scale_ * delta_time_;
    application_time_ += application_delta_time_;
  } else {
    application_delta_time_ = 0.0;
  }
}

bool GlobalClock::is_same_unit(TimeUnit src, TimeUnit dst) const {
  return (src == dst) || (converse_table_[src] == converse_table_[dst]);
}

F64 GlobalClock::convert_time(TimeUnit src_unit, TimeUnit dst_unit, const F64 time) const {
  const F64 scale = (is_same_unit(src_unit, dst_unit)) ? 1.0 : 
                                                         converse_table_[src_unit] / converse_table_[dst_unit];
  return scale * time;
}

F64 GlobalClock::absolute_time(TimeUnit unit) const {
  F64 global_time = 0.0;
    
#ifdef AER_WINDOWS

  LARGE_INTEGER ticksPerSecond;
  LARGE_INTEGER t;

  QueryPerformanceFrequency(&ticksPerSecond);
  QueryPerformanceCounter(&t);

  global_time = (t.QuadPart / ticksPerSecond.QuadPart) * 1000.0;

#else // AER_UNIX && AER_MACOS

  timeval t;
  gettimeofday(&t, NULL);
  global_time = t.tv_sec * 1000.0 + t.tv_usec * 0.001;

#endif

  return convert_time(MILLISECOND, unit, global_time);
}

F64 GlobalClock::relative_time(TimeUnit unit) const {
  F64 t = absolute_time() - start_time_;
  return convert_time(MILLISECOND, unit, t);
}

F64 GlobalClock::delta_time(TimeUnit unit) const {
  return convert_time(MILLISECOND, unit, delta_time_);
}

F64 GlobalClock::frame_time(TimeUnit unit) const {
  return convert_time(MILLISECOND, unit, frame_time_);
}

F64 GlobalClock::frame_elapsed_time(TimeUnit unit) const {
  return relative_time(unit) - frame_time(unit);
}

F64 GlobalClock::application_time(TimeUnit unit) const {
  return convert_time(MILLISECOND, unit, application_time_);
}

F64 GlobalClock::application_delta_time(TimeUnit unit) const {
  return convert_time(MILLISECOND, unit, application_delta_time_);
}

void GlobalClock::set_default_unit(TimeUnit unit) {
  default_unit_ = unit;
  converse_table_[DEFAULT] = converse_table_[default_unit_];
}

} // namespace aer

// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_UTILS_GLOBAL_CLOCK_H_
#define AER_UTILS_GLOBAL_CLOCK_H_

#include "aer/common.h"
#include "aer/utils/singleton.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
/// Time unit used by the clock
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
enum TimeUnit {
  NANOSECOND,
  MICROSECOND,
  MILLISECOND,
  SECOND,
  DEFAULT,
  kNumTimeUnit
};

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// The global clock is used across an application to
/// measure time.
/// Time is stored in millisecond.
///
/// TODO : set more than one application timeline
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class GlobalClock : public Singleton<GlobalClock>
{
 public:
  GlobalClock();

  // pause the application timeline
  void pause() { bPaused_ = true; }

  // resume the application timeline
  void resume() { bPaused_ = false; }

  // Update per-frame time values
  void update();

  // Convert time from one unit to another
  F64 convert_time(TimeUnit src_unit, TimeUnit dst_unit, const F64 time) const;


  // time from the system
  F64    absolute_time(TimeUnit unit = DEFAULT) const;
  
  // time from the start of the clock
  F64    relative_time(TimeUnit unit = DEFAULT) const;
  
  // time from the last frame
  F64       delta_time(TimeUnit unit = DEFAULT) const;
  
  // relative time at the beginning of the frame
  F64       frame_time(TimeUnit unit = DEFAULT) const;

  // time from the beginning of the frame (relative_time - frame_time)
  F64 frame_elapsed_time(TimeUnit unit = DEFAULT) const;

  // time of the application
  F64 application_time(TimeUnit unit = DEFAULT) const;

  // delta time of the application
  F64 application_delta_time(TimeUnit unit = DEFAULT) const;

  F64       time_scale() const { return time_scale_; }
  U32              fps() const { return fps_; }
  U32 framecount_total() const { return framecount_total_; }
  bool       is_paused() const { return bPaused_; }

  // Defines a scale by which application time is updated
  void set_time_scale(F64 scale);

  // specify the default time unit to return
  void set_default_unit(TimeUnit unit);


 private:
  bool is_same_unit(TimeUnit src, TimeUnit dst) const;

  F64 start_time_;              //  global time at the beginning of the clock
  F64 delta_time_;              //  duration of last frame
  F64 frame_time_;              //  relative time at the beginning of the frame
  F64 last_fps_time_;           //  last frame relative time (to count fps)

  F64 application_time_;        //  timeline of the application
  F64 application_delta_time_;  //  deltatime of the application
  F64 time_scale_;              //  application time scale

  U32 fps_;                     //  last second frame count
  U32 framecount_;              //  current second frame count
  U32 framecount_total_;        //  total frame elapsed from the start

  bool bPaused_;                //  when true, do not update application time

  TimeUnit default_unit_;       //  default unit used to retrieve time
  F64 converse_table_[kNumTimeUnit];


  friend class Singleton<GlobalClock>;
};

}  // namespace aer

#endif  // AER_UTILS_GLOBAL_CLOCK_H_

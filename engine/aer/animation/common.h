// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AER_ANIMATION_COMMON_H_
#define AER_ANIMATION_COMMON_H_

#include <vector>
#include "aer/common.h"

namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// Common datastructures used for animations
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
/// Pose transformation for a joint
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
struct JointPose_t {
  Quaternion   qRotation;
  Vector3      vTranslation;
  F32          fScale;
};

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
/// Set of transformations for an animation 
/// sample
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
struct AnimationSample_t {
  std::vector<JointPose_t> joints;
};


/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
/// Basic animation action
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
struct Action_t {
  virtual F32 duration() const = 0;

  char *pName  = nullptr;
  bool bLoop   = false;
};

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
/// Set of samples defining a skeleton 
/// animation clip
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
struct AnimationClip_t : Action_t {
  ~AnimationClip_t() {
    AER_SAFE_DELETEV(samples);
  }

  F32 duration() const override {
    return numframes / framerate;
  }

  AnimationSample_t   *samples  = nullptr;
  U32   numframes = 0u;
  F32   framerate = 0u;
};

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
/// Set of BlendShape making an expression
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
struct Expression_t : Action_t {
  virtual F32 duration() const override {
    return clip_duration;
  }

  F32 clip_duration;
  bool     bManualBypass;

  std::vector<U32> indices;
};


/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
/// Animation being played
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
struct SequenceClip_t {
  Action_t *action_ptr = nullptr;

  F32  global_start  = 0.0f;
  F32  rate          = 1.0f;
  U32  numloops      = 0u;
  F32  weight        = 1.0f;
  bool bEnable       = false;
  bool bPingPong     = false;
  //bool bFrozen   = false;


  /// Compute the localtime for the sequence, looping the animation if needed.
  /// Return false if the sequence finished.
  bool compute_localtime(const F32 global_time, F32& local_time) const {
    const Action_t &action = *(action_ptr); //
    const F32 clip_duration = action.duration();

    local_time  = global_time - global_start;
    local_time *= abs(rate);

    // clamp the local time if the action loops a finite number of time
    if (!action.bLoop || (action.bLoop && numloops > 0u)) {
      U32 total_loops = (action.bLoop) ? numloops : 1u;
      F32 finish_time = total_loops * clip_duration;

      if (local_time > finish_time) {
        return false;
      }

      local_time = glm::clamp(local_time, 0.0f, finish_time);
    }
    
    // loop the action
    U32 loop_id = 0u;
    if (action.bLoop) {
      loop_id    = local_time / clip_duration;
      local_time = fmod(local_time, clip_duration);
    }

    // Handles reverse playback with ping-pong
    bool bReverse = (rate < 0.0f);
         bReverse = bReverse != (bPingPong && ((loop_id & 1) == bReverse));
    if (bReverse) {
      local_time = clip_duration - local_time;
    }

    /*
    // Smooth-in / Smooth-out ping-pong loops
    if (bPingPong) {
      F32 clip_phase = local_time / clip_duration;
      clip_phase = glm::smoothstep(0.0f, 1.0f, clip_phase);
      local_time = clip_phase * clip_duration;
    }
    */

    return true;
  }

  /// Return the phase of the animation, given the local_time
  F32 phase(const F32 local_time) const {
    return local_time / action_ptr->duration();
  }
};


/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
/// Set of animations being played
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
//std::unordered_set<SequenceClip_t> Sequence_t;
typedef std::vector<SequenceClip_t> Sequence_t;
typedef Sequence_t::iterator        SequenceIterator_t;


}  // namespace aer

#endif  // AER_ANIMATION_COMMON_H_

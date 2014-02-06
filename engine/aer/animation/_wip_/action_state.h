// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AER_ANIMATION_ACTION_STATE_H_
#define AER_ANIMATION_ACTION_STATE_H_

#include <string>
#include <unordered_map>

#include "aer/common.h"
#include "aer/animation/blend_tree.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
///  State of an Action State Machine.
///  Holds available transitions for the state and
///  clips animation with blending specifications.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class ActionState {
 public:
  enum TransitionType {
    IMMEDIATE,
    CROSS_FADED,                    // smooth, non synchronized, cross-fade.
    kNumTransitionType
  };

  struct Transition {
    TransitionType type;            // IMMEDIATE or CROSS_FADED
    U32  src;                  // source state index
    U32  dst;                  // destinaton state index
    //F32  window_start;         // start of the availability window
    //F32  window_end;           // end of the availability window
    
    // [cross-fade only]
    F32  duration;             // duration of the transition
    bool      bSmoothed;            // smooth or flat LERP
  };


  /// Update the state
  //void update(const F32 dt); 

  /// Return true if the transition exist, false otherwise
  bool is_accessible(const std::string &name);
  
  /// Return a pointer to the transition if it exists, NULL otherwise
  Transition* transition(const std::string &name);

  // Return the name of the state
  const std::string& name() {
    return mName;
  }
  
  // Return the blend tree of the state
  BlendTree& blend_tree() {
    return mBlendTree;
  }


 private:
  typedef std::unordered_map<std::string, Transition> TransitionMap_t;
  typedef TransitionMap_t::iterator                   TransitionMapIterator_t;


  std::string     mName;
  TransitionMap_t mTransitions;
  BlendTree       mBlendTree;
};

}  // namespace aer

#endif  // AER_ANIMATION_ACTION_STATE_H_

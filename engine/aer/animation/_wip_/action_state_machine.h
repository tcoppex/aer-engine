// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AER_ANIMATION_ACTION_STATE_MACHINE_H_
#define AER_ANIMATION_ACTION_STATE_MACHINE_H_

#include <string>
#include <vector>

#include "aer/common.h"
#include "aer/animation/_wip_/action_state.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// Handle the set of animation states a model is allowed 
/// to be in.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class ActionStateMachine {
 public:
  /// Independant user of the machine use this to keep track of their
  /// current state.
  struct StateInfo {
    U32 id;                  // id of the current state
    F32 start_time;          // time tag

    // Pointer to the previously taken transition. 
    // (contains information on crossfading)
    ActionState::Transition *pTakenTransition;
  };

  /// Load machine states from a configuration file
  bool load(const std::string filename);

  /// Change state by using one of its transition
  bool access(const std::string &tname, StateInfo &state_info);

  /// Return a state from its index
  ActionState& state(const U32 index);

  /// Retrieve a state id given its name, return true if it succeed, false otherwise.
  /// Useful to check easily on event functions which events to trigger
  /// depending on the current state.
  bool stateid_from_name(const std::string &name, U32 &state_id);

 private:
  std::vector<ActionState> mStates;
};

}  // namespace aer

#endif  // AER_ANIMATION_ACTION_STATE_MACHINE_H_

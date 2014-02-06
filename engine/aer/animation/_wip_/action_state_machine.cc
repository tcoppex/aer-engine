// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------


#include "aer/animation/_wip_/action_state_machine.h"


namespace aer {

bool ActionStateMachine::load(const std::string filename) {
  AER_WARNING("not implemented yet");
  return false;
}

bool ActionStateMachine::access(const std::string &tname,
                                StateInfo &state_info) {
  ActionState &action_state = state(state_info.id);

  ActionState::Transition *pTransition = action_state.transition(tname);
  if (nullptr == pTransition) {
    return false;
  }

  /// Update state info
  state_info.id = pTransition->dst;
  //state_info.startTime = GlobalTimer::GetRelativeTime();
  state_info.pTakenTransition = pTransition;

  return true;
}

ActionState& ActionStateMachine::state(const U32 index) {
  AER_ASSERT(index < mStates.size());
  return mStates[index];
}

bool ActionStateMachine::stateid_from_name(const std::string &name, 
                                          U32 &state_id) {
  for (U32 i = 0u; i < mStates.size(); ++i) {
    if (name == mStates[i].name()) {
      state_id = i;
      return true;
    }
  }
  return false;
}

}  // namespace aer

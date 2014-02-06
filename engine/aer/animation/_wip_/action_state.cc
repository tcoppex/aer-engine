// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#include "aer/animation/_wip_/action_state.h"


namespace aer {

bool ActionState::is_accessible(const std::string &name) {
  return nullptr != transition(name);
}

ActionState::Transition* ActionState::transition(const std::string &name) {
  TransitionMapIterator_t it = mTransitions.find(name);
  return (it == mTransitions.end()) ? nullptr : &(it->second);;
}

}  // namespace aer

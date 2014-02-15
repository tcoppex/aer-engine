// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#include "aer/physic/ik_chain.h"


namespace aer {

void IKChain::clear() {
  AER_CHECK("TODO" && 0);
}

void IKChain::insert_node(IKNode* node) {
  AER_ASSERT(nullptr != node);

  chain_.push_front(node);

  // Update counters
  if (node->is_joint()) {
    ++num_joints_;
  }
  if (node->is_end_effector()) {
    ++num_end_effector_;
  }
  ++num_nodes_;
}

void IKChain::update() {
  // TODO : Check performances and validity !

  for (auto& node : chain_) {
    node->update();
  }
}

}  // namespace aer

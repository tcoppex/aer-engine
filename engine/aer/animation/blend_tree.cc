// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------


#include "aer/animation/blend_tree.h"

namespace aer {

BlendTree::BlendTree() :
  mRoot(nullptr)
{}

BlendTree::~BlendTree() {
  for (auto &node : mNodeMap) {
    AER_SAFE_DELETE(node.second);
  }

  for (auto &leave : mLeaveMap) {
    AER_SAFE_DELETE(leave.second);
  }
}

void BlendTree::evaluate(const F32 factor, Sequence_t& sequence) {
  AER_ASSERT(nullptr != mRoot);

  // Recursively compute blend weight for each clips
  mRoot->compute_weight(factor);

  // Assign weight to active clips (COSTLY !!)
  for (auto &leave : mLeaveMap) {
    for (auto &sc : sequence) {
      if (leave.first == sc.action_ptr->pName) {
        sc.weight = leave.second->weight();
      }
    }
  }
}

void BlendTree::activate_leaves(bool bEnable, Sequence_t& sequence) {
  AER_ASSERT(nullptr != mRoot);
  
  AER_WARNING("BlendTree's evaluate & active_leaves are costly !");

  for (auto &leave : mLeaveMap) {
    bool bFound = false;
    for (auto &sc : sequence) {
      if (leave.first == sc.action_ptr->pName) {
        sc.bEnable = bEnable;
        bFound = true;
      }
    }
    if (!bFound) {
      AER_WARNING(leave.first + " was not found in the blend tree");
    }
  }
}

BlendNode* BlendTree::add_node(const std::string& name, BlendNode *node) {
  mNodeMap[name] = node;
  mRoot = node;
  return node;
}

LeaveNode* BlendTree::add_leave(const std::string& name) {
  LeaveNode *leave = new LeaveNode();
  mLeaveMap[name] = leave;
  if (nullptr == mRoot) {
    mRoot = leave;
  }
  return leave;
}

}  // namespace aer

// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2014 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AER_PHYSIC_BASE_IK_CHAIN_H_
#define AER_PHYSIC_BASE_IK_CHAIN_H_

#include "aer/common.h"
#include "aer/physic/ik_chain.h"
#include "ik_demo/base_ik_node.h"//#include "aer/physic/base_ik_node.h"

namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
///
/// Specialization of IKChain with internalized coordinates
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
class BaseIKChain : public IKChain {
 public:
  BaseIKNode* set_root(const Vector3& pos_rel, const Vector3& rot_rel);

  BaseIKNode* add_node(const Vector3& pos_rel, const Vector3& rot_rel,
                       IKType_t type, BaseIKNode* parent);

 private:
  //void insert_node(IKNode* node);
};

}  // namespace aer

#endif  // AER_PHYSIC_BASE_IK_CHAIN_H_

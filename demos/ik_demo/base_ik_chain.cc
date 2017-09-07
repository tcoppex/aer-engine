// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2014 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#include "ik_demo/base_ik_chain.h" //#include "aer/physic/base_ik_chain.h"


namespace aer {

BaseIKNode* BaseIKChain::set_root(const Vector3& pos_rel,
                                  const Vector3& rot_rel)
{
  BaseIKNode *node = new BaseIKNode(pos_rel, rot_rel, IK_JOINT, nullptr, 0);
  insert_node(node);

  node->set_position_ws(pos_rel);
  node->set_rotation_ws(rot_rel);

  return node;
}

BaseIKNode* BaseIKChain::add_node(const Vector3& pos_rel, 
                                  const Vector3& rot_rel,
                                  IKType_t type,
                                  BaseIKNode* parent)
{
  U32 type_id = (IK_JOINT & type) ? num_joints() : num_end_effector();

  Vector3 r = pos_rel - parent->position_rel(); //
  BaseIKNode *node = new BaseIKNode(r, rot_rel, type, parent, type_id);
  insert_node(node);

  return node;
}

}  // namespace aer

// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2014 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#include "aer/physic/ik_node.h"

#include "aer/common.h"
#include "glm/gtc/matrix_transform.hpp"


namespace aer {

IKNode::IKNode(IKType_t type, IKNode *parent, U32 type_id) :
  type_(type),
  parent_(parent),
  type_id_(type_id)
{}

void IKNode::update() {
/// recompute GLOBAL POSES from local pose & theta
  Vector3 pos_ws = position_rel();
  Vector3 rot_ws = rotation_rel();

  for (IKNode *n = parent(); nullptr != n; n = n->parent()) {
    F32 fTheta = static_cast<F32>(n->theta());
    Matrix4x4 Mrot = glm::rotate(Matrix4x4(1.0f), fTheta, n->rotation_rel());

    pos_ws = Vector3(Mrot * Vector4(pos_ws, 1.0f));
    pos_ws += n->position_rel();
    rot_ws = Vector3(Mrot * Vector4(rot_ws, 1.0f));
  }

  set_position_ws(pos_ws);
  set_rotation_ws(rot_ws);
}

}  // namespace aer

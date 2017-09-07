// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2014 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AER_PHYSIC_BASE_IK_NODE_H_
#define AER_PHYSIC_BASE_IK_NODE_H_

#include "aer/common.h"
#include "aer/physic/ik_node.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
///
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
class BaseIKNode : public IKNode {
 public:  
  BaseIKNode(const Vector3& pos_rel, const Vector3& rot_rel, 
             IKType_t type, IKNode *parent, U32 type_id) : 
    IKNode(type, parent, type_id),
    position_rel_(pos_rel),
    rotation_rel_(rot_rel)
  {
    theta_min_ = glm::radians(360.0);
    theta_max_ = glm::radians(360.0);
    // world space position are updated when inserted in the chain
  }

  /// Position
  virtual const Vector3& position_rel() override { return position_rel_; }
  virtual const Vector3& position_ws()  override { return position_ws_;  }

  /// Rotation axis
  virtual const Vector3& rotation_rel() override { return rotation_rel_; }
  virtual const Vector3& rotation_ws()  override { return rotation_ws_;  }
  
  /// Rotation constraints
  virtual F64 theta_min() const override { return theta_min_; }
  virtual F64 theta_max() const override { return theta_max_; }


 private:
  /// Setters for updates
  virtual void set_position_ws(const Vector3& x) override { position_ws_ = x; }
  virtual void set_rotation_ws(const Vector3& x) override { rotation_ws_ = x; }


  Vector3 position_rel_;
  Vector3 position_ws_;

  Vector3 rotation_rel_;
  Vector3 rotation_ws_;

  F64     theta_min_;
  F64     theta_max_;


  friend class BaseIKChain;
};

}  // namespace aer

#endif  // AER_PHYSIC_BASE_IK_NODE_H_

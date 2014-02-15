// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2014 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AER_PHYSIC_IK_NODE_H_
#define AER_PHYSIC_IK_NODE_H_

#include "aer/common.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~
/// Type of IK node
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~
enum IKType_t {
  IK_JOINT        = 1 << 0,
  IK_END_EFFECTOR = 1 << 1
};

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
///
/// 
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
class IKNode {
 public:
  IKNode(IKType_t type, IKNode *parent, U32 type_id);

  /// Update the node's world-space coordinates
  void update();


  IKNode* parent() { return parent_; }

  U32 type_id() const { return type_id_; }         // XXX what if type is both ?
  
  F64 theta() const { return theta_; }
  
  bool is_joint() const { return type_ & IK_JOINT; }  
  bool is_end_effector() const { return type_ & IK_END_EFFECTOR; }

  /// Position
  virtual const Vector3& position_rel() = 0;
  virtual const Vector3& position_ws()  = 0;

  /// Rotation axis (vec3 or quaternion ?)
  virtual const Vector3& rotation_rel() = 0;
  virtual const Vector3& rotation_ws()  = 0;
  
  /// Rotation constraints
  virtual F64 theta_min() const = 0;
  virtual F64 theta_max() const = 0;

  /// Set new theta value
  void set_theta(F64 t) { theta_ = t; }
  void inc_theta(F64 t) { set_theta(theta_ + t); }


 protected:
  /// Setters for updates
  virtual void set_position_ws(const Vector3& x) = 0;
  virtual void set_rotation_ws(const Vector3& x) = 0;


 private:
  IKType_t type_;         // specify the type of IKNode (JOINT or END_EFFECTOR)

  IKNode *parent_;        // node's parent

  U32 type_id_;           // node's index depending on its type
  F64 theta_;             // varying rotation angle
};

}  // namespace aer

#endif  // AER_PHYSIC_IK_NODE_H_

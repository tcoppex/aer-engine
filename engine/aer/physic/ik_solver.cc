// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#include "aer/physic/ik_solver.h"


namespace aer {

// Damping values have to be set manually (BAD bad bad bad BAAAAAAAD)
const F64 IKSolver::kDefaultDampingLambda_DLS = 1.1;
const F64 IKSolver::kMaxAngle_DLS             = glm::radians(45.0);


void IKSolver::init(IKChain *pIKChain) {
  ik_chain_ = pIKChain;

  const U32 nEffector  = ik_chain_->num_end_effector();
  const U32 nJoint     = ik_chain_->num_joints();
  const U32 nRow       = 3u * nEffector;
  const U32 nCol       = nJoint;


  /// Jacobians
  jacobian_ee_  = arma::zeros(nRow, nCol);
  jacobian_ptr_ = &jacobian_ee_;    // Set the active Jacobian

  /// Singular Value Decomposition
  svd_.U = arma::mat(nRow, nRow);
  //svd_.V = arma::mat(nCol, nCol)
  //svd_.s = arma::vec(glm::min(nRow, nCol));

  /// Deltas
  delta_.pos    = arma::vec(nRow);
  delta_.t      = arma::vec(nRow);
  delta_.theta  = arma::vec(nCol);

  /// Reset parameters
  dls_.damping_lambda     = kDefaultDampingLambda_DLS;
  dls_.damping_lambda_sqr = dls_.damping_lambda * dls_.damping_lambda;
}

void IKSolver::update(const Vector3 targets[], U32 numtarget) {
  //AER_ASSERT(numtarget == ik_chain_->num_end_effector());

  // Setup Jacobian and deltaS vectors
  setup_jacobian(targets, numtarget);
  // Calculate the change in theta values (Damped Least Squares method)
  calculate_delta_thetas_DLS();
  // Update the rotation angles
  update_thetas();
}


void IKSolver::setup_jacobian(const Vector3 targets[], U32 numtarget) {
  Vector3 new_coords;

  for (auto n = ik_chain_->begin(); n != ik_chain_->end(); ++n) {
    // We're looking for end effector only
    if (false == (*n)->is_end_effector()) {
      continue;
    }

    // retrieve the end effector's target
    U32 ee_id = (*n)->type_id();
    U32 target_id = glm::min(ee_id, numtarget);
    const Vector3& target   = targets[target_id];
    const Vector3& n_pos_ws = (*n)->position_ws();

    // update the target position from its end effector
    new_coords = target - n_pos_ws;
    delta_.pos(3*ee_id + 0) = new_coords.x;
    delta_.pos(3*ee_id + 1) = new_coords.y;
    delta_.pos(3*ee_id + 2) = new_coords.z;

    // Update ancestors' (joints) entries in the Jacobian [reverse traversial]
    for (IKNode *m = (*n)->parent(); nullptr != m; m = m->parent()) {
      U32 joint_id = m->type_id();
      //AER_CHECK((0<=ee_id && ee_id<nEffector) && (0<=joint_id && joint_id<nJoint)); 

      new_coords = m->position_ws() - n_pos_ws;
      new_coords = glm::cross(new_coords, m->rotation_ws());

      jacobian_ee_(3*ee_id + 0, joint_id) = new_coords.x;
      jacobian_ee_(3*ee_id + 1, joint_id) = new_coords.y;
      jacobian_ee_(3*ee_id + 2, joint_id) = new_coords.z;
    }
  }
}

namespace {
F64 MaxAbsTheta(const arma::vec &theta) {
  return glm::max(theta.max(), glm::abs(theta.min()));
}
}  // namespace

void IKSolver::calculate_delta_thetas_DLS() {
  const arma::mat& J  = active_jacobian();
  const arma::mat& Jt = J.t();

  svd_.U = J * Jt;
  svd_.U.diag() += dls_.damping_lambda_sqr;
  
  // Traditional DLS
  delta_.t     = arma::solve(svd_.U, delta_.pos);
  delta_.theta = Jt * delta_.t;

  // Scale back to not exceed maximum angle changes
  F64 max_change = MaxAbsTheta(delta_.theta);
  if (max_change > kMaxAngle_DLS) {
    delta_.theta *= kMaxAngle_DLS / max_change;
  }
}

void IKSolver::update_thetas() {
  // Update the joints angles
  for (auto n = ik_chain_->begin(); n != ik_chain_->end(); ++n) {
    if ((*n)->is_joint()) {
      U32 joint_id = (*n)->type_id();
      (*n)->inc_theta(delta_.theta(joint_id));
    }
  }

  // Update the positions and rotation axes of every nodes
  ik_chain_->update();
}

}  // namespace aer

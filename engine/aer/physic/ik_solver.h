// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2014 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AER_PHYSIC_IK_SOLVER_H_
#define AER_PHYSIC_IK_SOLVER_H_

#include <array>

// linear algebra library
#include <armadillo>

#include "aer/common.h"
#include "aer/physic/ik_chain.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~  
///
/// Inverse Kinematic solver using the Damped Least Squares algorithm.
///
/// The solver compute new local poses for joints in an ikChain based on their 
/// global poses and on their EndEffector + Target pairs.
///
///
/// Reference :
/// "Introduction to Inverse Kinematics with Jacobian Transpose, Pseudoinverse 
///  and Damped Least Squares methods" - Samuel R. Buss
///
/// TODO : - implements up to 3 DoF (just 1 at the moment)
///        - use quaternion instead of a rotation axis, 
///              (it will solve the previous problem too)
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ 
class IKSolver {
 public:
  /// Initialize the solver for the given ik chain
  void init(IKChain *pIKChain);

  /// Resolve the ik system for the given targets
  /// If there is more end effector than targets, remaining effectors used
  /// the last targets.
  void update(const Vector3 targets[], U32 numtarget);


 private:
  void setup_jacobian(const Vector3 targets[], U32 numtarget);
  void calculate_delta_thetas_DLS();
  void update_thetas();

  const arma::mat& active_jacobian() {
    return *jacobian_ptr_;
  }


  IKChain *ik_chain_;

  /// Jacobians
  //arma::mat jacobian_target_;   //  targets based Jacobian [not implemented]
  arma::mat jacobian_ee_;         //  end-effector based Jacobian
  arma::mat *jacobian_ptr_;       //  active jacobian


  /// Singular Value Decomposition [J = U * diag(s) * transpose(V)]
  /// [intended to be used in later version]
  struct {
    arma::mat U;
    //arma::mat V;        //  not used by DLS
    //arma::vec s;        //  not used by DLS
  } svd_;


  /// Updated coordinates
  struct {
    arma::vec pos;      // vectors from end-effector to target
    arma::vec t;        // ~ Linearized change in end-effector positions based on theta
    arma::vec theta;    // change in joint angles
  } delta_;  


  /// Damped Least Squares parameters
  static const double kDefaultDampingLambda_DLS;
  static const double kMaxAngle_DLS;
  struct {
    double damping_lambda;
    double damping_lambda_sqr;
  } dls_;
};

}  // namespace aer

#endif  // AER_PHYSIC_IK_SOLVER_H_

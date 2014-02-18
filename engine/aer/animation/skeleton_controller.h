// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AER_ANIMATION_SKELETON_CONTROLLER_H_
#define AER_ANIMATION_SKELETON_CONTROLLER_H_

#include <string>
#include <vector>

#include "aer/common.h"
#include "aer/device/texture_buffer.h"
#include "aer/animation/common.h"
#include "aer/animation/skeleton.h"
#include "aer/animation/blend_tree.h"
#include "aer/loader/skeleton_proxy.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// Handle the skeleton animation pipeline
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class SkeletonController {
 public:
  /// Constants used to defines shared data size [could needs more].
  static const U32 kMaxJointsPerSkeleton = 256u;  //
  static const U32 kMaxClipsPerSkeleton  = 1024u; //


  SkeletonController() = default;

  /// Set the skeleton with its animations and setup per-instance datas
  bool init(const std::string &skeleton_ref);

  /// Launch the skeletal animation pipeline
  void update();

  /// Return the buffer of skinning matrices
  const Matrix3x4* skinning_matrices() const {
    return mSkinningMatrices.data();
  }

  /// Bind the texture containing skinning datas
  void bind_skinning_texture(I32 texture_unit) const {
    return sShared.skinning.texture.bind(texture_unit);
  }

  /// [temp] Use to define the blend tree externally
  BlendTree& blend_tree() {
    return mBlendTree;
  }


 private:
  typedef std::vector<AnimationSample_t> SampleBuffer_t;

  /// Buffers shared application-wised by skeleton controllers
  struct SharedData_t {
    void init(U32);

    SampleBuffer_t  samples;
    TBO_t           skinning;
    bool            bInit = false;
  };

  static SharedData_t sShared;
  
  /// Proxy to load skeleton
  static SkeletonProxy sSkeletonProxy;


  /// Compute the static pose of each contributing clips.
  /// Return the number of active clips.
  U32 compute_poses();

  /// Apply a normalized blending on previously sampled clips,
  /// and store the final local pose to mLocalPose.
  void blend_poses(U32 active_count);

  /// Generate global pose matrices (eg. for post-processing)
  void generate_global_pose_matrices();

  /// Generate final transformation datas for skinning
  void generate_skinning_datas();


  /// Skeleton reference
  Skeleton *mSkeleton = nullptr;

  /// Inputs
  Sequence_t mSequence;
  BlendTree mBlendTree; //

  /// Outputs
  AnimationSample_t      mLocalPose;
  std::vector<Matrix4x4> mGlobalPoseMatrices;   // should be [4x3]
  
  std::vector<Matrix3x4> mSkinningMatrices;
  std::vector<DualQuaternion> mDQuaternions;    // TODO: set as buffer of float4
  bool bUseDQBS_ = true;


  DISALLOW_COPY_AND_ASSIGN(SkeletonController);
};

}  // namespace aer

#endif  // AER_ANIMATION_SKELETON_CONTROLLER_H_

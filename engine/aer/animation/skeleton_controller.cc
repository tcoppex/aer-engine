// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#include "aer/animation/skeleton_controller.h"

#include "aer/loader/skma.h"
#include "glm/gtc/matrix_transform.hpp"
#include "glm/gtx/quaternion.hpp"




namespace aer {

/// Static fields
SkeletonController::SharedData_t SkeletonController::sShared;
SkeletonProxy SkeletonController::sSkeletonProxy;


void SkeletonController::SharedData_t::init(U32 tbo_elemsize) {
  if (bInit) {
    return;
  }

  /// Samples buffer
  samples.resize(kMaxClipsPerSkeleton);
  for (auto &sample : samples) {
    sample.joints.resize(kMaxJointsPerSkeleton);
  }

  /// Texture buffer holding skinning datas [matrices or dual-quaternions]
  skinning.buffer.generate();
  skinning.buffer.bind(GL_TEXTURE_BUFFER);
  U32 bytesize = kMaxJointsPerSkeleton * tbo_elemsize;
  skinning.buffer.allocate(bytesize, GL_STREAM_DRAW);
  skinning.buffer.unbind();

  skinning.texture.generate();
  skinning.texture.bind();
  skinning.texture.set_buffer(GL_RGBA32F, skinning.buffer);
  skinning.texture.unbind();
  CHECKGLERROR();

  bInit = true;
}

//-----------------------------------------------------------------------------

namespace {

/// Apply a linear interpolation on two samples
void LerpSamples(const AnimationSample_t &s1,
                 const AnimationSample_t &s2,
                 const F32 factor,
                 AnimationSample_t &dst_sample);

/// Generate a sample for sequence_clip at global_time
bool ComputePose(const F32 global_time,
                 SequenceClip_t& sequence_clip,
                 AnimationSample_t& dst_sample);

}  // namespace


bool SkeletonController::init(const std::string &skeleton_ref) {
  mSkeleton = sSkeletonProxy.get(skeleton_ref);

  if (nullptr == mSkeleton) {
    return false;
  }

  //------ [temp]
  /// Generates SequenceClips from the skeleton's animation clips
  mSkeleton->generate_sequence(mSequence); //
  //------

  /// Outputs
  U32 njoints = mSkeleton->numjoints();
  mLocalPose.joints.resize(njoints);
  mGlobalPoseMatrices.resize(njoints);
  mSkinningMatrices.resize(njoints); //
  mDQuaternions.resize(njoints); //

  // Shared data
  AER_ASSERT(mSkeleton->numclips() <= kMaxClipsPerSkeleton);
  AER_ASSERT(mSkeleton->numjoints() <= kMaxJointsPerSkeleton);
  U32 tbo_elemsize = glm::max(sizeof(mSkinningMatrices[0]),sizeof(mDQuaternions[0]));
  sShared.init(tbo_elemsize);

  return true;
}

void SkeletonController::update() {
  /// Activates each sequence's leaves on the blend tree
  /// [TODO: set once by the Action State Machine each time a state is entered]
  mBlendTree.activate_leaves(true, mSequence); //

  /// 1) Compute weight for active leaves
  mBlendTree.evaluate(1.0f, mSequence);


  /// 2) Compute the static pose of each contributing clips
  U32 active_count = compute_poses();
  if (active_count == 0u) {
    AER_WARNING("no animation clips provided");
    return;
  }

  /// 3) Blend between poses to get a unique local pose
  blend_poses(active_count);

  /// 4) Generate the global pose matrices
  generate_global_pose_matrices();

  /// 5) Generate the final skinning datas
  generate_skinning_datas();
}

U32 SkeletonController::compute_poses() {
  F32 global_time = GlobalClock::Get().application_time(SECOND);

  U32 active_count = 0u;
  for (auto &sc : mSequence) {
    if (sc.bEnable && ComputePose(global_time, sc, sShared.samples[active_count])) {
      ++active_count;
    }
  }

  return active_count;
}

void SkeletonController::blend_poses(U32 active_count) {

  SampleBuffer_t &samples = sShared.samples;

  const U32 kNumJoints = mLocalPose.joints.size();

  /// Bypass the weighting if their is only one action
  if (active_count == 1u) {
    std::copy(samples[0u].joints.begin(), 
              samples[0u].joints.begin() + kNumJoints,
              mLocalPose.joints.begin());
    return;
  }

  /// Compute local poses by blending each contributing samples by the factor
  /// previously calculate by the blendtree. Supposed blending associativity.
  /// (ie. flat weighted average)

  /// 1) Calculate the total weight for normalization
  F32 sum_weights = 0.0f;
# pragma omp parallel for reduction(+:sum_weights) schedule(static) num_threads(4)
  for (const auto &sc : mSequence) {
    if (sc.bEnable) {
      sum_weights += sc.weight;
    }
  }
  sum_weights = (sum_weights == 0.0f) ? 1.0f : sum_weights;


  /// 2) Copy the first weighted action as base
  SequenceIterator_t it = mSequence.begin();
  F32 w = it->weight / sum_weights;
# pragma omp parallel for schedule(static) num_threads(4)
  for (U32 i = 0u; i < kNumJoints; ++i) {
    const auto &src = samples[0u].joints[i];
          auto &dst = mLocalPose.joints[i];
    
    dst.qRotation    = w * src.qRotation;
    dst.vTranslation = w * src.vTranslation;
    dst.fScale       = w * src.fScale;
  }


  /// 3) Blend the rest
  U32 sid = 1u;
  for (it = ++it; sid < active_count; ++it, ++sid) {
    w = it->weight / sum_weights;
    AnimationSample_t& sample = samples[sid];

    // Cope with antipodality by checking quaternion neighbourhood
    F32 sign_q = glm::sign(glm::dot(sample.joints[0].qRotation, 
                                    sample.joints[kNumJoints-1u].qRotation));
    F32 w_q    = sign_q * w;

    // [could swap the 2 loops and do 3 reduces operations instead]
#   pragma omp parallel for schedule(static) num_threads(4)
    for (U32 i = 0u; i < kNumJoints; ++i) {
      const auto &src = sample.joints[i];
            auto &dst = mLocalPose.joints[i];

      dst.qRotation    += w_q * src.qRotation;
      dst.vTranslation += w   * src.vTranslation;
      dst.fScale       += w   * src.fScale;
    }

    // Normalize quaternion lerping
    for (auto &joint : mLocalPose.joints) {
      joint.qRotation /= glm::length(joint.qRotation);
    }
  }
}

void SkeletonController::generate_global_pose_matrices() {
  const I32 *parent_ids = mSkeleton->parent_ids();

  for (U32 i = 0u; i < mGlobalPoseMatrices.size(); ++i) {
    const auto &joint = mLocalPose.joints[i];

    mGlobalPoseMatrices[i]  = glm::translate(glm::mat4(1.0f), joint.vTranslation);
    mGlobalPoseMatrices[i] *= glm::mat4_cast(joint.qRotation);

    // multiply non-root bones with their parent
    if (i > 0u) {
      mGlobalPoseMatrices[i] = mGlobalPoseMatrices[parent_ids[i]] * 
                               mGlobalPoseMatrices[i];
    }
  }
}


void SkeletonController::generate_skinning_datas() {
  const U32 numSkinningData = mGlobalPoseMatrices.size(); 
  const Matrix4x4 *inverse_bind_matrices = mSkeleton->inverse_bind_matrices();

# pragma omp parallel for schedule(static) num_threads(4)
  for (U32 i = 0u; i < numSkinningData; ++i) {
    // generate skinning matrices
    Matrix4x4 m = mGlobalPoseMatrices[i] * inverse_bind_matrices[i];
    mSkinningMatrices[i] = Matrix3x4(glm::transpose(m));

    // convert to Dual Quaternions
    mDQuaternions[i] = DualQuaternion(mSkinningMatrices[i]);
  }


  /// Upload to the shared skinning texture buffer (device)
  DeviceBuffer &buffer = sShared.skinning.buffer;
  buffer.bind(GL_TEXTURE_BUFFER);
  
  U32 bytesize(0u);
  F32 *data_ptr(nullptr);

  if (SKINNING_DQB == skinning_method()) {
    // DUAL QUATERNION BLENDING
    bytesize = numSkinningData * sizeof(mDQuaternions[0u]);
    data_ptr = reinterpret_cast<F32*>(mDQuaternions.data());
  } else {
    // LINEAR BLENDING
    bytesize = numSkinningData * sizeof(mSkinningMatrices[0u]);
    data_ptr = reinterpret_cast<F32*>(mSkinningMatrices.data());    
  }

  buffer.upload(0u, bytesize, data_ptr);
  buffer.unbind();

  CHECKGLERROR();
}


namespace {

void LerpSamples(const AnimationSample_t &s1,
                 const AnimationSample_t &s2,
                 const F32 factor,
                 AnimationSample_t &dst_sample) 
{
  const U32 nJoints = s1.joints.size();

# pragma omp parallel for schedule(static) num_threads(4)
  for (U32 i = 0u; i < nJoints; ++i) {
    JointPose_t &dst = dst_sample.joints[i];

    AER_CHECK(s1.joints.size() > i);
    AER_CHECK(s2.joints.size() > i);
    const JointPose_t &J1 = s1.joints[i];
    const JointPose_t &J2 = s2.joints[i];

    // for quaternions, use shortMix (slerp) or fastMix (nlerp) but NOT mix
    dst.qRotation    = glm::fastMix(J1.qRotation,J2.qRotation,    factor);
    dst.vTranslation = glm::mix(J1.vTranslation, J2.vTranslation, factor);
    dst.fScale       = glm::mix(J1.fScale,       J2.fScale,       factor);
  }
}

bool ComputePose(const F32 global_time,
                 SequenceClip_t& sequence_clip,
                 AnimationSample_t& dst_sample) {
  // TODO : - Use compressed joints data
  
  F32 local_time;  
  if (!sequence_clip.compute_localtime(global_time, local_time)) {
    sequence_clip.bEnable = false;
    return false;
  }

  const AnimationClip_t *clip = static_cast<AnimationClip_t*>(sequence_clip.action_ptr);

  /// Found the frame boundaries
  F32 lerp_frame = local_time * clip->framerate;
  U32 frame      = static_cast<U32>(lerp_frame) % clip->numframes;
  U32 next_frame = (frame + 1u) % clip->numframes;

  /// Compute the correct time sample for the pose
  const AnimationSample_t &s1 = clip->samples[frame];
  const AnimationSample_t &s2 = clip->samples[next_frame];
  F32 lerp_factor = lerp_frame - frame;

  LerpSamples(s1, s2, lerp_factor, dst_sample);
  return true;
}

}  // namespace

}  // namespace aer

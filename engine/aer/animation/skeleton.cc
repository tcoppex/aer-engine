// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2014 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#include "aer/animation/skeleton.h"

#include "aer/loader/skma.h"
#include "glm/gtc/matrix_transform.hpp"

// =============================================================================
namespace aer {
// =============================================================================

void Skeleton::init(const SKAFile& skaFile) {
  const U32 numjoints = skaFile.numbones();
  const U32 numclips  = skaFile.numsequences();

  mJoint.names.resize(numjoints);
  mJoint.parent_ids.resize(numjoints);
  mJoint.inverse_bind_matrices.resize(numjoints);

  // Name buffer
  U32 numelems = (numjoints * kJointNameSize + numclips * kClipNameSize);
  AER_CHECK(mNameBuffer.init(numelems * sizeof(char)));

  init_armature(skaFile);
  init_animations(skaFile);
}

// -----------------------------------------------------------------------------

void Skeleton::init_armature(const SKAFile& skaFile) {
  const SKAFile::TBone* const pBones = skaFile.bones();
  const U32 numjoints = skaFile.numbones();

  mJoint.names.resize(numjoints);
  mJoint.parent_ids.resize(numjoints);
  mJoint.inverse_bind_matrices.resize(numjoints);

  /// [debug]
  mGlobalBindMatrices.resize(numjoints);

  /// Allocate memory for joint names
  for (U32 i = 0u; i < numjoints; ++i) {
    mJoint.names[i] = mNameBuffer.allocate<char>(kJointNameSize);
  }

  // Retrieve joint datas
  for (U32 i = 0u; i < numjoints; ++i) {
    const auto &bone = pBones[i];
    
    // - Name
    snprintf(mJoint.names[i], kJointNameSize, "%s", bone.name);

    // - Parent id
    const U32 parent_id = bone.parentId;
    mJoint.parent_ids[i] = parent_id;

    // - Inverse bind pose matrix
    const SKAFile::TQuaternion &Q = bone.joint.qRotation;
    const SKAFile::TVector     &V = bone.joint.vTranslation;

    const Quaternion qRotation(Q.W, Q.X, Q.Y, Q.Z);
    const Vector3    vTranslation(V.X, V.Y, V.Z);

    Matrix4x4 rotation = glm::mat4_cast(qRotation);

    // [debug] local bind
    mGlobalBindMatrices[i] = glm::translate(Matrix4x4(1.0f), vTranslation) *
                             rotation;

    // inverse local bind
    mJoint.inverse_bind_matrices[i] = glm::transpose(rotation) *
                                      glm::translate(Matrix4x4(1.0f), -vTranslation);

    // for non-root bones
    if (i > 0u) {
      // [debug] global bind
      mGlobalBindMatrices[i] = mGlobalBindMatrices[parent_id] * mGlobalBindMatrices[i];

      // inverse global bind
      mJoint.inverse_bind_matrices[i] *= mJoint.inverse_bind_matrices[parent_id];
    }
  }
}

// -----------------------------------------------------------------------------

void Skeleton::init_animations(const SKAFile& skaFile) {
  // Note: SKA TSequence are AnimationClip
  //       SKA TFrame are AnimationSample
  const SKAFile::TSequence* pSequences = skaFile.sequences();  
  const SKAFile::TFrame*    pFrames    = skaFile.frames();
  const U32 numclips  = skaFile.numsequences();

  mClips.resize(numclips);

  // Allocate memory for clip names
  for (U32 i = 0u; i < numclips; ++i) {
    mClips[i].pName = mNameBuffer.allocate<char>(kClipNameSize);
  }

  AER_WARNING("Animation clips are set to loop by default");

  // Retrieve animation clips data
  for (U32 i = 0u; i < numclips; ++i) {
    AnimationClip_t &clip = mClips[i];

    snprintf(clip.pName, kClipNameSize, "%s", pSequences[i].name);
    clip.numframes  = pSequences[i].numFrame;
    clip.framerate  = pSequences[i].animRate;
    clip.samples    = new AnimationSample_t[clip.numframes];
    clip.bLoop      = true;  //

    AER_DEBUG_CODE(
    float clip_duration = clip.numframes / clip.framerate;
    printf("%d : %s (%.2fs)\n", i, clip.pName, clip_duration);
    )

    U32 ska_frameid = pSequences[i].startFrame;
    for (U32 fid = 0u; fid < clip.numframes; ++fid) {
      AnimationSample_t &sample = clip.samples[fid];
      sample.joints.resize(numjoints());

      for (U32 jid = 0u; jid < numjoints(); ++jid, ++ska_frameid) {
        JointPose_t &joint = sample.joints[jid];

        const SKAFile::TQuaternion &Q = pFrames[ska_frameid].qRotation;
        const SKAFile::TVector     &V = pFrames[ska_frameid].vTranslate;
        const F32                   S = pFrames[ska_frameid].fScale;
        
        joint.qRotation    = Quaternion(Q.W, Q.X, Q.Y, Q.Z);
        joint.vTranslation = Vector3(V.X, V.Y, V.Z);
        joint.fScale       = S;
      }
    }
  }
}

// =============================================================================
}  // namespace aer
// =============================================================================

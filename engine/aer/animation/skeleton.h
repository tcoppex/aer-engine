// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2014 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_ANIMATION_SKELETON_H_
#define AER_ANIMATION_SKELETON_H_

#include <vector>

#include "aer/common.h"
#include "aer/animation/common.h"
#include "aer/memory/stack_allocator.h"
#include "aer/utils/global_clock.h"

// =============================================================================
namespace aer {
// =============================================================================

class SKAFile;

/**
 * @class Skeleton
 * @brief Defines a hierarchy of joints used in skeletal animations
*/
class Skeleton {
 public:
  // TODO : externalize setup from skaFile
  void init(const SKAFile& skaFile);

  U32 numjoints() const {
    return mJoint.parent_ids.size();
  }

  U32 numclips() const {
    return mClips.size();
  }

  const char* const* names() const {
    return mJoint.names.data();
  }

  const I32* parent_ids() const {
    return mJoint.parent_ids.data();
  }

  const Matrix4x4* inverse_bind_matrices() const {
    return mJoint.inverse_bind_matrices.data();
  }

  const Matrix4x4* bind_matrices() const {
    return mGlobalBindMatrices.data();
  }

  // ----------------------------------------------------------
  void generate_sequence(Sequence_t &sequence) {
    F32 global_time = GlobalClock::Get().application_time(SECOND);
    sequence.resize(mClips.size());
    for (U32 i = 0u; i < sequence.size(); ++i) {
      sequence[i].action_ptr   = &mClips[i];
      sequence[i].global_start = global_time;
    }
  }
  // ----------------------------------------------------------

 private:
  // ---------------------------------------------------------------------------
  /// @name Constant values
  // ---------------------------------------------------------------------------

  static const U32 kJointNameSize = 32u;
  static const U32 kClipNameSize  = 32u;

  // ---------------------------------------------------------------------------
  /// @name Initalizers
  // ---------------------------------------------------------------------------

  void init_armature(const SKAFile& skaFile);
  void init_animations(const SKAFile& skaFile);

  // ---------------------------------------------------------------------------
  /// @name Attributes
  // ---------------------------------------------------------------------------

  // Common
  StackAllocator mNameBuffer;           // Stores joints and clips name

  /// Armature
  struct {
    std::vector<char*>     names;
    std::vector<I32>       parent_ids;
    std::vector<Matrix4x4> inverse_bind_matrices;
  } mJoint;

  /// Animation clips
  std::vector<AnimationClip_t> mClips;

  /// [debug]
  std::vector<Matrix4x4> mGlobalBindMatrices;
};

// =============================================================================
}  // namespace aer
// =============================================================================

#endif  // AER_ANIMATION_SKELETON_H_

// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AER_ANIMATION_MORPH_CONTROLLER_H_
#define AER_ANIMATION_MORPH_CONTROLLER_H_

#include "aer/common.h"
#include "aer/animation/common.h"
#include "aer/animation/blend_shape.h"
#include "aer/animation/blend_tree.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// WIP draft
/// (idea : mimic the skeleton controller for blend shapes)
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class MorphController {
 public:
  MorphController() = default;

  ///
  bool init(BlendShape *blendshape);

  /// Launch the mesh morphing pipeline
  void update();


  //-----------------------------
  BlendShape& blend_shape() {
    AER_ASSERT(nullptr != mBlendShape_ptr);
    return *mBlendShape_ptr;
  }

  BlendTree& blend_tree() {
    return mBlendTree;
  }

  U32 total_expressions() const {
    return mTotalExpressions;
  }

  void add_expressions(Expression_t *expressions, U32 size) {
    mBlendShape_ptr->add_expressions(mSequence, expressions, size);
  }
  //-----------------------------


 private:
  void compute_expressions();
  void update_buffers();

  /// Reference on the blendshapes handler
  BlendShape *mBlendShape_ptr = nullptr;

  /// Inputs
  Sequence_t mSequence;
  BlendTree  mBlendTree;

  /// Buffer of unpacked blendshape weights [could be shared]
  std::vector<F32> mWeightsBuffer;
  U32              mTotalExpressions;


  DISALLOW_COPY_AND_ASSIGN(MorphController);
};

}  // namespace aer

#endif  // AER_ANIMATION_MORPH_CONTROLLER_H_

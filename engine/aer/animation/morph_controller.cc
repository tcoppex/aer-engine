// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2014 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#include "aer/animation/morph_controller.h"
#include "aer/utils/global_clock.h"

// =============================================================================
namespace {
// =============================================================================

/// Calculate the sequence_clip phase at global_time
/// @return false if the clip is finished
bool ComputeExpression(const aer::F32 global_time,
                       aer::SequenceClip_t& sequence_clip,
                       aer::F32& dst_weight)
{
  aer::F32 local_time;

  if (!sequence_clip.compute_localtime(global_time, local_time)) {
    sequence_clip.bEnable = false;
    return false;
  }
  dst_weight = sequence_clip.phase(local_time);

  return true;
}

// =============================================================================
}  // namespace
// =============================================================================


// =============================================================================
namespace aer {
// =============================================================================

bool MorphController::init(BlendShape *blendshape) {
  AER_ASSERT(nullptr == mBlendShape_ptr);

  if (nullptr == blendshape) {
    return false;
  }

  mBlendShape_ptr = blendshape;
  mBlendShape_ptr->generate_sequence(mSequence);
  AER_DEBUG_CODE(mBlendShape_ptr->DEBUG_display_names();)

  mWeightsBuffer.resize(mBlendShape_ptr->count());

  return true;
}

// -----------------------------------------------------------------------------

void MorphController::update() {
  // TODO : test if the tree is set or not
  mBlendTree.activate_leaves(true, mSequence); //
  mBlendTree.evaluate(1.0f, mSequence);
  //--

  /// Compute the factor (the phase) of every expressions
  /// and multiply it by its weight
  compute_expressions();

  /// Update Weights & Indices device's buffers
  update_buffers();
}

// -----------------------------------------------------------------------------

void MorphController::compute_expressions() {
  F32 global_time = GlobalClock::Get().application_time(SECOND);

  mWeightsBuffer.assign(mWeightsBuffer.size(), 0.0f);

  F32 dst_weight;
  for (auto &sc : mSequence) {
    if (!sc.bEnable || !ComputeExpression(global_time, sc, dst_weight)) {
      continue;
    }

    const Expression_t* expression = static_cast<Expression_t*>(sc.action_ptr);
    
    // [to move] Allow a user to manually set the time weight
    if (expression->bManualBypass) {
      dst_weight = 1.0f;
    }

    for (auto index : expression->indices) {
      mWeightsBuffer[index] = dst_weight * sc.weight;
    }
  }
}

// -----------------------------------------------------------------------------

void MorphController::update_buffers() {
  U32 cursor_index;

  DeviceBuffer &tbo_indices = mBlendShape_ptr->texture_buffer(BlendShape::BS_INDICES).buffer;
  tbo_indices.bind(GL_TEXTURE_BUFFER);

  U32 *indices;
  tbo_indices.map(&indices, GL_WRITE_ONLY);
  {
    cursor_index = 0u;
    for (U32 index = 0u; index < mBlendShape_ptr->count(); ++index) {
      if (mWeightsBuffer[index] > 0.0f) {
        indices[cursor_index++] = index;
      }
    }
  }
  tbo_indices.unmap(&indices);


  DeviceBuffer &tbo_weights = mBlendShape_ptr->texture_buffer(BlendShape::BS_WEIGHTS).buffer;
  tbo_weights.bind(GL_TEXTURE_BUFFER);

  F32 *weights;
  tbo_weights.map(&weights, GL_WRITE_ONLY);
  {
    cursor_index = 0u;
    for (U32 index = 0u; index < mBlendShape_ptr->count(); ++index) {
      F32 w = mWeightsBuffer[index];
      if (w > 0.0f) {
        weights[cursor_index++] = w;
      }
    }
  }
  tbo_weights.unmap(&indices);

  // Buffer size used
  mTotalExpressions = cursor_index;

  DeviceBuffer::Unbind(GL_TEXTURE_BUFFER);
}

// =============================================================================
}  // namespace aer
// =============================================================================

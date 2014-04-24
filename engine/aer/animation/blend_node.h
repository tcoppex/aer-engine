// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2014 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_ANIMATION_BLEND_NODE_H_
#define AER_ANIMATION_BLEND_NODE_H_

#include "aer/common.h"

// =============================================================================
namespace aer {
// =============================================================================

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
// Datastructures used to represent blend tree nodes
// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 

/**
 * @class BlendNode
 * @brief Base node used in the Blend Tree
*/
class BlendNode {
public:
  /// Compute final weight for each clips
  virtual void compute_weight(const F32 weight) = 0;
};

// =============================================================================

/**
 * @class LerpNode
 * @brief Interpolates two nodes together
*/
class LerpNode : public BlendNode {
public:
  LerpNode(BlendNode *input1, BlendNode *input2) :
    mFactor(0.0f),
    mpInput1(input1),
    mpInput2(input2)
  {}

  void compute_weight(const F32 weight) override {
    mpInput1->compute_weight((1.0f - mFactor) * weight);
    mpInput2->compute_weight(mFactor * weight);
  }

  void set_factor(const F32 factor) {
    mFactor = factor;
  }

  F32 factor() const {
    return mFactor;
  }

 private:
  F32 mFactor;
  BlendNode *mpInput1;
  BlendNode *mpInput2;
};

// =============================================================================

/**
 * @class CoeffNode
 * @brief Multiply a subtree by a coefficient factor
*/
class CoeffNode : public BlendNode {
  public:
    CoeffNode(BlendNode *input1) :
      mFactor(1.0f),
      mpInput1(input1)
    {}

    void compute_weight(const F32 weight) override {
      mpInput1->compute_weight(mFactor * weight);
    }

    void set_factor(const F32 factor) {
      mFactor = factor;
    }

    F32 factor() const {
      return mFactor;
    }

   private:
    F32 mFactor;
    BlendNode *mpInput1;
};

// =============================================================================

/**
 * @class LeaveNode
 * @brief Leave Node storing the final weight
*/
class LeaveNode : public BlendNode {
 public:
  LeaveNode() :
    mWeight(1.0f)
  {}

  void compute_weight(const F32 weight) override {
    mWeight = weight;
  }

  F32 weight() const {
    return mWeight;
  }

 private:
  F32 mWeight;
};

// =============================================================================
}  // namespace aer
// =============================================================================

#endif  // AER_ANIMATION_BLEND_NODE_H_

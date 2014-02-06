// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AER_ANIMATION_LAYERED_ASM_H_
#define AER_ANIMATION_LAYERED_ASM_H_

#include <vector>
#include "aer/animation/blend_tree.h"
#include "aer/animation/_wip_/action_state_machine.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// Defines a stack of Action State Machines, each one
/// contributing to the final Blend Tree, by concatening
/// sub-BlendTree from bottom to top of the stack.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class LayeredASM {
 public:
  enum LayerType {
    LERP_LAYER,
    ADDITIVE_LAYER,   // not implemented yet
    kNumTypeLayer
  };

  /// -- Layer Management --
  void add_layer(const LayerType type, ActionStateMachine *pASM);

  void set_layer_factor(const U32 layer_id, const F32 factor);


  /// -- State handler --
  /// Update each layer's state
  //void update(const F32 dt);

  /// Test whether or not a transition is accessible from the specified layer
  bool is_accessible(const U32 layer_id, const std::string &tname);

  /// Change current state for layer layer_id
  bool access(const U32 layer_id, const std::string &tname);

  /// -- Blend contributions --
  /// Compute weights and attribute them to the active clips
  void evaluate(Sequence_t &sequence);
  
  /// Return the named node from the specified layer or NULL if it does not exist.
  /// !! The node is search only in the current state !!
  template<typename T>
  T* find_layer_node(const U32 layer_id, const std::string name);


 private:
  struct Layer {
    LayerType           type;               // type of layer
    F32                 factor;             // contribution factor
    ActionStateMachine  *pASM;              // reference the layer's ASM
    ActionStateMachine::StateInfo state;    // Info to access the state
  };

  /// Return the current state for the given layer
  ActionState& layer_state(const U32 layer_id);

  /// Stack of layer
  std::vector<Layer> mLayers;
};


template<typename T>
T* LayeredASM::find_layer_node(const U32 layer_id, const std::string name) {
  AER_ASSERT(layer_id < mLayers.size());
  BlendTree& tree = layer_state(layer_id).blend_tree();
  return tree.find_node<T>(name);
}

}  // namespace aer

#endif  // AER_ANIMATION_LAYERED_ASM_H_

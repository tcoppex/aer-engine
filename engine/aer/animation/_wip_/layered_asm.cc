// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#include "aer/animation/_wip_/layered_asm.h"


namespace aer {

void LayeredASM::add_layer(const LayerType type, ActionStateMachine *pASM) {
  AER_ASSERT(pASM != nullptr);

  Layer layer;
  layer.type   = type;
  layer.factor = 0.0f;
  layer.pASM   = pASM;

  // TODO : get the default infos from the ActionStateMachine
  layer.state.id                = 0u;
  layer.state.start_time        = 0.0f;
  layer.state.pTakenTransition  = nullptr;

  mLayers.push_back(layer);
}

void LayeredASM::set_layer_factor(const U32 layer_id, const F32 factor) {
  AER_ASSERT(layer_id < mLayers.size());
  mLayers[layer_id].factor = factor;
}


bool LayeredASM::is_accessible(const U32 layer_id, const std::string &tname) {
  AER_ASSERT(layer_id < mLayers.size());
  return layer_state(layer_id).is_accessible(tname);
}

bool LayeredASM::access(const U32 layer_id, const std::string &tname) {
  AER_ASSERT(layer_id < mLayers.size());
  Layer &layer = mLayers[layer_id];
  return layer.pASM->access(tname, layer.state);  
}

void LayeredASM::evaluate(Sequence_t &sequence) {
  F32 totalWeight = 1.0f;

  const U32 N = mLayers.size();
  for (U32 i = 0u; i < N; ++i) {
    const U32 layer_id = N-(i+1u);
    Layer &layer = mLayers[layer_id];
   
    BlendTree &tree = layer_state(layer_id).blend_tree();

    // TODO : -activate / deactivate clips
    //        -handle cross fading with layer.state.pTakenTransition

    tree.evaluate(totalWeight * layer.factor, sequence);
    totalWeight *= 1.0f - layer.factor;
  }
}

ActionState& LayeredASM::layer_state(const U32 layer_id) {
  AER_ASSERT(layer_id < mLayers.size());
  Layer &layer = mLayers[layer_id];
  return layer.pASM->state(layer.state.id);
}

}  // namespace aer

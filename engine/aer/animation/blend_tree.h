// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AER_ANIMATION_BLEND_TREE_H_
#define AER_ANIMATION_BLEND_TREE_H_

#include <unordered_map>
#include <string>

#include "aer/common.h"
#include "aer/animation/blend_node.h"
#include "aer/animation/common.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// Aggregates animations blending operations 
/// (eg. clips and blend shapes -todo-)
///
/// Note : string should be replaced by a StringID, a
///        integral reference to a buffer of name.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class BlendTree {
 public:
  BlendTree();
  ~BlendTree();

  /// Parse the tree to calculate blending weights and assign them to clips
  void evaluate(const F32 factor, Sequence_t& sequence);

  /// Enable or disable the tree leaves from sequence
  void activate_leaves(bool bEnable, Sequence_t& sequence);

  /// Add a blend node to the tree, the last one is used as the
  /// root of the tree.
  BlendNode* add_node(const std::string& name, BlendNode *node);

  /// Add a leave node to the tree, name should be a reference to
  /// an animation data. If the root is unspecified, it become the
  /// the new root.
  LeaveNode* add_leave(const std::string& name);

  /// Search for an internal node by its name and cast it to the requested type.
  /// Return nullptr if no matching node was found.
  template<typename T>
  T* find_node(const std::string& name);


 private:
  typedef std::unordered_map<std::string, BlendNode*> NodeMap_t;
  typedef std::unordered_map<std::string, LeaveNode*> LeaveMap_t;

  typedef NodeMap_t::iterator   NodeMapIterator_t;
  typedef LeaveMap_t::iterator  LeaveMapIterator_t;


  // Root of the tree
  BlendNode *mRoot;

  // Reference internal nodes by name
  NodeMap_t mNodeMap;

  // Reference leaves by name
  LeaveMap_t mLeaveMap;
};

template<typename T>
T* BlendTree::find_node(const std::string& name) {
  NodeMapIterator_t it = mNodeMap.find(name);
  if (it != mNodeMap.end()) {
    return dynamic_cast<T*>(it->second);
  }
  return nullptr;
}

}  // namespace aer

#endif  // AER_ANIMATION_BLEND_TREE_H_

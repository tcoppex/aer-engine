// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AER_BVH_NODE_H_
#define AER_BVH_NODE_H_

#include "aer/common.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// 
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class BVHNode {
 public:
  static const U32 kLeafFlag   = 0x80000000;  // (1u << 31u);
  static const U32 kOffsetFlag = 0x7FFFFFFF;

  BVHNode() : 
    id_(0u),
    parent_(0u),
    flags_(0u)
  {}

  BVHNode(U32 node_id_, U32 parent_id_, U32 offset, bool bIsALeaf) {
    set(node_id_, parent_id_, offset, bIsALeaf);
  }


  void set(U32 node_id_, U32 parent_id_, U32 offset, bool bIsALeaf) {
    id_ = node_id_;
    parent_ = parent_id_;
    flags_ = offset & kOffsetFlag;
    if (bIsALeaf) {
      flags_ |= kLeafFlag;
    }
  }

  U32 id()        const { return id_; }
  U32 parent()    const { return parent_; }
  U32 flags()     const { return flags_; }

  U32 offset()    const { return flags_ & kOffsetFlag; }
  bool is_leaf()  const { return flags_ & kLeafFlag; }

 private:
  U32 id_;              // node id
  U32 parent_;          // parent id
  U32 flags_;           // bits 0..30 : offset to first child OR leafId
                        // bits 31    : flag wether the node is a leaf

};

} // aer

#endif  // AER_BVH_NODE_H_

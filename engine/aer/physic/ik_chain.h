// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2014 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AER_PHYSIC_IK_CHAIN_H_
#define AER_PHYSIC_IK_CHAIN_H_

#include <forward_list>

#include "aer/common.h"
#include "aer/physic/ik_node.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
///
/// Represents a hierarchical set of joints use to solve Inverse Kinematics
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~
class IKChain {
 public:
  typedef std::forward_list<IKNode*>  Container_t;
  typedef Container_t::iterator       iterator;
  typedef Container_t::const_iterator const_iterator;


  IKChain() : 
    num_nodes_(0u),
    num_joints_(0u),
    num_end_effector_(0u)
  {}

  ~IKChain() {
    clear();
  }

  /// Insert a node in the list
  void insert_node(IKNode* node);

  /// Reset the chains
  void clear();

  /// Update the word-space coordinates of the chain's nodes
  void update();

  /// Getters
  U32 num_nodes()        const { return num_nodes_;        }
  U32 num_joints()       const { return num_joints_;       }
  U32 num_end_effector() const { return num_end_effector_; }

  /// Give access to the chain's iterators
  iterator       begin()       { return chain_.begin(); }
  const_iterator begin() const { return chain_.begin(); }
  iterator       end()         { return chain_.end();   }
  const_iterator end()   const { return chain_.end();   }


 private:
  Container_t chain_;

  U32 num_nodes_;
  U32 num_joints_;
  U32 num_end_effector_;
};

}  // namespace aer

#endif  // AER_PHYSIC_IK_CHAIN_H_

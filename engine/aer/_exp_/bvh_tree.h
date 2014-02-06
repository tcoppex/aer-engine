// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
//
// -----------------------------------------------------------------------------

#ifndef AER_BVH_TREE_H_
#define AER_BVH_TREE_H_


#include <vector>
#include "aer/common.h"

#include "aer/rendering/bvh_node.h"
#include "aer/rendering/mesh.h" //
#include "aer/rendering/bounding_box.h" //
#include "aer/view/frustum.h" //
#include "aer/misc/intersection.h" //


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// Bounding volume hierarchy using a surface area
/// heuristic.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class BVHTree {
 public:
  BVHTree();

  ///
  void build(U32 nelems, const Mesh *meshes, const BoundingBox *bboxes);

  ///
  //void run(const aer::Camera &camera);

  ///
  void debug_display(const Vector4 planes[Frustum::kNumPlane],
                     BVHNode *node = nullptr, 
                     U32 depth = 0u);


 private:
  /// Maps common buffers for the building process
  struct BuildParams_t {
    U32 *sorted_indices[3];      // per-axis sorted indices
    U32 *lut_ptr;                // Used to resort indices_ptr
    U32 *resort_ptr;             // tmp buffer to store resorted indices

    BuildParams(U32 *sortedX,
                U32 *sortedY,
                U32 *sortedZ,
                U32 *lut,
                U32 *resort) : 
    sorted_indices{sortedX, sortedY, sortedZ},
    lut_ptr(lut),
    resort_ptr(resort)
    {}
  };

  /// Node's attributes use specifically by this tree
  struct NodeAttrib_t {
    NodeAttrib_t() : 
      failing_plane_id(0u) 
    {}

    BoundingBox aabb;
    U8          failing_plane_id;
  };



  virtual void init() {}

  /// Sort AABBs indices per axis
  void sortAABBs(BuildParams_t &params);

  /// subroutine of the build process
  void subBuild(const BuildParams_t &params, 
                U32 node_id,
                U32 parent_id,
                U32 offset,
                U32 count,
                U32 *counter_ptr);

  ///
  bool find_best_sa_cost(const BuildParams_t &params, 
                         U32 offset, 
                         U32 count, 
                         U8 axis, 
                         F32 *bestcost_ptr,
                         U32 *bestid_ptr);

  ///
  void resort(const BuildParams_t &params,
              U32 offset,
              U32 count,
              U32 split_id,
              U8 base_axis,
              U8 axis);


  /// Test wether a node is inside the given frustum or not
  bool inside_view_frustum(const Vector4 planes[Frustum::kNumPlane],
                           U32 node_id) {
    NodeAttrib_t &n = attribs_[node_id];
    return intersect_frustumAABB(planes, n.aabb, &n.failing_plane_id) != OUTSIDE;
  }


  U32 numnodes_;                  // number of internal nodes
  U32 numleaves_;                 // number of external nodes (leaves)

  std::vector<BVHNode>      tree_;
  std::vector<NodeAttrib_t> attribs_;

  /// Leaves attribs (pointers to external buffers)
  const Mesh *meshes_;            //
  const BoundingBox *bboxes_;     // mapped leaves' AABBs (per objects)

  /// Buffer used during the build process [could be shared]
  std::vector<U32> build_buffer_ui_;


  DISALLOW_COPY_AND_ASSIGN(BVHTree);
};

} // aer

#endif  // AER_BVH_TREE_H_

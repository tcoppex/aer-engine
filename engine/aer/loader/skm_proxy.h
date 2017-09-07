// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_LOADER_SKM_PROXY_H_
#define AER_LOADER_SKM_PROXY_H_

#include <string>
#include <vector>

#include "aer/common.h"
#include "aer/animation/blend_shape.h"
#include "aer/memory/resource_proxy.h"
#include "aer/rendering/mesh.h"

// =============================================================================
namespace aer {
// =============================================================================

// forward declarations
class Skeleton;
class SKMFile;

// -----------------------------------------------------------------------------

/**
 * @struct SkinnedVertex
 * @brief Datastructs used to compute attribs size
*/
struct SkinnedVertex {
  F32 position[3];
  F32 normal[3];
  F32 texCoord[2];
  U8  jointIndex[4];
  F32 jointWeight[3];
};

// -----------------------------------------------------------------------------

/**
 * @struct VertexOffset
*/
struct VertexOffset {
  UPTR position;
  UPTR normal;
  UPTR texCoord;
  UPTR jointIndex;
  UPTR jointWeight;

  VertexOffset(const U32 nvertices) {
    SkinnedVertex sv;

    position    = 0u;
    normal      = position    + nvertices * sizeof(sv.position);
    texCoord    = normal      + nvertices * sizeof(sv.normal);
    jointIndex  = texCoord    + nvertices * sizeof(sv.texCoord);
    jointWeight = jointIndex  + nvertices * sizeof(sv.jointIndex);

    UPTR offset = jointWeight + nvertices * sizeof(sv.jointWeight);
    AER_CHECK(offset == nvertices * sizeof(sv));
  }
};

// -----------------------------------------------------------------------------

/**
 * @struct SKMInfo_t
 * @brief Holds binding informations between mesh & skeleton
*/
struct SKMInfo_t {
  Mesh                      mesh;
  BlendShape                *blendshape = nullptr;
  std::string               skeleton_name;
  std::vector<std::string>  material_ids;


  ~SKMInfo_t() {
    AER_SAFE_DELETE(blendshape);
  }

  bool has_blendshape() const {
    return nullptr != blendshape;
  }

  bool has_skeleton() const {
    return false == skeleton_name.empty();
  }

  U32 material_count() const {
    return material_ids.size();
  }
};

// -----------------------------------------------------------------------------

/**
 * @class SKMProxy
 * @brief Manages loaded SKM Object datas
 * @see ResourceProxy, SKMInfo_t
*/
class SKMProxy : public ResourceProxy<SKMInfo_t> {
public:
  virtual SKMInfo_t* load(const std::string& id) override;

private:
  /// All the mesh initializer could be exported to the Mesh object !
  void init_mesh(const SKMFile& skmFile, Mesh& mesh);
  void init_mesh_vertices(const SKMFile &skmFile, Mesh &mesh);
  void init_mesh_indices(const SKMFile &skmFile, Mesh &mesh);
};

// =============================================================================
}  // namespace aer
// =============================================================================

#endif  // AER_LOADER_SKM_PROXY_H_

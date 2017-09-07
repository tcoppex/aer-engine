// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef MARCHEUR_SKM_MODEL_H_
#define MARCHEUR_SKM_MODEL_H_

#include <string>
#include <vector>

#include "aer/common.h"
#include "aer/animation/morph_controller.h"
#include "aer/animation/skeleton_controller.h"
#include "aer/loader/skm_proxy.h"
#include "aer/rendering/mesh.h"
#include "aer/rendering/material.h"
#include "aer/view/camera.h"


// =============================================================================
namespace aer {
// =============================================================================

/**
 *  @name SKModel
 *  @brief Model loaded from a SKM / SKA file
*/
class SKMModel {
 public:
  SKMModel();
  ~SKMModel();

  /// Load model (mesh / skeleton / materials)
  /// @return true if the model has been successfuly loaded
  /// @param skm_path : path for the model to load
  /// @param material_path : path for the materials [optional]
  bool load(const std::string &skm_path,
            const std::string &material_path = "");

  /// Update model's animation states
  void update();

  /// Render the model
  void render(const Camera &camera);


  // ---------------------------------------------------------------------------
  /// @name Getters
  // ---------------------------------------------------------------------------
  const Mesh& mesh() const {
    AER_ASSERT(bLoaded);
    return mSKMInfo->mesh;
  }

  // [temp]
  //------------------------
  SkeletonController& skeleton_controller() {
    return mSkeletonController;
  }

  MorphController& morph_controller() {
    return mMorphController;
  }

  const Material& material(U32 index) const {
    return mMaterials[index];
  }
  //------------------------


 private:
  /// Material can be changed individually
  bool load_material(const std::string &path);

  /// The singleton SKMProxy is only used internally by SKMModel
  static SKMProxy sProxy;


  // ---------------------------------------------------------------------------
  /// @name Attributes
  // ---------------------------------------------------------------------------

  std::string mNameRef;                    // to release the SKMinfo [remove]
  SKMInfo_t *mSKMInfo;

  MorphController mMorphController;        // contains BlendShape reference
  SkeletonController mSkeletonController;  // contains Skeleton reference

  std::vector<Material> mMaterials;

  bool bLoaded;
};

// =============================================================================
}  // namespace aer
// =============================================================================

#endif  // MARCHEUR_SKM_MODEL_H_

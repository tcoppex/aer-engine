// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#include "aura/skm_model.h"
#include "aer/loader/skm_proxy.h"


namespace aer {

SKMProxy SKMModel::sProxy;

SKMModel::SKMModel() :
  mSKMInfo(nullptr),
  bLoaded(false)
{}

SKMModel::~SKMModel() {
  if (!bLoaded) {
    return;
  }

  sProxy.release(mNameRef);
}

bool SKMModel::load(const std::string &skm_path,
                    const std::string &material_path) {
  AER_ASSERT(false == bLoaded);


  mNameRef = skm_path; //
  mSKMInfo = sProxy.get(mNameRef);

  if (nullptr == mSKMInfo) {
    AER_WARNING(skm_path + " : file does not exist");
    return false;
  }

  if (mSKMInfo->has_blendshape()) {
    mMorphController.init(mSKMInfo->blendshape);
  }

  if (mSKMInfo->has_skeleton()) {
    AER_CHECK(mSkeletonController.init(mSKMInfo->skeleton_name));
  }

  if (!material_path.empty()) {
    load_material(material_path);
  }

  bLoaded = true;
  return bLoaded;
}

bool SKMModel::load_material(const std::string &path) {
  MATFile matFile;
  if (!matFile.load(path.c_str())) {
    return false;
  }

  mMaterials.resize(mSKMInfo->material_count());
  std::string dirname(matFile.filepath()); //
  dirname += "textures/"; //

  for (U32 i = 0u; i < mMaterials.size(); ++i) {
    Material &material = mMaterials[i];

    const std::string& matname = mSKMInfo->material_ids[i];
    const char *Kd_name = matFile.material_from_name(matname, MATFile::TEXTURE_DIFFUSE);

    if (Kd_name != nullptr) {
      std::string filename(dirname + Kd_name);  //
      material.set_diffuse_map_id(filename);
    }
  }

  // TODO : - define a global sampler with anisotropy sampling
  //        - generate mip-map for the texture


  return true;
}

void SKMModel::update() {
  mMorphController.update();
  mSkeletonController.update();
}

void SKMModel::render(const Camera &camera) {
  mesh().draw(); //
}

} // namespace aer

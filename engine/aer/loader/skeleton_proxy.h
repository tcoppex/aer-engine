// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_LOADER_SKELETON_PROXY_H_
#define AER_LOADER_SKELETON_PROXY_H_

#include <string>

#include "aer/common.h"
#include "aer/memory/resource_proxy.h"
#include "aer/animation/skeleton.h"
#include "aer/loader/skma.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
///
/// Manage skeletons resources
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
class SkeletonProxy : public aer::ResourceProxy<Skeleton> {
 public:
  virtual Skeleton* load(const std::string& id) override {
    SKAFile skaFile;

    std::string filename = id + ".ska";
    if (!skaFile.load(filename.c_str())) {
      return nullptr;
    }

    Skeleton *skeleton = new Skeleton();
    if (!skeleton) {
      return nullptr;
    }
    skeleton->init(skaFile);

    return skeleton;
  }
};

}  // namespace aer


#endif  // AER_LOADER_SKELETON_PROXY_H_

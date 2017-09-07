// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_LOADER_TEXTURE_2D_PROXY_H_
#define AER_LOADER_TEXTURE_2D_PROXY_H_

#include "aer/common.h"
#include "aer/memory/resource_proxy.h"
#include "aer/device/texture_2d.h"

#include "aer/utils/singleton.h"


namespace aer {


/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
///
/// Manage texture 2D resources
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
class Texture2DProxy : public ResourceProxy<Texture2D>, 
                       public Singleton<Texture2DProxy> {
 public:
  virtual Texture2D* load(const std::string& id) override;

  friend class Singleton<Texture2DProxy>;
};

}  // namespace aer

#endif  // AER_LOADER_TEXTURE_2D_PROXY_H_

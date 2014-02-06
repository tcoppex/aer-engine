// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_DEVICE_DEFAULT_SAMPLER_H_
#define AER_DEVICE_DEFAULT_SAMPLER_H_

#include "aer/common.h"
#include "aer/device/sampler.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// Defines some default samplers.
/// Those can't be attached to texture 2D, but are rather
/// made to bind them directly to texture unit.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class DefaultSampler {
 public:
  static
  const Sampler& NearestClampled() {
    static bool bInit = false;
    if (!bInit) {
      kNearestClamped.generate();
      kNearestClamped.set_filters(GL_NEAREST, GL_NEAREST);
      kNearestClamped.set_wraps(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
      bInit = true;
    }
    return kNearestClamped;
  }

  static
  const Sampler& LinearClampled() {
    static bool bInit = false;
    if (!bInit) {
      kLinearClamped.generate();
      kLinearClamped.set_filters(GL_LINEAR, GL_LINEAR);
      kLinearClamped.set_wraps(GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE, GL_CLAMP_TO_EDGE);
      bInit = true;
    }
    return kLinearClamped;
  }

  static
  const Sampler& LinearRepeat() {
    static bool bInit = false;
    if (!bInit) {
      kLinearRepeat.generate();
      kLinearRepeat.set_filters(GL_LINEAR, GL_LINEAR);
      kLinearRepeat.set_wraps(GL_REPEAT, GL_REPEAT, GL_REPEAT);
      bInit = true;
    }
    return kLinearRepeat;
  }

  static
  const Sampler& AnisotropyRepeat() {
    static bool bInit = false;
    if (!bInit) {
      kAnisoRepeat.generate();
      kAnisoRepeat.set_filters(GL_LINEAR, GL_LINEAR);
      kAnisoRepeat.set_wraps(GL_REPEAT, GL_REPEAT, GL_REPEAT);
      kAnisoRepeat.set_anisotropy_level(16.0f);
      bInit = true;
    }
    return kAnisoRepeat;
  }

 private:
  static Sampler kNearestClamped;
  static Sampler kLinearClamped;
  static Sampler kLinearRepeat;
  static Sampler kAnisoRepeat;
};

}  // namespace aer

#endif  // AER_DEVICE_DEFAULT_SAMPLER_H_
// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_RENDERING_DRAWABLE_H_
#define AER_RENDERING_DRAWABLE_H_

#include "aer/common.h"

namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
///
/// Interface for drawable objects
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
class Drawable {
 public:
  virtual void draw() const = 0;
  virtual void draw_instances(const U32 count) const = 0;
};

}  // namespace aer

#endif  // AER_RENDERING_DRAWABLE_H_

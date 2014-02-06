// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_DEVICE_VERTEX_ARRAY_H_
#define AER_DEVICE_VERTEX_ARRAY_H_

#include "aer/common.h"
#include "aer/device/device_resource.h"

namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ +
///
///
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class VertexArray : public DeviceResource {
 public:
  static
  void Unbind() {
    glBindVertexArray(0u);
  }

  void generate() {
    AER_ASSERT(!is_generated());
    glGenVertexArrays(1, &id_);
  }

  void release() {
    if (is_generated()) {
      glDeleteVertexArrays(1, &id_);
      id_ = 0u;
    }
  }

  void bind() const {
    AER_ASSERT(is_generated());
    glBindVertexArray(id_);
  }

  void unbind() const {
    Unbind();
  }
};
  
}  // namespace aer

#endif  // AER_DEVICE_VERTEX_ARRAY_H_

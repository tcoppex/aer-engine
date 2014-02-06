// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_DEVICE_RESOURCE_H_
#define AER_DEVICE_RESOURCE_H_

#include "aer/common.h"
#include "aer/core/opengl.h"

namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
///  Interface for OpenGL objects using the glGen*
///  class of function.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class DeviceResource
{
 public:
  DeviceResource()
    : id_(0u)
  {}

  virtual void generate() = 0;
  virtual void release()  = 0;

  inline const U32 id() const { return id_; }
  inline bool is_generated() const { return 0u != id_; }

 protected:
  U32 id_;
};

}  // namespace aer

#endif  // AER_DEVICE_RESOURCE_H_

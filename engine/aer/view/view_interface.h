// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_VIEW_VIEW_INTERFACE_H_
#define AER_VIEW_VIEW_INTERFACE_H_

#include "aer/common.h"

namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// [Abstract] location in a 3D space
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class Position3DInterface
{
 public:
  virtual void set_position(const Vector3&) = 0;
  virtual const Vector3& position() const   = 0;
};


/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// [Abstract] View point in a 3D space
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class View3DInterface : public Position3DInterface
{
 public:
  virtual void set_direction(const Vector3& direction) = 0;
  virtual void set_target(const Vector3& target)       = 0;

  virtual const Vector3& direction() const = 0;
  virtual const Vector3& up()        const = 0;

  virtual const Matrix4x4& view_matrix() = 0;
};

}  // namespace aer

#endif  // AER_VIEW_VIEW_INTERFACE_H_

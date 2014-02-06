// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_VIEW_VIEW_H_
#define AER_VIEW_VIEW_H_

#include "aer/common.h"
#include "aer/view/view_interface.h"

namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// Defines a system of coordinates in the 3D space
/// used for describing a view.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class View : public View3DInterface {
 public:
  inline View();
  inline View(const Vector3& position, const Vector3& target, const Vector3& up);

  /// Getters
  inline const Vector3& position()  const;
  inline const Vector3& direction() const;
  inline const Vector3& up()        const;

  inline const Matrix4x4& view_matrix() const;
  inline const Matrix4x4& view_matrix();

  inline bool is_dirty() const;

  /// Setters
  inline void set(const Vector3& position, const Vector3& target, const Vector3& up);
  inline void set_position(const Vector3& position);
  inline void set_direction(const Vector3& direction);
  inline void set_target(const Vector3& target);

 
 private:
  inline void build();

  Vector3 position_;
  Vector3 direction_;
  Vector3 up_;
  Matrix4x4 view_matrix_;
  bool bRebuild_;
};

}  // namespace aer

#include "aer/view/view-inl.h"

#endif  // AER_VIEW_VIEW_H_

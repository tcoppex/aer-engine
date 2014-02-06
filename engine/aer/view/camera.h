// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_VIEW_CAMERA_H_
#define AER_VIEW_CAMERA_H_

#include "aer/view/view.h"
#include "aer/view/frustum.h"


namespace aer {

/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
///
/// Describe a camera.
///
/// It is important to note that views, frustums & cameras
/// don't update their inner matrix when parameters are
/// change but instead wait for a matrix request and rebuild
/// them if needed.
/// const reference must thus be shared if their matrix is
/// built.
///
/// + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + ~ + 
class Camera : public View3DInterface {
 public:
  inline Camera(const View& view, const Frustum& frustum);

  virtual void update() {}

  /// Getters
  inline const Vector3& position()  const;
  inline const Vector3& direction() const;
  inline const Vector3& up()        const;

  inline const Matrix4x4& view_matrix() const;
  inline const Matrix4x4& view_matrix();

  inline const Matrix4x4& projection_matrix() const;
  inline const Matrix4x4& projection_matrix();

  inline const View&    view()    const;
  inline const Frustum& frustum() const;

  inline const Matrix4x4& view_projection_matrix() const;
  inline const Matrix4x4& view_projection_matrix();

  inline bool is_dirty() const;

  /// Setters
  inline void set_position(const Vector3& position);
  inline void set_direction(const Vector3& direction);
  inline void set_target(const Vector3& target);

  inline void set_view(const View& view);
  inline void set_frustum(const Frustum& frustum);

 protected:
  inline void build();

  View      view_;
  Frustum   frustum_;
  Matrix4x4 view_projection_matrix_;
  bool      bRebuild_;

};

}  // namespace aer

#include "aer/view/camera-inl.h"

#endif  // AER_VIEW_CAMERA_H_

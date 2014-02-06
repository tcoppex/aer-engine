// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------


#ifndef AER_VIEW_CAMERA_INL_H_
#define AER_VIEW_CAMERA_INL_H_

namespace aer {

Camera::Camera(const View& view, const Frustum& frustum) 
  : view_(view),
    frustum_(frustum),
    bRebuild_(true)
{}

const Vector3& Camera::position() const {
  return view_.position();
}

const Vector3& Camera::direction() const {
  return view_.direction();
}

const Vector3& Camera::up() const {
  return view_.up();
}

const Matrix4x4& Camera::view_matrix() const {
  AER_ASSERT(bRebuild_ || !view_.is_dirty());
  return view_.view_matrix();
}

const Matrix4x4& Camera::view_matrix() {
  bRebuild_ = bRebuild_ || view_.is_dirty();
  return view_.view_matrix();
}

const Matrix4x4& Camera::projection_matrix() const {
  AER_ASSERT(bRebuild_ || !frustum_.is_dirty());
  return frustum_.projection_matrix();
}

const Matrix4x4& Camera::projection_matrix() {
  bRebuild_ = bRebuild_ || frustum_.is_dirty();
  return frustum_.projection_matrix();
}

const View& Camera::view() const {
  return view_;
}

const Frustum& Camera::frustum() const {
  return frustum_;
}

const Matrix4x4& Camera::view_projection_matrix() const {
  AER_ASSERT(!is_dirty());
  return view_projection_matrix_;
}

const Matrix4x4& Camera::view_projection_matrix() {
  if (is_dirty()) {
    build();
  }
  return view_projection_matrix_;
}

bool Camera::is_dirty() const {
  return bRebuild_ || frustum_.is_dirty() || view_.is_dirty();
}

void Camera::set_position(const Vector3& position) {
  view_.set_position(position);
}

void Camera::set_direction(const Vector3& direction) {
  view_.set_direction(direction);
}

void Camera::set_target(const Vector3& target) {
  view_.set_target(target);
}

void Camera::set_view(const View& view) {
  view_ = view;
  bRebuild_ = true;
}

void Camera::set_frustum(const Frustum& frustum) {
  frustum_ = frustum;
  bRebuild_ = true;
}

void Camera::build() {
  view_projection_matrix_ = frustum_.projection_matrix() * view_.view_matrix();
  bRebuild_ = false;
}


}  // namespace aer

#endif  // AER_VIEW_CAMERA_INL_H_

// -----------------------------------------------------------------------------
// CreativeCommons BY-SA 3.0 2013 <Thibault Coppex>
//
// -----------------------------------------------------------------------------

#ifndef AER_VIEW_VIEW_INL_H_
#define AER_VIEW_VIEW_INL_H_

#include "glm/gtc/matrix_transform.hpp"  // for lookAt


namespace aer {

View::View() {
  set( Vector3(0.0f, 0.0f,  0.0f),
       Vector3(0.0f, 0.0f, -1.0f),
       Vector3(0.0f, 1.0f,  0.0f));
}

View::View(const Vector3& position, const Vector3& target, const Vector3& up) {
  set(position, target, up);
}

const Vector3& View::position() const {
  return position_;
}

const Vector3& View::direction() const {
  return direction_;
}

const Vector3& View::up() const {
  return up_;
}

const Matrix4x4& View::view_matrix() const {
  AER_ASSERT(!is_dirty());
  return view_matrix_;
}

const Matrix4x4& View::view_matrix() {
  if (is_dirty()) {
    build();
  }
  return view_matrix_;
}

bool View::is_dirty() const {
  return bRebuild_;
}


void View::set(const Vector3& position, const Vector3& target, const Vector3& up) {
  set_position(position);
  set_target(target);
  up_ = up;
  bRebuild_ = true;
}

void View::set_position(const Vector3& position) {
  position_ = position;
  bRebuild_ = true;
}

void View::set_direction(const Vector3& direction) {
  direction_  = direction;
  bRebuild_ = true;
}

void View::set_target(const Vector3& target) {
  direction_  = glm::normalize(target - position_);
  bRebuild_ = true;
}

void View::build() {
  view_matrix_ = glm::lookAt(position_, position_ + direction_, up_);
  bRebuild_    = false;
}

}  // namespace aer

#endif  // AER_VIEW_VIEW_INL_H_
